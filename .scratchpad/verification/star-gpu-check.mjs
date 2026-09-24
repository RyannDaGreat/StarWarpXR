import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';
const browser = await puppeteer.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  page.on('console', message => console.log('BROWSER', message.type(), message.text()));
  await page.goto('http://localhost:5173/.scratchpad/verification/gpu-check.html');
  const result = await page.evaluate(async () => {
    const { StarWarp, createStarState, destroyStarState } = await import('/src/stars/star-warp.js');
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 9 } });
    const errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const width = 32, height = 32, count = 1024;
    const warp = new StarWarp(device, width, height);
    const previous = createStarState(device, width, height, 777);
    const next = createStarState(device, width, height, 777);
    const texture = device.createTexture({ size: [width,height], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, new Float32Array(width*height*4), { bytesPerRow: width*16 }, [width,height]);
    const view = texture.createView();
    const fields = { temporalL:view, temporalR:view, crossL:view, crossR:view };
    async function read(buffer, bytes) {
      const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer,0,staging,0,bytes); device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const result = new Uint8Array(staging.getMappedRange()).slice(); staging.unmap(); staging.destroy();
      return Array.from(result);
    }
    const initial = await read(previous.posL, count*8);
    const initialMeta = await read(previous.metaL, count*8);
    let encoder = device.createCommandEncoder();
    warp.encode(encoder, previous, next, fields, 123, false, { numStars:count });
    device.queue.submit([encoder.finish()]);
    const mono = await read(next.posL, count*8);
    const monoMeta = await read(next.metaL, count*8);
    const unchanged = await read(previous.posL, count*8);
    encoder = device.createCommandEncoder();
    warp.encode(encoder, previous, next, fields, 123, true, { numStars:count });
    device.queue.submit([encoder.finish()]);
    const maskBytes = await read(next.mask, (2*count+4)*4);
    const masks = new Uint32Array(new Uint8Array(maskBytes).buffer);
    let left=0, right=0, shared=0;
    for (let i=0; i<2*count; i++) { if (masks[i]&1) left++; if (masks[i]&2) right++; if (masks[i]===3) shared++; }
    const counters = Array.from(masks.slice(2*count,2*count+3));
    const finalPrevious = await read(previous.posL, count*8);
    destroyStarState(previous); destroyStarState(next); warp.destroy(); texture.destroy(); device.destroy();
    return { errors, unchanged: JSON.stringify(initial)===JSON.stringify(unchanged) && JSON.stringify(initial)===JSON.stringify(finalPrevious), monoMatches: JSON.stringify(initial)===JSON.stringify(mono), metaMatches: JSON.stringify(initialMeta)===JSON.stringify(monoMeta), counters, observed:[left,right,shared] };
  });
  assert.deepEqual(result.errors, []);
  assert.equal(result.unchanged, true);
  assert.equal(result.monoMatches, true);
  assert.equal(result.metaMatches, true);
  assert.deepEqual(result.counters, result.observed);
  assert.ok(result.counters[0] > 0 && result.counters[0] < 2048);
  assert.equal(result.counters[0], result.counters[1]);
  assert.equal(result.counters[0], result.counters[2]);
  console.log('PASS: real WGSL zero-flow identity, metadata, immutable previous state, stereo masks and counters', result);
} finally { await browser.close(); }
