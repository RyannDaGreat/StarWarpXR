import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';
const browser = await puppeteer.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  page.on('console', message => console.log('BROWSER', message.type(), message.text()));
  page.on('pageerror', error => console.error(error));
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle0' });
  const result = await page.evaluate(async () => {
    const { createPresenter } = await import('/src/xr/presenter.js');
    const { mat4 } = await import('/node_modules/.vite/deps/gl-matrix.js');
    const source = document.createElement('canvas'); source.width = 4; source.height = 4;
    const device = await (await navigator.gpu.requestAdapter()).requestDevice();
    const context = source.getContext('webgpu');
    context.configure({ device, format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST });
    const pixels = new Uint8Array(4 * 4 * 4);
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
      const color = y < 2 ? (x < 2 ? [255,0,0,255] : [0,255,0,255]) : (x < 2 ? [0,0,255,255] : [255,255,0,255]);
      pixels.set(color, (y * 4 + x) * 4);
    }
    device.queue.writeTexture({ texture: context.getCurrentTexture() }, pixels, { bytesPerRow: 16 }, [4,4]);
    const canvas = document.createElement('canvas'); canvas.width = 4; canvas.height = 4;
    const presenter = createPresenter(canvas);
    const panel = document.createElement('canvas'); panel.width = 2; panel.height = 2;
    presenter.upload(source, panel);
    const views = ['left','right'].map(eye => ({ eye, xrView: { eye }, viewProj: mat4.create() }));
    presenter.present({ framebuffer: null, getViewport: view => ({ x: view.eye === 'left' ? 0 : 2, y: 0, width: 2, height: 4 }) }, views, mat4.fromTranslation(mat4.create(), [100,0,0]));
    const actual = new Uint8Array(pixels.length);
    presenter.gl.readPixels(0,0,4,4,presenter.gl.RGBA,presenter.gl.UNSIGNED_BYTE,actual);
    const error = presenter.gl.getError();
    const realGetError = presenter.gl.getError.bind(presenter.gl);
    presenter.gl.getError = () => presenter.gl.INVALID_OPERATION;
    let laterFailure;
    try { presenter.upload(source, panel); }
    catch (problem) { laterFailure = problem.message; console.log('Expected injected later-frame error:', laterFailure); }
    presenter.gl.getError = realGetError;
    presenter.destroy(); device.destroy();
    return { actual: Array.from(actual), error, laterFailure };
  });
  assert.equal(result.error, 0);
  assert.match(result.laterFailure, /WebGPU canvas upload failed: 1282/);
  assert.deepEqual(result.actual.slice(0,4), [0,0,255,255], 'GL bottom-left must be source blue lower-left');
  assert.deepEqual(result.actual.slice(8,12), [255,255,0,255], 'GL bottom-right must be source yellow lower-right');
  assert.deepEqual(result.actual.slice(48,52), [255,0,0,255], 'GL top-left must be source red upper-left');
  assert.deepEqual(result.actual.slice(56,60), [0,255,0,255], 'GL top-right must be source green upper-right');
  console.log('PASS: real WebGPU canvas → WebGL SBS copy; eye assignment, orientation, RGBA verified');
} finally { await browser.close(); }
