import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mat4 } from 'gl-matrix';
import { createGPUPresenter } from '../src/xr/gpu-presenter.js';

/**
 * Command. Build a recording GPU mock; no shader execution or native WebXR validation.
 * @param {import('node:test').TestContext} t - Test-owned mock tracker.
 * @returns {object} Device plus recorded buffers, textures, passes, uniforms and binding calls.
 * @example const gpu = mockGPU(t); // gpu.buffers starts empty
 */
function mockGPU(t) {
  const buffers = [], textures = [], passes = [], writes = [];
  const layer = { destroy: t.mock.fn() };
  const pipeline = { getBindGroupLayout: t.mock.fn(() => ({})) };
  const device = {
    destroy: t.mock.fn(),
    createShaderModule: t.mock.fn(),
    createRenderPipeline: t.mock.fn(() => pipeline),
    createSampler: t.mock.fn(() => ({})),
    /** Command. Record an owned buffer allocation and destruction spy. */
    createBuffer(descriptor) {
      const buffer = { descriptor, destroy: t.mock.fn() };
      buffers.push(buffer);
      return buffer;
    },
    /** Command. Record an owned panel texture allocation. */
    createTexture(descriptor) {
      const texture = {
        descriptor, width: descriptor.size[0], height: descriptor.size[1],
        createView: t.mock.fn(() => ({ texture })), destroy: t.mock.fn(),
      };
      textures.push(texture);
      return texture;
    },
    createBindGroup: t.mock.fn(descriptor => descriptor),
    /** Command. Record render passes; deliberately omit scissor and vertex-buffer APIs. */
    createCommandEncoder() {
      return {
        /** Command. Record a render pass and its draw commands. */
        beginRenderPass(descriptor) {
          const pass = {
            descriptor, setViewport: t.mock.fn(), setPipeline: t.mock.fn(),
            setBindGroup: t.mock.fn(), draw: t.mock.fn(), end: t.mock.fn(),
          };
          passes.push(pass);
          return pass;
        },
        finish: t.mock.fn(() => ({ passes })),
      };
    },
    queue: {
      /** Command. Snapshot uniforms at write time, as the real queue does. */
      writeBuffer(buffer, offset, data) { writes.push({ buffer, offset, data: Array.from(data) }); },
      copyExternalImageToTexture: t.mock.fn(), submit: t.mock.fn(),
    },
  };
  const binding = {
    getPreferredColorFormat: t.mock.fn(() => 'bgra8unorm'),
    createProjectionLayer: t.mock.fn(() => layer),
    getViewSubImage: t.mock.fn((target, xrView) => xrView.subimage),
  };
  return { device, layer, pipeline, binding, buffers, textures, passes, writes };
}

/** Command. Verify native stereo presentation, per-draw data isolation, resize and ownership. */
test('native GPU presenter keeps independent eye/panel uniforms and owns only its allocations', t => {
  const gpu = mockGPU(t);
  const constructors = [];
  const globals = {
    GPUBufferUsage: { COPY_DST: 8, UNIFORM: 64 },
    GPUTextureUsage: { COPY_DST: 2, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 },
    /** Command. Record native binding construction with the borrowed session/device. */
    XRGPUBinding: function (session, device) {
      constructors.push([session, device]);
      return gpu.binding;
    },
  };
  for (const [name, value] of Object.entries(globals)) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    t.after(() => {
      if (original) Object.defineProperty(globalThis, name, original);
      else delete globalThis[name];
    });
  }
  const session = { end: t.mock.fn() };
  const presenter = createGPUPresenter(gpu.device, session);
  assert.deepEqual(constructors, [[session, gpu.device]]);
  assert.equal(presenter.layer, gpu.layer);
  assert.deepEqual(gpu.binding.createProjectionLayer.mock.calls[0].arguments, [{ colorFormat: 'bgra8unorm' }]);
  const pipeline = gpu.device.createRenderPipeline.mock.calls[0].arguments[0];
  assert.deepEqual(pipeline.fragment.targets, [{ format: 'bgra8unorm' }]);
  assert.equal(pipeline.vertex.buffers, undefined);

  const imageView = {};
  const image = { createView: t.mock.fn(() => imageView), destroy: t.mock.fn() };
  const nextImage = { createView: t.mock.fn(() => imageView), destroy: t.mock.fn() };
  const panel = { width: 1200, height: 256 };
  presenter.upload(image, panel);
  const firstPanel = gpu.textures[0];
  assert.deepEqual(firstPanel.descriptor, {
    size: [1200, 256], format: 'rgba8unorm', usage: 2 | 4 | 16,
  });
  assert.deepEqual(gpu.device.queue.copyExternalImageToTexture.mock.calls[0].arguments, [
    { source: panel, flipY: false }, { texture: firstPanel }, [1200, 256],
  ]);
  presenter.upload(nextImage, panel);
  assert.equal(gpu.textures.length, 1);
  assert.equal(firstPanel.destroy.mock.callCount(), 0);
  presenter.upload(image, { width: 600, height: 256 });
  assert.equal(firstPanel.destroy.mock.callCount(), 1);
  presenter.upload(image, { width: 600, height: 128 });
  assert.equal(gpu.textures[1].destroy.mock.callCount(), 1);
  assert.equal(gpu.textures.length, 3);

  // Reversed eye order catches index-based SBS selection. Same target catches atlas clears.
  const target = { createView: t.mock.fn(descriptor => ({ descriptor })), destroy: t.mock.fn() };
  const views = ['right', 'left'].map((eye, index) => {
    const descriptor = { dimension: '2d', baseArrayLayer: 1 - index, arrayLayerCount: 1 };
    const viewport = { x: 30 + index * 700, y: 20, width: 650, height: 800 };
    const gpuViewProj = mat4.perspectiveZO(mat4.create(), Math.PI / 2, 1, .1, 100);
    mat4.translate(gpuViewProj, gpuViewProj, [index ? -.03 : .03, .2, -.5]);
    return {
      eye, gpuViewProj,
      xrView: { subimage: { colorTexture: target, viewport, getViewDescriptor: t.mock.fn(() => descriptor) } },
    };
  });
  const panelModel = mat4.fromTranslation(mat4.create(), [.2, -.5, -2]);
  mat4.rotateY(panelModel, panelModel, .4);
  mat4.scale(panelModel, panelModel, [1.4, .3, 1]);
  presenter.present(presenter.layer, views, panelModel);
  assert.equal(gpu.passes.length, 2);
  assert.equal(gpu.buffers.length, 4);
  assert.equal(new Set(gpu.writes.map(write => write.buffer)).size, 4);
  const fullscreen = Array.from(mat4.fromScaling(mat4.create(), [2, 2, 1]));
  for (const [index, view] of views.entries()) {
    const pass = gpu.passes[index];
    const subimage = view.xrView.subimage;
    assert.deepEqual(gpu.binding.getViewSubImage.mock.calls[index].arguments, [gpu.layer, view.xrView]);
    assert.equal(target.createView.mock.calls[index].arguments[0], subimage.getViewDescriptor.mock.calls[0].result);
    assert.deepEqual(pass.descriptor.colorAttachments[0], {
      view: target.createView.mock.calls[index].result, loadOp: 'load', storeOp: 'store',
    });
    assert.deepEqual(pass.setViewport.mock.calls[0].arguments, [subimage.viewport.x, 20, 650, 800, 0, 1]);
    assert.equal(pass.setPipeline.mock.calls[0].arguments[0], gpu.pipeline);
    assert.deepEqual(pass.draw.mock.calls.map(call => call.arguments), [[6], [6]]);
    assert.equal(pass.end.mock.callCount(), 1);
    const eyeWrite = gpu.writes[index * 2], panelWrite = gpu.writes[index * 2 + 1];
    assert.deepEqual(eyeWrite.data, [...fullscreen, index === 0 ? .5 : 0, 0, .5, 1]);
    assert.deepEqual(panelWrite.data, [...mat4.multiply(mat4.create(), view.gpuViewProj, panelModel), 0, 0, 1, 1]);
    for (const [drawIndex, write] of [eyeWrite, panelWrite].entries()) {
      assert.equal(write.offset, 0);
      assert.deepEqual(write.buffer.descriptor, { size: 80, usage: 8 | 64 });
      const group = pass.setBindGroup.mock.calls[drawIndex].arguments[1];
      assert.equal(group.entries[0].resource.buffer, write.buffer);
      assert.equal(group.entries[2].resource, drawIndex ? gpu.textures[2].createView.mock.calls[0].result : imageView);
    }
  }
  assert.equal(gpu.device.queue.submit.mock.callCount(), 1);
  assert.notDeepEqual(gpu.writes[1].data, gpu.writes[3].data);
  presenter.present(presenter.layer, views, panelModel);
  assert.equal(gpu.buffers.length, 4, 'reuse storage after the preceding submission');
  presenter.destroy();
  for (const buffer of gpu.buffers) assert.equal(buffer.destroy.mock.callCount(), 1);
  for (const texture of gpu.textures) assert.equal(texture.destroy.mock.callCount(), 1);
  assert.equal(gpu.layer.destroy.mock.callCount(), 1);
  for (const borrowed of [image, nextImage, target, gpu.device]) assert.equal(borrowed.destroy.mock.callCount(), 0);
  assert.equal(session.end.mock.callCount(), 0);
});

/** Command. Guard architectural boundaries and top-left shader mapping pending real GPU verification. */
test('GPU presenter stays scene-independent with top-left UVs and an iterable view loop', () => {
  const source = readFileSync(new URL('../src/xr/gpu-presenter.js', import.meta.url), 'utf8');
  assert.deepEqual(source.match(/^import .*$/gm), ["import { mat4 } from 'gl-matrix';"]);
  assert.doesNotMatch(source, /setScissorRect|webgl|\bviewProj\b|for\s*\([^)]*\bin\s+views/i);
  assert.match(source, /for \(const view of views\)/);
  assert.match(source, /vec2f\(position\.x, -position\.y\) \+ \.5/);
  assert.match(source, /array<vec2f, 6>/);
  assert.match(source, /@builtin\(vertex_index\)/);
});
