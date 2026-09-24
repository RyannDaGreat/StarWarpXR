import { mat4 } from 'gl-matrix';

const UNIFORM_BYTES = (16 + 4) * Float32Array.BYTES_PER_ELEMENT;
const SHADER = `
struct Uniforms { transform: mat4x4f, uvRegion: vec4f }
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var image: texture_2d<f32>;

// Query. Reads uniforms; unit quad index -> clip position and top-left UV.
// Example: identity transform, full UV region, index 0 -> (-.5,.5,0,1), (0,0).
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> Vertex {
  let positions = array<vec2f, 6>(
    vec2f(-.5, .5), vec2f(-.5, -.5), vec2f(.5, .5),
    vec2f(.5, .5), vec2f(-.5, -.5), vec2f(.5, -.5)
  );
  let position = positions[index];
  var result: Vertex;
  result.position = uniforms.transform * vec4f(position, 0, 1);
  result.uv = (vec2f(position.x, -position.y) + .5) * uniforms.uvRegion.zw + uniforms.uvRegion.xy;
  return result;
}

// Query. Samples RGBA at interpolated top-left UV; e.g. (0,0) reads the top row.
@fragment fn fragmentMain(vertex: Vertex) -> @location(0) vec4f {
  return textureSample(image, imageSampler, vertex.uv);
}
`;

/**
 * Command. Allocate native WebXR WebGPU presentation resources on a borrowed device.
 * Verified on a real WebKit GPU with simulated XR texture arrays; actual headset
 * compositor integration still requires hardware testing.
 * @param {GPUDevice} device - Renderer device from an xrCompatible adapter; never destroyed here.
 * @param {XRSession} session - Session requested with the webgpu feature; never ended here.
 * @returns {object} Owned projection layer and upload/present/destroy commands.
 * @example const presenter = createGPUPresenter(device, session); // presenter.layer is an XRProjectionLayer
 */
export function createGPUPresenter(device, session) {
  const binding = new XRGPUBinding(session, device);
  const colorFormat = binding.getPreferredColorFormat();
  const layer = binding.createProjectionLayer({ colorFormat });
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vertexMain' },
    fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: colorFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const fullscreen = mat4.fromScaling(mat4.create(), [2, 2, 1]);
  const buffers = [];
  let imageView, panelTexture, panelView;

  /**
   * Command. Encode one quad using storage unique to its draw within this submission.
   * @param {GPURenderPassEncoder} pass - Active eye pass.
   * @param {number} slot - Draw index, distinct across all eyes and panels.
   * @param {GPUTextureView} texture - Borrowed sampled view of (H,W,4) RGBA, e.g. 256×1200.
   * @param {Float32Array} transform - Column-major matrix (16,), unit quad to GPU clip space.
   * @param {number[]} uvRegion - (4,) [u,v,width,height] in top-left texture coordinates.
   * @example draw(pass, 0, imageView, fullscreen, [0,0,.5,1]); // undefined; encodes left SBS quad
   */
  function draw(pass, slot, texture, transform, uvRegion) {
    const buffer = buffers[slot] ??= device.createBuffer({
      size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const data = new Float32Array(UNIFORM_BYTES / Float32Array.BYTES_PER_ELEMENT);
    data.set(transform); data.set(uvRegion, 16);
    device.queue.writeBuffer(buffer, 0, data);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: texture },
      ],
    }));
    pass.draw(6);
  }

  return {
    layer,
    /**
     * Command. Borrow the SBS texture and copy the panel canvas without flipping its rows.
     * @param {GPUTexture} imageTexture - Sampleable (H,2W,4) RGBA SBS texture, e.g. 768×1536; not owned.
     * @param {HTMLCanvasElement|null} panelCanvas - Canvas2D (H,W,4) RGBA panel, e.g. 256×1200; null hides it.
     * @example presenter.upload(rendererTexture, controlsCanvas); // undefined; updates sampled textures
     */
    upload(imageTexture, panelCanvas) {
      imageView = imageTexture.createView();
      if (!panelCanvas) return;
      const { width, height } = panelCanvas;
      if (!panelTexture || panelTexture.width !== width || panelTexture.height !== height) {
        panelTexture?.destroy();
        panelTexture = device.createTexture({
          size: [width, height], format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        panelView = panelTexture.createView();
      }
      device.queue.copyExternalImageToTexture(
        { source: panelCanvas, flipY: false }, { texture: panelTexture }, [width, height],
      );
    },
    /**
     * Command. Submit each eye's SBS half and spatial panel directly to native XR textures.
     * @param {XRProjectionLayer} layer - Destination projection layer.
     * @param {object[]} views - {xrView, eye, gpuViewProj}; matrix (16,) uses native Z in [0,1].
     * @param {Float32Array|null} panelModel - Column-major unit-panel world transform (16,); null skips the panel.
     * @example presenter.present(presenter.layer, views, panelModel); // undefined; submits stereo frame
     */
    present(layer, views, panelModel) {
      const encoder = device.createCommandEncoder();
      let slot = 0;
      for (const view of views) {
        const subimage = binding.getViewSubImage(layer, view.xrView);
        const pass = encoder.beginRenderPass({ colorAttachments: [{
          view: subimage.colorTexture.createView(subimage.getViewDescriptor()),
          // A clear can erase the other eye when both views share an atlas layer.
          loadOp: 'load', storeOp: 'store',
        }] });
        const { x, y, width, height } = subimage.viewport;
        pass.setViewport(x, y, width, height, 0, 1);
        pass.setPipeline(pipeline);
        draw(pass, slot++, imageView, fullscreen, [view.eye === 'right' ? .5 : 0, 0, .5, 1]);
        if (panelModel) draw(pass, slot++, panelView, mat4.multiply(mat4.create(), view.gpuViewProj, panelModel), [0, 0, 1, 1]);
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
    },
    /**
     * Command. Destroy owned buffers, panel texture and layer; leave borrowed device/textures intact.
     * @example presenter.destroy(); // undefined; releases presenter-owned GPU allocations
     */
    destroy() {
      for (const buffer of buffers) buffer.destroy();
      panelTexture?.destroy();
      layer.destroy();
    },
  };
}
