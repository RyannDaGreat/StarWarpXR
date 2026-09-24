import { mat4 } from 'gl-matrix';

const VERTEX = `#version 300 es
in vec2 position;
uniform mat4 transform;
uniform vec4 uvRegion;
out vec2 uv;
// Command. Emits a transformed unit quad vertex and its texture coordinate.
void main() {
  uv = (position + 0.5) * uvRegion.zw + uvRegion.xy;
  gl_Position = transform * vec4(position, 0.0, 1.0);
}`;
const FRAGMENT = `#version 300 es
precision mediump float;
uniform sampler2D image;
in vec2 uv;
out vec4 color;
// Command. Writes the sampled presentation pixel.
void main() { color = texture(image, uv); }
`;

/**
 * Command. Compile a WebGL shader, reporting its full compiler error.
 * @param {WebGL2RenderingContext} gl - Destination context.
 * @param {number} type - Shader type.
 * @param {string} source - GLSL source.
 * @returns {WebGLShader} Compiled shader.
 */
function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

/**
 * Command. Create WebGL presentation resources; no knowledge of scenes or stars.
 * Copies a WebGPU canvas into a WebXR WebGL layer once per frame. This is a
 * compatibility bridge, not zero-copy WebGPU/XR interop; profile it on hardware.
 * @param {HTMLCanvasElement} canvas - Dedicated WebGL canvas, separate from WebGPU.
 * @returns {object} Presenter commands and context.
 */
export function createPresenter(canvas) {
  const gl = canvas.getContext('webgl2', { xrCompatible: true, alpha: false, antialias: false });
  if (!gl) throw new Error('WebXR presentation requires WebGL 2');
  const program = gl.createProgram();
  const shaders = [compile(gl, gl.VERTEX_SHADER, VERTEX), compile(gl, gl.FRAGMENT_SHADER, FRAGMENT)];
  for (const shader of shaders) gl.attachShader(program, shader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const vertices = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-.5,-.5, .5,-.5, -.5,.5, -.5,.5, .5,-.5, .5,.5]), gl.STATIC_DRAW);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const location = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
  let validated = false;
  const transform = gl.getUniformLocation(program, 'transform');
  const region = gl.getUniformLocation(program, 'uvRegion');
  const fullscreen = mat4.fromScaling(mat4.create(), [2, 2, 1]);
  const textures = [gl.createTexture(), gl.createTexture()];
  for (const texture of textures) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Command. Upload a canvas into the selected texture; flips DOM top-left to GL UVs. */
  function upload(source, index) {
    gl.bindTexture(gl.TEXTURE_2D, textures[index]);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  return {
    gl,
    /** Command. Upload rendered SBS image and a small control atlas to the GPU. */
    upload(image, panel) { upload(image, 0); upload(panel, 1); },
    /** Command. Render each eye into its runtime-provided viewport, with a spatial panel. */
    present(layer, views, panelModel) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program); gl.bindVertexArray(vao); gl.activeTexture(gl.TEXTURE0);
      for (const view of views) {
        const viewport = layer.getViewport(view.xrView);
        gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
        gl.bindTexture(gl.TEXTURE_2D, textures[0]);
        gl.uniformMatrix4fv(transform, false, fullscreen);
        // The renderer writes left then right, regardless of runtime viewport ordering.
        gl.uniform4f(region, view.eye === 'right' ? .5 : 0, 0, .5, 1);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindTexture(gl.TEXTURE_2D, textures[1]);
        gl.uniformMatrix4fv(transform, false, mat4.multiply(mat4.create(), view.viewProj, panelModel));
        gl.uniform4f(region, 0, 0, 1, 1);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
      if (!validated) {
        const error = gl.getError();
        if (error !== gl.NO_ERROR) throw new Error(`WebGPU → WebXR presentation failed: WebGL error ${error}`);
        validated = true;
      }
    },
    /** Command. Free this presenter's GPU resources. */
    destroy() {
      for (const texture of textures) gl.deleteTexture(texture);
      for (const shader of shaders) gl.deleteShader(shader);
      gl.deleteBuffer(vertices); gl.deleteVertexArray(vao); gl.deleteProgram(program);
    },
  };
}
