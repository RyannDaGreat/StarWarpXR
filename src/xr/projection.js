import { mat4 } from 'gl-matrix';

/**
 * Pure function. Convert a WebGPU [0,1] clip-depth projection to GL [-1,1].
 * The scene/motion renderer expects GL matrices; its raster shader converts back once.
 * @param {ArrayLike<number>} projection - Column-major projection matrix (16,).
 * @returns {Float32Array} Independent GL projection matrix (16,).
 * @example toGLProjection(mat4.create())[14] // -1; z_gl = 2*z_gpu - w
 */
export function toGLProjection(projection) {
  const result = mat4.clone(projection);
  for (let column = 0; column < 4; column++) {
    result[column * 4 + 2] = 2 * projection[column * 4 + 2] - projection[column * 4 + 3];
  }
  return result;
}
