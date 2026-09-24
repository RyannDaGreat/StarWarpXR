/**
 * Pure function. Fullscreen quad for display pass.
 * @returns {Float32Array} Flat [6,4] rows (clip X,Y, texture U,V).
 * @example quadVertices().length // 24
 * WebGPU convention: v=0 at top, v=1 at bottom.
 *
 * >>> quadVertices().length
 * 24
 */
export function quadVertices() {
    return new Float32Array([
        -1, -1,  0, 1,
         1, -1,  1, 1,
         1,  1,  1, 0,
        -1, -1,  0, 1,
         1,  1,  1, 0,
        -1,  1,  0, 0,
    ]);
}
