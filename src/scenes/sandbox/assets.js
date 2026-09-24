import { mat4 } from 'gl-matrix';
import { boxVertices, beveledBoxVertices, sphereVertices, terrainMeshVertices } from './geometry.js';
import { sceneWGSL, skyWGSL, shadowWGSL } from './shaders.js';
import { TERRAIN_INSTANCE_IDX } from './instances.js';
import { FLOATS_PER_INSTANCE } from '../../render/layout.js';

/**
 * Pure function. Create sandbox meshes, static terrain row and material/sky shaders.
 * Meshes are flat [N,6] Float32Arrays (XYZ position, XYZ normal); static rows are
 * [36] Float32Arrays (model[16], prevModel[16], RGBA[4]). No GPU resources are created.
 * @returns {object} {meshes: {key: Float32Array}, staticInstances: [{index, data}],
 *   shaders: {sceneWGSL, skyWGSL, shadowWGSL}} using the renderer's binding ABI.
 * @example createSandboxAssets().meshes.box.length // 216 (36 vertices)
 * @example createSandboxAssets().staticInstances[0].index // 32767
 */
export function createSandboxAssets() {
    const dominoHalf = [0.75, 1.5, 0.18];
    const dominoBevelRadius = 0.04, dominoBevelSegments = 2;
    const terrainGridSize = 600, terrainExtent = 4500;
    // Sentinel recognized by the sandbox fragment shader; identity preserves world geometry.
    const terrainColor = [0.12, 0.48, 0.08, 1.0];
    const identity = mat4.create();
    const data = new Float32Array(FLOATS_PER_INSTANCE);
    data.set(identity, 0);
    data.set(identity, 16);
    data.set(terrainColor, 32);
    return {
        meshes: {
            box: boxVertices(),
            domino: beveledBoxVertices(...dominoHalf, dominoBevelRadius, dominoBevelSegments),
            sphere: sphereVertices(),
            terrain: terrainMeshVertices(terrainGridSize, terrainExtent),
        },
        staticInstances: [{ index: TERRAIN_INSTANCE_IDX, data }],
        shaders: { sceneWGSL, skyWGSL, shadowWGSL },
    };
}
