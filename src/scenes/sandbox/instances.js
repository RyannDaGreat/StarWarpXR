/**
 * Retained upstream instance buffer construction and transform history (no camera).
 * Bridges physics → rendering by building per-instance GPU data each frame.
 */

import { mat4, quat } from 'gl-matrix';

import { FLOATS_PER_INSTANCE, MAX_INSTANCES } from '../../render/layout.js';

// Last storage row belongs to the static terrain, not dynamic scene objects.
export const TERRAIN_INSTANCE_IDX = MAX_INSTANCES - 1;

export class SceneManager {
    /** Command. Allocate mutable instance storage and transform history. */
    constructor() {
        this.instanceData = new Float32Array(MAX_INSTANCES * FLOATS_PER_INSTANCE);
        this.prevTransforms = new Map();  // bodyId → mat4 (previous frame)
        this.numBoxInstances = 0;
        this.numSphereInstances = 0;
        this.numDominoInstances = 0;  // dominoes only (for separate draw call with beveled VB)
    }

    /**
     * Command. Build instance buffer; mutates instanceData and prevTransforms.
     *
     * @param {object} sceneData - from PhysicsWorld.getSceneData()
     * @returns {object} Counts and ordered {mesh, firstInstance, instanceCount} batches.
     * @example manager.buildInstances(sceneData).batches[0] // {mesh:'box', firstInstance:0, instanceCount:1}
     */
    buildInstances(sceneData) {
        const {
            floor, dominoes, dominoHalf, spheres, sphereRadius,
            mazeWalls, mazeChest, mazeIvy,
            towerPlatforms, towerRamps, towerFlag,
            mmStructure, mmWheels, mmChains,
            terrainBlocks, trees, shrubs, mushrooms,
            signposts, fence,
        } = sceneData;
        let boxIdx = 0;

        // Floor instance (box 0)
        const floorModel = this._makeModel(floor.pos, floor.rot, [floor.half.x, floor.half.y, floor.half.z]);
        this._writeInstance(boxIdx++, floorModel, 'floor', [0.35, 0.35, 0.4, 1.0]);

        // Domino instances
        for (let i = 0; i < dominoes.length; i++) {
            const d = dominoes[i];
            const model = this._makeModel(d.pos, d.rot, [dominoHalf.x, dominoHalf.y, dominoHalf.z]);
            this._writeInstance(boxIdx++, model, `domino_${i}`, [0.92, 0.90, 0.85, 1.0]);
        }

        this.numDominoInstances = dominoes.length;

        // Maze wall instances: gray-green stone color (detected in shader via color)
        if (mazeWalls) {
            for (let i = 0; i < mazeWalls.length; i++) {
                const w = mazeWalls[i];
                const model = this._makeModel(w.pos, w.rot, w.half);
                // Color signals "maze wall" to the shader: R=0.45, G=0.44, B=0.42, A=1
                this._writeInstance(boxIdx++, model, `maze_${i}`, [0.45, 0.44, 0.42, 1.0]);
            }
        }

        // Treasure chest instance: gold/brown color
        if (mazeChest) {
            const c = mazeChest;
            const model = this._makeModel(c.pos, c.rot, [c.half.x, c.half.y, c.half.z]);
            // Color signals "chest" to shader: warm gold R=0.85, G=0.65, B=0.2
            this._writeInstance(boxIdx++, model, 'maze_chest', [0.85, 0.65, 0.20, 1.0]);
        }

        // Maze ivy: render-only leaf clusters and vine tendrils (no physics body)
        if (mazeIvy) {
            for (let i = 0; i < mazeIvy.length; i++) {
                const iv = mazeIvy[i];
                const model = this._makeModel(iv.pos, iv.rot, iv.half);
                this._writeInstance(boxIdx++, model, `ivy_${i}`, iv.color);
            }
        }

        // Tower platform instances
        if (towerPlatforms) {
            for (let i = 0; i < towerPlatforms.length; i++) {
                const p = towerPlatforms[i];
                const model = this._makeModel(p.pos, p.rot, p.half);
                this._writeInstance(boxIdx++, model, `tower_plat_${i}`, p.color);
            }
        }

        // Tower ramp instances
        if (towerRamps) {
            for (let i = 0; i < towerRamps.length; i++) {
                const r = towerRamps[i];
                const model = this._makeModel(r.pos, r.rot, r.half);
                this._writeInstance(boxIdx++, model, `tower_ramp_${i}`, r.color);
            }
        }

        // Tower flag instances (pole + flag)
        if (towerFlag) {
            for (let i = 0; i < towerFlag.length; i++) {
                const f = towerFlag[i];
                const model = this._makeModel(f.pos, f.rot, f.half);
                this._writeInstance(boxIdx++, model, `tower_flag_${i}`, f.color);
            }
        }

        // Marble machine structure (fixed boxes: platforms, stairs, ramps, gutters)
        if (mmStructure) {
            for (let i = 0; i < mmStructure.length; i++) {
                const s = mmStructure[i];
                const model = this._makeModel(s.pos, s.rot, s.half);
                this._writeInstance(boxIdx++, model, `mm_struct_${i}`, s.color);
            }
        }

        // Marble machine spinning wheels (dynamic)
        if (mmWheels) {
            for (let i = 0; i < mmWheels.length; i++) {
                const w = mmWheels[i];
                const model = this._makeModel(w.pos, w.rot, w.half);
                this._writeInstance(boxIdx++, model, `mm_wheel_${i}`, w.color);
            }
        }

        // Marble machine dangling chains (dynamic)
        if (mmChains) {
            for (let i = 0; i < mmChains.length; i++) {
                const c = mmChains[i];
                const model = this._makeModel(c.pos, c.rot, c.half);
                this._writeInstance(boxIdx++, model, `mm_chain_${i}`, c.color);
            }
        }

        // Forest terrain columns
        if (terrainBlocks) {
            for (let i = 0; i < terrainBlocks.length; i++) {
                const b = terrainBlocks[i];
                const model = this._makeModel(b.pos, b.rot, b.half);
                this._writeInstance(boxIdx++, model, `terrain_${i}`, b.color);
            }
        }

        // Forest trees (trunks + canopies)
        if (trees) {
            for (let i = 0; i < trees.length; i++) {
                const t = trees[i];
                const model = this._makeModel(t.pos, t.rot, t.half);
                this._writeInstance(boxIdx++, model, `tree_${i}`, t.color);
            }
        }

        // Forest shrubs
        if (shrubs) {
            for (let i = 0; i < shrubs.length; i++) {
                const s = shrubs[i];
                const model = this._makeModel(s.pos, s.rot, s.half);
                this._writeInstance(boxIdx++, model, `shrub_${i}`, s.color);
            }
        }

        // Forest mushrooms
        if (mushrooms) {
            for (let i = 0; i < mushrooms.length; i++) {
                const m = mushrooms[i];
                const model = this._makeModel(m.pos, m.rot, m.half);
                this._writeInstance(boxIdx++, model, `mushroom_${i}`, m.color);
            }
        }

        // Area signposts (poles + boards)
        if (signposts) {
            for (let i = 0; i < signposts.length; i++) {
                const s = signposts[i];
                const model = this._makeModel(s.pos, s.rot, s.half);
                this._writeInstance(boxIdx++, model, `sign_${i}`, s.color);
            }
        }

        // Perimeter fence (render-only: posts + rails)
        if (fence) {
            for (let i = 0; i < fence.length; i++) {
                const f = fence[i];
                const model = this._makeModel(f.pos, f.rot, f.half);
                this._writeInstance(boxIdx++, model, `fence_${i}`, f.color);
            }
        }

        this.numBoxInstances = boxIdx;

        // Sphere instances (after all boxes)
        let sphereIdx = boxIdx;
        for (let i = 0; i < spheres.length; i++) {
            const s = spheres[i];
            const model = this._makeModel(s.pos, s.rot, [sphereRadius, sphereRadius, sphereRadius]);
            this._writeInstance(sphereIdx++, model, `sphere_${i}`, [0.2, 0.4, 0.95, 1.0]);
        }

        this.numSphereInstances = sphereIdx - boxIdx;

        return {
            numBoxInstances: this.numBoxInstances,
            numSphereInstances: this.numSphereInstances,
            numDominoInstances: this.numDominoInstances,
            batches: [
                { mesh: 'box', firstInstance: 0, instanceCount: 1 },
                { mesh: 'domino', firstInstance: 1, instanceCount: this.numDominoInstances },
                { mesh: 'box', firstInstance: 1 + this.numDominoInstances,
                    instanceCount: this.numBoxInstances - 1 - this.numDominoInstances },
                { mesh: 'sphere', firstInstance: this.numBoxInstances, instanceCount: this.numSphereInstances },
                { mesh: 'terrain', firstInstance: TERRAIN_INSTANCE_IDX, instanceCount: 1 },
            ].filter(batch => batch.instanceCount > 0),
        };
    }

    /**
     * Pure function. Position/quaternion/half-extents → column-major [16] model matrix.
     * Multiplies half-extents by 2 for unit primitives (half-extent/radius 0.5).
     * @param {object} pos - XYZ translation.
     * @param {object} rot - XYZW quaternion.
     * @param {number[]} halfExtents - XYZ half-extents [3].
     * @returns {Float32Array} Model matrix [16].
     * @example manager._makeModel({x:1,y:2,z:3}, {x:0,y:0,z:0,w:1}, [0.5,1,2])[12] // 1
     */
    _makeModel(pos, rot, halfExtents) {
        const m = mat4.create();
        const q = quat.fromValues(rot.x, rot.y, rot.z, rot.w);
        mat4.fromRotationTranslationScale(m, q, [pos.x, pos.y, pos.z],
            [halfExtents[0] * 2, halfExtents[1] * 2, halfExtents[2] * 2]);
        return m;
    }

    /**
     * Command. Write one row (model + prevModel + RGBA), updating transform history.
     * @param {number} idx - Instance row index.
     * @param {Float32Array} model - Column-major transform [16].
     * @param {string} id - Stable instance identity.
     * @param {number[]} color - RGBA [4], including retained material sentinels.
     * @example manager._writeInstance(0, mat4.create(), 'floor', [0.35,0.35,0.4,1]) // undefined
     */
    _writeInstance(idx, model, id, color) {
        if (idx >= TERRAIN_INSTANCE_IDX) throw new RangeError('Sandbox instances overlap reserved terrain slot');
        const offset = idx * FLOATS_PER_INSTANCE;

        // Current model
        this.instanceData.set(model, offset);

        // Previous model (or current if first frame)
        const prev = this.prevTransforms.get(id);
        if (prev) {
            this.instanceData.set(prev, offset + 16);
        } else {
            this.instanceData.set(model, offset + 16);
        }

        // Color
        this.instanceData[offset + 32] = color[0];
        this.instanceData[offset + 33] = color[1];
        this.instanceData[offset + 34] = color[2];
        this.instanceData[offset + 35] = color[3];

        // Store for next frame
        this.prevTransforms.set(id, new Float32Array(model));
    }

    /**
     * Query. Return a mutable view of active instance rows; reads manager state.
     * @returns {Float32Array} Flat [N,36]: model[16], prevModel[16], RGBA[4].
     * @example manager.getActiveData().length // 36 * (numBoxInstances + numSphereInstances)
     */
    getActiveData() {
        const total = this.numBoxInstances + this.numSphereInstances;
        return this.instanceData.subarray(0, total * FLOATS_PER_INSTANCE);
    }
}

