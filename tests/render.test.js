import test from 'node:test';
import assert from 'node:assert/strict';
import { mat4, vec3, vec4 } from 'gl-matrix';
import { WebGPURenderer, buildLightSpaceMatrix } from '../src/render/renderer.js';
import { StarDraw } from '../src/render/star-draw.js';
import { displayWGSL } from '../src/render/shaders.js';
import { sceneWGSL, skyWGSL, shadowWGSL } from '../src/scenes/sandbox/shaders.js';
import { boxVertices, beveledBoxVertices, sphereVertices, terrainMeshVertices, terrainHeight } from '../src/scenes/sandbox/geometry.js';
import { SceneManager, TERRAIN_INSTANCE_IDX } from '../src/scenes/sandbox/instances.js';
import { createSandboxAssets } from '../src/scenes/sandbox/assets.js';
import { FLOATS_PER_INSTANCE, MAX_INSTANCES } from '../src/render/layout.js';

/**
 * Pure function. Assert each numeric component agrees within tolerance; throws on mismatch.
 * @param {ArrayLike<number>} actual - Computed vector/matrix.
 * @param {ArrayLike<number>} expected - Reference of the same length.
 * @param {number} tolerance - Maximum absolute error.
 * @example close([0.5], [0.500001]) // undefined
 */
function close(actual, expected, tolerance = 1e-5) {
    assert.equal(actual.length, expected.length);
    actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < tolerance, `${i}: ${value} != ${expected[i]}`));
}

/**
 * Command. Create a GPU command recorder; captures writes and render draw calls, no rasterization.
 * @returns {object} Fake device plus its mutable records.
 * @example recordingDevice().passes.length // 0
 */
function recordingDevice() {
    const writes = [], passes = [];
    const device = {
        queue: {
            writeBuffer(buffer, offset, data) {
                const bytes = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
                writes.push({ buffer, offset, bytes: bytes.slice(0, 304) });
            },
            submit() {},
        },
        createBuffer() { return { destroyed: false, destroy() { this.destroyed = true; } }; },
        createBindGroup(desc) { return desc; },
        createCommandEncoder() {
            return {
                beginRenderPass(desc) {
                    const pass = { desc, draws: [], vertices: [], setPipeline() {}, setBindGroup() {}, setVertexBuffer(slot, buffer) { this.vertices.push(buffer); }, draw(...args) { this.draws.push(args); }, end() {} };
                    passes.push(pass);
                    return pass;
                },
                finish() { return {}; },
            };
        },
    };
    return { device, writes, passes };
}

/**
 * Command. Construct a renderer with fake GPU attachments for frame-contract tests.
 * @returns {object} Renderer, captured commands, and compute/raster invocation records.
 * @example fixture().renderer.W // 32
 */
function fixture() {
    const records = recordingDevice();
    const assets = { meshes: { custom: new Float32Array(18) }, staticInstances: [], shaders: { sceneWGSL, skyWGSL, shadowWGSL } };
    const renderer = new WebGPURenderer({}, 32, 24, assets);
    renderer.meshes = { custom: { buffer: 'customVB', vertexCount: 3 } };
    renderer.quadVertCount = 6;
    renderer.device = records.device;
    renderer.ctx = { getCurrentTexture: () => ({ createView: () => 'canvas' }) };
    for (const name of ['cameraUniformBuf', 'cameraUniformBufR', 'skyUniformBuf', 'skyUniformBufR', 'shadowUniformBuf', 'lightUniformBuf', 'displayUniformBuf', 'instanceBuf', 'colorTexView', 'colorTexRView', 'motionTexView', 'motionTexRView', 'crossTexLView', 'crossTexRView']) renderer[name] = name;
    renderer.starStates = [{ name: 'previous' }, { name: 'output' }];
    const compute = [], raster = [];
    renderer.numStars = 100;
    renderer.stars = { encode: (...args) => compute.push(args) };
    renderer.starDraw = { encode: (...args) => raster.push(args) };
    return { renderer, ...records, compute, raster };
}

/**
 * Pure function. Build a finite GL perspective frame with identity instance transforms.
 * @returns {object} Mono frame options (matrices are column-major [16]).
 * @example frameOptions().displayMode // 1
 */
function frameOptions() {
    const vp = mat4.perspectiveNO(mat4.create(), Math.PI / 3, 4 / 3, 0.1, 100);
    return { viewProj: vp, prevViewProj: vp, invViewProj: mat4.invert(mat4.create(), vp), instanceData: new Float32Array(), batches: [{ mesh: 'custom', firstInstance: 7, instanceCount: 2 }], displayMode: 1, frameSeed: 42 };
}

test('GL camera depth converts once; sky unprojects GL near/far; shadows already use ZO', () => {
    assert.match(sceneWGSL, /\(out.currClip.z \+ out.currClip.w\) \* 0.5/);
    assert.match(skyWGSL, /nearClip\s*= vec4f\(in.clipXY, -1.0, 1.0\)/);
    assert.match(skyWGSL, /farClip\s*= vec4f\(in.clipXY, 1.0, 1.0\)/);
    assert.match(shadowWGSL, /return shadow.lightSpaceMatrix \* wp;/);
    const { viewProj, invViewProj } = frameOptions();
    for (const [z, expected] of [[-0.1, 0], [-100, 1]]) {
        const clip = vec4.transformMat4([], [0,0,z,1], viewProj);
        close([(clip[2] + clip[3]) / (2 * clip[3])], [expected]);
    }
    const near = vec3.transformMat4([], [0.3,0.2,-1], invViewProj);
    const far = vec3.transformMat4([], [0.3,0.2,1], invViewProj);
    close(vec3.normalize([], vec3.subtract([], far, near)), vec3.normalize([], near));
});

test('shadow depth covers origin and sun-facing winding survives back-face culling', () => {
    for (const sun of [[0,1,0], [0.6,0.8,0]]) {
        const matrix = buildLightSpaceMatrix(sun, 200, 300);
        close(vec3.transformMat4([], [0,0,0], matrix), [0,0,0.5]);
        close([vec3.transformMat4([], vec3.scale([], sun, 150), matrix)[2]], [0]);
        close([vec3.transformMat4([], vec3.scale([], sun, -150), matrix)[2]], [1]);
        const tangent = vec3.normalize([], vec3.cross([], [0,0,1], sun));
        const bitangent = vec3.cross([], sun, tangent);
        const a = vec3.transformMat4([], tangent, matrix), b = vec3.transformMat4([], bitangent, matrix);
        assert.ok(a[0] * b[1] - a[1] * b[0] > 0, 'sun-facing triangle must be CCW');
    }
});

test('scene skips compute; stereo uploads actual eyes and independent inverse matrices', () => {
    const { renderer, writes, compute, passes } = fixture();
    const opts = frameOptions();
    const left = mat4.translate(mat4.create(), opts.viewProj, [0.03,0,0]);
    const right = mat4.translate(mat4.create(), opts.viewProj, [-0.03,0,0]);
    renderer.frame({ ...opts, stereo: { mode: 2, viewProjL: left, viewProjR: right, prevViewProjL: left, prevViewProjR: right } });
    assert.equal(renderer.canvas.width, 64);
    assert.equal(compute.length, 0);
    for (const [name, expected] of [['cameraUniformBuf', left], ['cameraUniformBufR', right], ['skyUniformBuf', mat4.invert(mat4.create(), left)], ['skyUniformBufR', mat4.invert(mat4.create(), right)]]) {
        close(new Float32Array(writes.find(write => write.buffer === name).bytes.buffer).slice(0,16), expected);
    }
    assert.equal(passes.length, 4); // shadow, left, right, display
    assert.match(displayWGSL, /if \(right\) \{ return textureLoad\(sceneR,p,0\); \}/);
});

test('disabled and nighttime shadows clear to fully lit depth without drawing casters', () => {
    for (const settings of [{ shadowsEnabled: false, elapsedSecs: 0 }, { shadowsEnabled: true, elapsedSecs: 750 }]) {
        const { renderer, passes } = fixture();
        renderer.shadowsEnabled = settings.shadowsEnabled;
        renderer.frame({ ...frameOptions(), elapsedSecs: settings.elapsedSecs });
        assert.equal(passes[0].desc.depthStencilAttachment.depthClearValue, 1);
        assert.deepEqual(passes[0].draws, []);
    }
});

test('stars consume explicit previous/output state then swap; scene never advances state', () => {
    const { renderer, compute, raster } = fixture();
    const [previous, output] = renderer.starStates;
    renderer.frame({ ...frameOptions(), displayMode: 6 });
    assert.equal(compute[0][1], previous);
    assert.equal(compute[0][2], output);
    assert.equal(compute[0][4], 42);
    assert.equal(compute[0][5], false);
    assert.deepEqual(compute[0][6], { numStars: 100 });
    assert.equal(raster[0][1], output);
    assert.equal(raster[0][2], 100);
    assert.deepEqual(renderer.starStates, [output, previous]);
    renderer.frame(frameOptions());
    assert.deepEqual(renderer.starStates, [output, previous]);
});

test('resetHistory reseeds explicit buffers deterministically without touching StarDraw settings', () => {
    globalThis.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 };
    const { renderer, writes } = fixture();
    const draw = renderer.starDraw;
    Object.assign(draw, { starAAEnabled: false, starColorQEnabled: true, starSizeQEnabled: true, starSizeMaxPx: 13, cullOrphansEnabled: true });
    renderer.starStates = [];
    renderer.resetHistory(123);
    const old = renderer.starStates;
    const initial = writes[0].bytes;
    writes.length = 0;
    renderer.resetHistory(123);
    assert.deepEqual(writes[0].bytes, initial);
    assert.ok(old.every(state => Object.values(state).every(buffer => buffer.destroyed)));
    assert.notEqual(renderer.starStates[0], renderer.starStates[1]);
    assert.equal(renderer.starDraw, draw);
    assert.equal(renderer.starDraw.starSizeMaxPx, 13);
    assert.equal(renderer.numStars, 100);
    assert.equal(renderer.frameCount, 0);
    writes.length = 0;
    renderer.resetHistory(456);
    assert.notDeepEqual(writes[0].bytes, initial);
});

test('unsupported modes and recorded device failures throw rather than rendering silently', () => {
    const { renderer } = fixture();
    assert.throws(() => renderer.frame({ ...frameOptions(), displayMode: 0 }), /Only display modes/);
    assert.throws(() => renderer.frame({ ...frameOptions(), stereo: { mode: 1 } }), /stereo.mode=2/);
    renderer.gpuError = new Error('device lost test');
    assert.throws(() => renderer.frame(frameOptions()), /device lost test/);
    assert.throws(() => renderer.resetHistory(), /device lost test/);
});

test('retained geometry layouts and instance previous transforms', () => {
    assert.equal(boxVertices().length, 216);
    assert.equal(beveledBoxVertices(0.75,1.5,0.18,0.04,2).length % 18, 0);
    assert.equal(sphereVertices(0.5,4,3).length, 432);
    assert.equal(terrainMeshVertices(2,100).length, 144);
    assert.equal(terrainHeight(0,0), -10);
    const scene = new SceneManager();
    const first = scene._makeModel({x:1,y:2,z:3}, {x:0,y:0,z:0,w:1}, [0.5,1,2]);
    close(Array.from(first).slice(12,15), [1,2,3]);
    scene._writeInstance(0, first, 'box', [1,0,0,1]);
    scene._writeInstance(0, mat4.create(), 'box', [1,0,0,1]);
    close(scene.instanceData.slice(16,32), first);
});

test('required assets upload arbitrary meshes/static slots and destroy every mesh buffer', t => {
    assert.throws(() => new WebGPURenderer({}, 32, 24), /requires scene assets/);
    const { renderer, writes } = fixture();
    const oldBuffer = globalThis.GPUBuffer, oldTexture = globalThis.GPUTexture;
    t.after(() => { globalThis.GPUBuffer = oldBuffer; globalThis.GPUTexture = oldTexture; });
    globalThis.GPUBuffer = class {};
    globalThis.GPUTexture = class {};
    globalThis.GPUBufferUsage = { VERTEX: 1, COPY_DST: 2 };
    const data = new Float32Array(FLOATS_PER_INSTANCE).fill(0.25);
    renderer.assets.staticInstances = [{ index: 19, data }];
    renderer._createVertexBuffers();
    assert.deepEqual(Object.keys(renderer.meshes), ['custom']);
    assert.equal(renderer.meshes.custom.vertexCount, 3);
    assert.equal(renderer.quadVertCount, 6);
    const upload = writes.find(write => write.buffer === renderer.instanceBuf);
    assert.equal(upload.offset, 19 * FLOATS_PER_INSTANCE * Float32Array.BYTES_PER_ELEMENT);
    close(new Float32Array(upload.bytes.buffer), data);
    const buffer = renderer.meshes.custom.buffer;
    renderer.starStates = [];
    renderer.stars = renderer.starDraw = renderer.ctx = renderer.device = null;
    renderer.destroy();
    assert.equal(buffer.destroyed, true);
});

test('pipelines compile supplied scene shaders and renderer-owned display shader', () => {
    const { renderer } = fixture();
    const modules = [];
    renderer.assets.shaders = { skyWGSL: 'custom sky', sceneWGSL: 'custom scene', shadowWGSL: 'custom shadow' };
    renderer.device.createShaderModule = ({ code }) => { modules.push(code); return code; };
    renderer.device.createRenderPipeline = descriptor => descriptor;
    renderer._createPipelines();
    assert.deepEqual(modules, ['custom sky', 'custom scene', 'custom shadow', displayWGSL]);
});

test('arbitrary scene batches preserve offsets and order in shadow and both eyes', () => {
    const { renderer, passes } = fixture();
    renderer.meshes.second = { buffer: 'secondVB', vertexCount: 12 };
    const opts = frameOptions();
    const batches = [...opts.batches, { mesh: 'second', firstInstance: 23, instanceCount: 4 }];
    renderer.frame({ ...opts, batches, stereo: { mode: 2, viewProjL: opts.viewProj,
        viewProjR: opts.viewProj, prevViewProjL: opts.viewProj, prevViewProjR: opts.viewProj } });
    const draws = [[3, 2, 0, 7], [12, 4, 0, 23]];
    assert.deepEqual(passes[0].draws, draws);
    for (const pass of passes.slice(1, 3)) {
        assert.deepEqual(pass.draws, [[6], ...draws]);
        assert.deepEqual(pass.vertices.slice(1), ['customVB', 'secondVB']);
    }
    const empty = fixture();
    empty.renderer.frame({ ...opts, batches: [] });
    assert.deepEqual(empty.passes[0].draws, []);
    assert.deepEqual(empty.passes[1].draws, [[6]]);
});

test('sandbox assets retain original geometry, terrain sentinel and reserved row', () => {
    const assets = createSandboxAssets();
    assert.deepEqual(assets.meshes.box, boxVertices());
    assert.deepEqual(assets.meshes.domino, beveledBoxVertices(0.75, 1.5, 0.18, 0.04, 2));
    assert.deepEqual(assets.meshes.sphere, sphereVertices());
    assert.deepEqual(assets.meshes.terrain, terrainMeshVertices(600, 4500));
    assert.deepEqual(assets.shaders, { sceneWGSL, skyWGSL, shadowWGSL });
    assert.equal(assets.staticInstances.length, 1);
    const { index, data } = assets.staticInstances[0];
    assert.equal(index, MAX_INSTANCES - 1);
    assert.equal(index, TERRAIN_INSTANCE_IDX);
    assert.equal(data.length, FLOATS_PER_INSTANCE);
    close(data.slice(0, 16), mat4.create());
    close(data.slice(16, 32), mat4.create());
    close(data.slice(32), [0.12, 0.48, 0.08, 1]);
});

test('sandbox owns ordered topology and cannot overwrite its reserved terrain row', () => {
    const manager = new SceneManager();
    const body = { pos: {x: 0, y: 0, z: 0}, rot: {x: 0, y: 0, z: 0, w: 1} };
    const half = { x: 0.75, y: 1.5, z: 0.18 };
    const scene = { floor: { ...body, half }, dominoes: [body, body], dominoHalf: half,
        mazeWalls: [{ ...body, half: [1, 2, 3] }], spheres: [body], sphereRadius: 0.5 };
    const result = manager.buildInstances(scene);
    assert.deepEqual(result, { numBoxInstances: 4, numSphereInstances: 1, numDominoInstances: 2,
        batches: [
            { mesh: 'box', firstInstance: 0, instanceCount: 1 },
            { mesh: 'domino', firstInstance: 1, instanceCount: 2 },
            { mesh: 'box', firstInstance: 3, instanceCount: 1 },
            { mesh: 'sphere', firstInstance: 4, instanceCount: 1 },
            { mesh: 'terrain', firstInstance: TERRAIN_INSTANCE_IDX, instanceCount: 1 },
        ] });
    assert.equal(manager.getActiveData().length, 5 * FLOATS_PER_INSTANCE);
    assert.deepEqual(manager.buildInstances({ ...scene, dominoes: [], mazeWalls: [], spheres: [] }).batches,
        [{ mesh: 'box', firstInstance: 0, instanceCount: 1 },
            { mesh: 'terrain', firstInstance: TERRAIN_INSTANCE_IDX, instanceCount: 1 }]);
    assert.throws(() => manager._writeInstance(TERRAIN_INSTANCE_IDX, mat4.create(), 'overflow', [1,1,1,1]), /reserved terrain slot/);
});

test('StarDraw uses candidate-indexed masks and never mutates coordinate state', () => {
    const { device, writes, passes } = recordingDevice();
    const draw = Object.assign(Object.create(StarDraw.prototype), { device, W: 1024, H: 1024, uniforms: ['L','R'], views: ['left','right'], pipeline: { getBindGroupLayout: () => ({}) }, starAAEnabled: true, starSizeMaxPx: 8 });
    const state = Object.freeze({ mergedPosL: 'posL', mergedPosR: 'posR', mergedMetaL: 'metaL', mergedMetaR: 'metaR', mask: 'mask' });
    draw.encode(device.createCommandEncoder(), state, 100, true);
    assert.deepEqual(passes.map(pass => pass.draws), [[[1200]], [[1200]]]);
    assert.deepEqual(writes.map(write => new Uint32Array(write.bytes.buffer)[11]), [1,2]);
    assert.match(displayWGSL, /srgbEncode\(rgb.r\)/);
});
