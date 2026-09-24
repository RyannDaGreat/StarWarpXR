/**
 * WebGPU scene/shadow/SBS display adapter. Star coordinates and rasterization are separate.
 */

import { quadVertices } from './geometry.js';
import { displayWGSL } from './shaders.js';
import { MAX_INSTANCES, FLOATS_PER_INSTANCE } from './layout.js';

import { mat4, vec3 } from 'gl-matrix';
import { StarWarp, createStarState, destroyStarState } from '../stars/star-warp.js';
import { StarDraw } from './star-draw.js';
export const STARS_MODE = 6;
/**
 * Pure function. Directional shadow transform, world XYZ → WebGPU NDC XYZ (depth 0..1).
 * Uses a right-handed view so back-face culling retains the sun-facing surfaces.
 * @param {number[]} sunDir - Unit direction toward the sun, e.g. [0,1,0,0].
 * @param {number} halfExtent - Half-width/height in world units.
 * @param {number} depthRange - Near-to-far world distance, centered at the origin.
 * @returns {Float32Array} Column-major [16] matrix.
 * @example buildLightSpaceMatrix([0,1,0,0], 200, 300)[15] // 1
 */
export function buildLightSpaceMatrix(sunDir, halfExtent, depthRange) {
    const zenithThreshold = 0.99;
    const up = Math.abs(sunDir[1]) > zenithThreshold ? [0,0,1] : [0,1,0];
    const eye = vec3.scale([], sunDir, depthRange / 2);
    const view = mat4.lookAt(mat4.create(), eye, [0,0,0], up);
    const projection = mat4.orthoZO(mat4.create(), -halfExtent, halfExtent,
        -halfExtent, halfExtent, 0, depthRange);
    return mat4.multiply(mat4.create(), projection, view);
}

/**
 * Command. Encode ordered mesh batches into a scene or shadow render pass.
 * @param {GPURenderPassEncoder} pass - Pass with pipeline and bind group already set.
 * @param {object} meshes - Mesh keys mapped to {buffer, vertexCount} GPU resources.
 * @param {object[]} batches - Ordered {mesh, firstInstance, instanceCount} draws.
 * @example encodeBatches(pass, meshes, [{mesh:'cube', firstInstance:7, instanceCount:2}]) // undefined; draws two cubes
 */
export function encodeBatches(pass, meshes, batches) {
    for (const { mesh, firstInstance, instanceCount } of batches) {
        const { buffer, vertexCount } = meshes[mesh];
        pass.setVertexBuffer(0, buffer);
        pass.draw(vertexCount, instanceCount, 0, firstInstance);
    }
}
export class WebGPURenderer {
    /**
     * Command. Store canvas/resolution and explicitly supplied scene assets.
     * @param {HTMLCanvasElement} canvas - WebGPU output canvas.
     * @param {number} W - Per-eye width in pixels.
     * @param {number} H - Per-eye height in pixels.
     * @param {object} assets - Required {meshes, staticInstances, shaders}: meshes map keys to
     *   flat [N,6] Float32Arrays (XYZ position/normal); staticInstances contain {index,data}
     *   with flat [36] rows (model[16], prevModel[16], RGBA[4]); shaders contain
     *   sceneWGSL, skyWGSL, shadowWGSL matching this renderer's pipeline/binding ABI.
     * @example new WebGPURenderer(canvas, 1024, 1024, assets) // uninitialized renderer
     */
    constructor(canvas, W, H, assets) {
        if (!assets) throw new TypeError('WebGPURenderer requires scene assets');
        this.assets = assets;
        this.canvas = canvas;
        this.W = W;
        this.H = H;
        this.frameCount = 0;

        this.shadowsEnabled = true;
        this.shadowResolution = 4096;
        this.pointLightsEnabled = true;
        this.numStars = 10000;
        this.daySpeedMultiplier = 0.2;  // 5x slower than original (1500-sec cycle)
    }

    /**
     * Command. Allocate device/resources and compile retained scene pipelines.
     * @returns {Promise<void>} Resolves after resource setup; GPU errors are reported and fail frame().
     * @example await renderer.init() // undefined; ready for frame()
     */
    async init() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance', xrCompatible: typeof XRGPUBinding === 'function' });
        if (!adapter) throw new Error('WebGPU: no adapter found');

        this.device = await adapter.requestDevice({
            
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            },
        });
        this.device.lost.then(info => {
            if (info.reason !== 'destroyed') {
                this.gpuError = new Error('WebGPU device lost: ' + info.message);
                console.error(this.gpuError);
            }
        });

        this.device.addEventListener('uncapturederror', event => {
            this.gpuError = event.error;
            console.error('WebGPU error:', event.error);
        });

        this.ctx = this.canvas.getContext('webgpu');
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

        this._createTextures();
        this._createBuffers();
        this._createPipelines();
        this._createBindGroups();
        this._createVertexBuffers();
        this.stars = new StarWarp(this.device, this.W, this.H);
        this.resetHistory();
        this.starDraw = new StarDraw(this.device, this.W, this.H);
        this._createDisplayBindGroup();
    }

    /**
     * Command. Replace/reseed both coordinate states after a mode/session/teleport change.
     * Preserves renderer star count and all StarDraw settings/resources. Call after init;
     * the app must also reset previous camera and instance transforms to current.
     * @param {number} seed - Explicit initial seed, shared by the two ping-pong states.
     * @example renderer.resetHistory(777) // undefined; fresh coordinate histories
     */
    resetHistory(seed = 777) {
        if (this.gpuError) throw this.gpuError;
        if (!this.device) throw new Error('resetHistory requires an initialized device');
        for (const state of this.starStates || []) destroyStarState(state);
        this.starStates = [createStarState(this.device, this.W, this.H, seed),
            createStarState(this.device, this.W, this.H, seed)];
        this.frameCount = 0;
    }

    /**
     * Command. Rebuild resolution-dependent resources and reseed star histories in place.
     * Retains the device/context (including native XR bindings), scene pipelines and buffers;
     * recreates the fixed-resolution shadow map too. The app resets camera/instance history.
     * @param {number} W - Integer per-eye width >= 2; SBS width must fit the device.
     * @param {number} H - Integer per-eye height >= 2; textures/compute must fit the device.
     * @example renderer.resize(1536, 1536) // undefined; same device, fresh 1536² eye targets
     */
    resize(W, H) {
        if (this.gpuError) throw this.gpuError;
        if (!this.device) throw new Error('resize requires an initialized device');
        if (!Number.isInteger(W) || !Number.isInteger(H) || W < 2 || H < 2) {
            throw new RangeError('resize requires integer W,H >= 2');
        }
        const { maxTextureDimension2D, maxStorageBufferBindingSize, maxBufferSize,
            maxComputeWorkgroupsPerDimension } = this.device.limits;
        const scratchBytes = W * H * Uint32Array.BYTES_PER_ELEMENT;
        const pixelsPerWorkgroup = 256; // StarWarp pixel dispatch workgroup size.
        if (2 * W > maxTextureDimension2D || H > maxTextureDimension2D) {
            throw new RangeError('resize exceeds maxTextureDimension2D (including SBS width)');
        }
        if (scratchBytes > maxStorageBufferBindingSize || scratchBytes > maxBufferSize) {
            throw new RangeError('resize exceeds maxStorageBufferBindingSize or maxBufferSize');
        }
        if (Math.ceil(W * H / pixelsPerWorkgroup) > maxComputeWorkgroupsPerDimension) {
            throw new RangeError('resize exceeds maxComputeWorkgroupsPerDimension');
        }
        if (W === this.W && H === this.H) return;

        const { starAAEnabled, starColorQEnabled, starSizeQEnabled,
            starSizeMaxPx, cullOrphansEnabled } = this.starDraw;
        this.stars.destroy();
        this.starDraw.destroy();
        for (const value of Object.values(this)) {
            if (value instanceof GPUTexture) value.destroy();
        }
        this.W = W; this.H = H;
        this.canvas.width = this._stereoActive ? W * 2 : W;
        this.canvas.height = H;
        this._createTextures();
        this._createBindGroups();
        this.stars = new StarWarp(this.device, W, H);
        this.resetHistory();
        this.starDraw = new StarDraw(this.device, W, H);
        Object.assign(this.starDraw, { starAAEnabled, starColorQEnabled, starSizeQEnabled,
            starSizeMaxPx, cullOrphansEnabled });
        this._createDisplayBindGroup();
    }

    /** Command. Release owned GPU resources, including the privately owned device. */
    destroy() {
        this.stars?.destroy(); this.starDraw?.destroy();
        for (const state of this.starStates || []) destroyStarState(state);
        for (const { buffer } of Object.values(this.meshes || {})) buffer.destroy();
        for (const value of Object.values(this)) {
            if (value instanceof GPUBuffer || value instanceof GPUTexture) value.destroy();
        }
        this.ctx?.unconfigure();
        this.device?.destroy();
    }
    /** Command. Allocate owned scene, motion, cross-eye and shadow textures. */
    _createTextures() {
        const { device, W, H } = this;
        this.colorTex = device.createTexture({
            size: [W, H], format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.colorTexR = device.createTexture({ size: [W,H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        this.colorTexRView = this.colorTexR.createView();
        this.motionTex = device.createTexture({
            size: [W, H], format: 'rgba32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        // Stereo: right-eye temporal motion + per-eye cross-eye flow targets.
        this.motionTexR = device.createTexture({
            size: [W, H], format: 'rgba32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.crossTexL = device.createTexture({   // written by eye L: flow L->R
            size: [W, H], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.crossTexR = device.createTexture({   // written by eye R: flow R->L
            size: [W, H], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.depthTex = device.createTexture({
            size: [W, H], format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        // Shadow map: configurable resolution depth texture, sampled in scene shader for PCF.
        this.shadowTex = device.createTexture({
            size: [this.shadowResolution, this.shadowResolution], format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.colorTexView  = this.colorTex.createView();
        this.motionTexView = this.motionTex.createView();
        this.depthTexView  = this.depthTex.createView();
        this.shadowTexView = this.shadowTex.createView({ aspect: 'depth-only' });
        this.motionTexRView = this.motionTexR.createView();
        this.crossTexLView = this.crossTexL.createView();
        this.crossTexRView = this.crossTexR.createView();
    }

    /** Command. Allocate camera, light, instance and display buffers. */
    _createBuffers() {
        const {device} = this;
        const floatBytes = Float32Array.BYTES_PER_ELEMENT;
        // Camera uniform: viewProj (64) + prevViewProj (64) + sunDir (16) + lightSpaceMatrix (64)
        //                 + eyePos (16) + eyeDir (16) + otherViewProj (64) = 304 bytes
        this.cameraUniformBuf = device.createBuffer({
            size: 304, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.cameraUniformBufR = device.createBuffer({    // right stereo eye
            size: 304, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Shadow uniform: lightSpaceMatrix (64 bytes) for the shadow depth pass
        this.shadowUniformBuf = device.createBuffer({
            size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sky uniform: invViewProj (64) + sunDir (16) + time vec4 (16) = 96 bytes
        this.skyUniformBuf = device.createBuffer({
            size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Point light buffer: count(u32) + 3 pad(u32) + 32 × PointLight(2 × vec4f = 32 bytes)
        // Total: 16 + 32 × 32 = 1040 bytes
        this.lightUniformBuf = device.createBuffer({
            size: 1040, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Instance storage buffer: MAX_INSTANCES × 144 bytes
        this.instanceBuf = device.createBuffer({
            size: MAX_INSTANCES * FLOATS_PER_INSTANCE * floatBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.displayUniformBuf = device.createBuffer({
            size: 4 * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.skyUniformBufR = device.createBuffer({size:96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    }
    /** Command. Compile retained upstream scene/shadow/sky and distilled display pipelines. */
    _createPipelines() {
        const { device } = this;
        const { skyWGSL, sceneWGSL, shadowWGSL } = this.assets.shaders;
        const mod = (code) => device.createShaderModule({ code });
        const skyModule       = mod(skyWGSL);
        const sceneModule     = mod(sceneWGSL);
        const shadowModule    = mod(shadowWGSL);
        const displayModule   = mod(displayWGSL);

        // Shadow pipeline: depth-only, no fragment shader, same vertex layout as scene.
        this.shadowPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shadowModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 6 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x3' },
                    ],
                }],
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
        });

        // Scene pipeline: instanced, 6 floats per vertex (position + normal)
        this.scenePipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: sceneModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 6 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x3' },
                    ],
                }],
            },
            fragment: {
                module: sceneModule, entryPoint: 'fs',
                targets: [
                    { format: 'rgba8unorm' },
                    { format: 'rgba32float' },
                    { format: 'rgba16float' },   // cross-eye flow (zero in mono)
                ],
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
        });

        // Sky pipeline: fullscreen quad, writes to same MRT as scene, depth = 1.0
        this.skyPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: skyModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 4 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: skyModule, entryPoint: 'fs',
                targets: [
                    { format: 'rgba8unorm' },
                    { format: 'rgba32float' },
                    { format: 'rgba16float' },   // cross-eye flow (sky: zero disparity)
                ],
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'always',  // always pass; depth already cleared to 1.0
            },
            primitive: { topology: 'triangle-list' },
        });

        // Distilled scene/star display pipeline.
        this.displayPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: displayModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 4 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: displayModule, entryPoint: 'fs',
                targets: [{ format: this.canvasFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });

    }
    /** Command. Bind scene resources for each eye. */
    _createBindGroups() {
        const { device } = this;
        const buf = (b) => ({ buffer: b });

        // Sky bind group: sky uniforms
        this.skyBindGroup = device.createBindGroup({
            layout: this.skyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.skyUniformBuf) },
            ],
        });

        // Shadow comparison sampler — used by scene shader for PCF
        this.shadowSampler = device.createSampler({
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Shadow pass bind group: light-space uniform + instance storage
        this.shadowPassBindGroup = device.createBindGroup({
            layout: this.shadowPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.shadowUniformBuf) },
                { binding: 1, resource: buf(this.instanceBuf) },
            ],
        });

        // Scene bind group: camera uniform + instance storage + shadow map + shadow sampler + lights
        const sceneEntries = (camBuf) => [
            { binding: 0, resource: buf(camBuf) },
            { binding: 1, resource: buf(this.instanceBuf) },
            { binding: 2, resource: this.shadowTexView },
            { binding: 3, resource: this.shadowSampler },
            { binding: 4, resource: buf(this.lightUniformBuf) },
        ];
        this.sceneBindGroup = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: sceneEntries(this.cameraUniformBuf),
        });
        this.sceneBindGroupR = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: sceneEntries(this.cameraUniformBufR),
        });

        this.skyBindGroupR = device.createBindGroup({layout:this.skyPipeline.getBindGroupLayout(0), entries:[{binding:0,resource:buf(this.skyUniformBufR)}]});
    }
    /** Command. Upload supplied XYZ/normal meshes, static instance rows and fullscreen quad. */
    _createVertexBuffers() {
        const { device } = this;
        const uploadVB = (data) => {
            const vb = device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(vb, 0, data);
            return vb;
        };

        this.meshes = Object.fromEntries(Object.entries(this.assets.meshes).map(([key, data]) =>
            [key, { buffer: uploadVB(data), vertexCount: data.length / 6 }]));
        const quadData = quadVertices();
        this.quadVB = uploadVB(quadData);
        this.quadVertCount = quadData.length / 4;
        for (const { index, data } of this.assets.staticInstances) {
            const byteOffset = index * FLOATS_PER_INSTANCE * Float32Array.BYTES_PER_ELEMENT;
            device.queue.writeBuffer(this.instanceBuf, byteOffset, data);
        }
    }

    /**
     * Command. Render one complete frame; writes uniforms, GPU targets and coordinate state.
     *
     * @param {object} opts
     * @param {Float32Array} opts.viewProj - current GL viewProj matrix (depth -1..1)
     * @param {Float32Array} opts.prevViewProj - previous frame's GL viewProj
     * @param {Float32Array} opts.invViewProj - inverse of current viewProj (for sky ray reconstruction)
     * @param {Float32Array} opts.instanceData - Flat [N,36]: model[16], prevModel[16], RGBA[4].
     * @param {object[]} opts.batches - Ordered {mesh, firstInstance, instanceCount} draws for both passes.
     * @param {number} opts.displayMode - 1 scene or 6 stars
     * @param {object} opts.stereo - mode=2 for L|R SBS; viewProjL/R, prevViewProjL/R [16]
     *   use GL clip depth, with optional invViewProjL/R and eyePosL/R, eyeDirL/R.
     * @param {number} opts.frameSeed - incrementing seed
     * @param {number} opts.elapsedSecs - seconds since page load, drives day/night cycle
     * @param {number[]} opts.eyePos - [x, y, z] camera world position (for flashlight)
     * @param {number[]} opts.eyeDir - [x, y, z] camera forward unit vector (for flashlight)
     * @example renderer.frame(frameOptions) // undefined; submits a mono or L|R frame
     */
    frame({ viewProj, prevViewProj, invViewProj, instanceData, batches, displayMode, frameSeed, elapsedSecs = 0, eyePos = [0,0,0], eyeDir = [0,0,-1], lights = [], stereo = null }) {
        if (this.gpuError) throw this.gpuError;
        const { device, W, H } = this;


        const starsMode = displayMode === STARS_MODE;


        // Day/night cycle: base period 300 seconds, scaled by daySpeedMultiplier.
        // daySpeedMultiplier=0 freezes time; 1.0 = original 5-min cycle; 3.0 = 100s cycle.
        const DAY_CYCLE_SECS = 300;
        const effectiveSecs = elapsedSecs * this.daySpeedMultiplier;
        const angle = (2 * Math.PI * effectiveSecs) / DAY_CYCLE_SECS + Math.PI / 2;
        // sunDir: x=cos(angle) sweeps east→west, y=sin(angle) rises/sets, z=slight tilt north
        const rawX = Math.cos(angle);
        const rawY = Math.sin(angle);
        const rawZ = 0.3;
        const sunDir = [...vec3.normalize([], [rawX, rawY, rawZ]), 0.0];

        // Orthographic light-space matrix for directional shadow map.
        // Covers a 400×400 world-unit area centred at origin, depth range 0..300.
        const lightSpaceMatrix = buildLightSpaceMatrix(sunDir, 200, 300);

        // Both display modes use the actual per-eye GL view-projection matrices.
        if (displayMode !== 1 && displayMode !== STARS_MODE) throw new Error('Only display modes 1 (scene) and 6 (stars) are supported');
        if (stereo && stereo.mode !== 0 && stereo.mode !== 2) throw new Error('Only mono or parallel SBS stereo.mode=2 is supported');
        const stereoActive = stereo?.mode === 2;
        const canvasWidth = stereoActive ? W * 2 : W;
        if (this.canvas.width !== canvasWidth) this.canvas.width = canvasWidth;
        if (this.canvas.height !== H) this.canvas.height = H;
        this._stereoActive = stereoActive;

        // Upload camera uniforms: viewProj (64) + prevViewProj (64) + sunDir (16) + lightSpaceMatrix (64)
        //                         + eyePos (16) + eyeDir (16) + otherViewProj (64) = 304 bytes (76 floats)
        const buildCamData = (vp, prevVp, otherVp, eyePos, eyeDir) => {
            const camData = new Float32Array(76);
            camData.set(vp, 0);
            camData.set(prevVp, 16);
            camData.set(sunDir, 32);
            camData.set(lightSpaceMatrix, 36);
            camData.set([eyePos[0], eyePos[1], eyePos[2], 0.0], 52);
            camData.set([eyeDir[0], eyeDir[1], eyeDir[2], 0.0], 56);
            camData.set(otherVp, 60);
            return camData;
        };
        if (stereoActive) {
            device.queue.writeBuffer(this.cameraUniformBuf, 0,
                buildCamData(stereo.viewProjL, stereo.prevViewProjL, stereo.viewProjR, stereo.eyePosL ?? eyePos, stereo.eyeDirL ?? eyeDir));
            device.queue.writeBuffer(this.cameraUniformBufR, 0,
                buildCamData(stereo.viewProjR, stereo.prevViewProjR, stereo.viewProjL, stereo.eyePosR ?? eyePos, stereo.eyeDirR ?? eyeDir));
        } else {
            device.queue.writeBuffer(this.cameraUniformBuf, 0,
                buildCamData(viewProj, prevViewProj, viewProj, eyePos, eyeDir));
        }

        // Shadow uniform: lightSpaceMatrix only (used in depth-only shadow pass)
        device.queue.writeBuffer(this.shadowUniformBuf, 0, lightSpaceMatrix);

        // Upload sky uniforms: invViewProj (64) + sunDir (16) + time vec4 (16)
        const skyData = new Float32Array(24);
        const inverseL = stereoActive ? (stereo.invViewProjL ?? mat4.invert(mat4.create(), stereo.viewProjL)) : invViewProj;
        if (!inverseL) throw new Error('Singular left view-projection matrix');
        skyData.set(inverseL, 0);
        skyData.set(sunDir, 16);
        skyData[20] = elapsedSecs;  // time.x = elapsed seconds for cloud animation
        // skyData[21..23] = padding zeros
        device.queue.writeBuffer(this.skyUniformBuf, 0, skyData);
        if (stereoActive) {
            const inverseR = stereo.invViewProjR ?? mat4.invert(mat4.create(), stereo.viewProjR);
            if (!inverseR) throw new Error('Singular right view-projection matrix');
            skyData.set(inverseR, 0);
            device.queue.writeBuffer(this.skyUniformBufR, 0, skyData);
        }

        // Upload point lights: count(u32) + 3 pad(u32) + 32 × (posAndRadius vec4f + color vec4f)
        const lightData = new Float32Array(4 + 32 * 8);  // 260 floats = 1040 bytes
        const lightU32 = new Uint32Array(lightData.buffer);
        const numLights = this.pointLightsEnabled ? Math.min(lights.length, 32) : 0;
        lightU32[0] = numLights;
        for (let i = 0; i < numLights; i++) {
            const base = 4 + i * 8;
            lightData[base]     = lights[i].pos[0];
            lightData[base + 1] = lights[i].pos[1];
            lightData[base + 2] = lights[i].pos[2];
            lightData[base + 3] = lights[i].radius;
            lightData[base + 4] = lights[i].color[0];
            lightData[base + 5] = lights[i].color[1];
            lightData[base + 6] = lights[i].color[2];
            lightData[base + 7] = lights[i].intensity;
        }
        device.queue.writeBuffer(this.lightUniformBuf, 0, lightData);

        // Upload instance data
        if (instanceData.length > 0) {
            device.queue.writeBuffer(this.instanceBuf, 0, instanceData);
        }

        device.queue.writeBuffer(this.displayUniformBuf, 0, new Uint32Array([displayMode,W,H,stereoActive ? 2 : 0]));

        const encoder = device.createCommandEncoder();

        // --- Shadow pass: render all geometry into the configured depth map ---
        // Only run when shadows are enabled AND sun is above horizon (night has no shadow).
        if (this.shadowsEnabled && sunDir[1] > 0.0) {
            const shadowPass = encoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.shadowTexView,
                    depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
                },
            });
            shadowPass.setPipeline(this.shadowPipeline);
            shadowPass.setBindGroup(0, this.shadowPassBindGroup);

            encodeBatches(shadowPass, this.meshes, batches);

            shadowPass.end();
        } else {
            // Shadows disabled or sun below horizon: clear shadow map to 1.0 so PCF always passes.
            const clearPass = encoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.shadowTexView,
                    depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
                },
            });
            clearPass.end();
        }

        // --- Scene render (sky + instanced MRT), once per eye in stereo ---
        const encodeScene = (camBindGroup, motionView, crossView, colorView, skyGroup) => {
            const scenePass = encoder.beginRenderPass({
                colorAttachments: [
                    { view: colorView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
                    { view: motionView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
                    { view: crossView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
                ],
                depthStencilAttachment: {
                    view: this.depthTexView,
                    depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
                },
            });

            // Sky background: fullscreen quad at far plane
            scenePass.setPipeline(this.skyPipeline);
            scenePass.setBindGroup(0, skyGroup);
            scenePass.setVertexBuffer(0, this.quadVB);
            scenePass.draw(this.quadVertCount);

            // Scene geometry on top (depth < 1.0 wins)
            scenePass.setPipeline(this.scenePipeline);
            scenePass.setBindGroup(0, camBindGroup);

            encodeBatches(scenePass, this.meshes, batches);

            scenePass.end();
        };
        // In stereo the "main" pass IS the left eye (its motion drives the L stream).
        encodeScene(this.sceneBindGroup, this.motionTexView, this.crossTexLView, this.colorTexView, this.skyBindGroup);
        if (stereoActive) {
            encodeScene(this.sceneBindGroupR, this.motionTexRView, this.crossTexRView, this.colorTexRView, this.skyBindGroupR);
        }

        if (starsMode) {
            const [previous, output] = this.starStates;
            this.stars.encode(encoder, previous, output, {
                temporalL:this.motionTexView, temporalR:this.motionTexRView,
                crossL:this.crossTexLView, crossR:this.crossTexRView,
            }, frameSeed, stereoActive, { numStars: this.numStars });
            this.starDraw.encode(encoder, output, this.numStars, stereoActive);
            this.starStates.reverse();
        }
        // --- Display ---
        const canvasView = this.ctx.getCurrentTexture().createView();
        const dispPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: canvasView,
                loadOp: 'clear', storeOp: 'store',
                clearValue: [0.1, 0.1, 0.1, 1],
            }],
        });
        dispPass.setPipeline(this.displayPipeline);
        dispPass.setBindGroup(0, this.displayBindGroup);
        dispPass.setVertexBuffer(0, this.quadVB);
        dispPass.draw(this.quadVertCount);
        dispPass.end();

        device.queue.submit([encoder.finish()]);
        this.frameCount++;
    }
    /** Command. Bind independent scene/star textures for each eye. */
    _createDisplayBindGroup() {
        this.displayBindGroup = this.device.createBindGroup({
            layout:this.displayPipeline.getBindGroupLayout(0), entries:[
                {binding:0,resource:{buffer:this.displayUniformBuf}},
                {binding:1,resource:this.colorTexView}, {binding:2,resource:this.colorTexRView},
                {binding:3,resource:this.starDraw.views[0]}, {binding:4,resource:this.starDraw.views[1]},
            ],
        });
    }
}
