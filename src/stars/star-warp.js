import { starSplatWGSL, starScanRowsWGSL, starScanCdfWGSL, starUpdateWGSL, mergeSelectWGSL } from './shaders.js';

export const MAX_STARS = 1 << 20;
const POSITION_BYTES = MAX_STARS * 2 * 4;

/**
 * Command. Allocate and seed explicit state; no clock or global RNG is read.
 * Positions: [MAX_STARS, 2] float32 (x,y) pixels; metadata: [MAX_STARS, 2] (q:f32,id:u32).
 * @param {GPUDevice} device - Owner of buffers.
 * @param {number} W - Width in pixels, at least two.
 * @param {number} H - Height in pixels, at least two.
 * @param {number} seed - Explicit initial random seed.
 * @returns {object} Two streams, merged coordinate buffers, mask and ID counters.
 * @example createStarState(device, 1024, 1024, 777) // GPU coordinate state
 */
export function createStarState(device, W, H, seed = 777) {
    const state = {};
    for (const name of ['posL','metaL','posR','metaR','mergedPosL','mergedMetaL','mergedPosR','mergedMetaR','mask','counters']) {
        const size = name === 'mask' ? (2 * MAX_STARS + 4) * 4 : name === 'counters' ? 8 : name.startsWith('merged') ? POSITION_BYTES * 2 : POSITION_BYTES;
        state[name] = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    }
    // writeBuffer snapshots its input, so both eyes can reuse these CPU arrays.
    const pos = new Float32Array(MAX_STARS * 2);
    const meta = new ArrayBuffer(POSITION_BYTES);
    const q = new Float32Array(meta), ids = new Uint32Array(meta);
    for (const [eye, offset] of [['L',0],['R',1]]) {
        let rngState = (seed + offset) | 0;
        // Upstream mulberry32 initialization, retained verbatim numerically.
        /**
         * Command. Advance the local mulberry32 seed.
         * @returns {number} Uniform sample in [0,1).
         * @example random() // first sample for seed 777: approximately 0.776783
         */
        const random = () => {
            rngState = (rngState + 0x6D2B79F5) | 0;
            let t = Math.imul(rngState ^ (rngState >>> 15), rngState | 1);
            t = (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        for (let i = 0; i < MAX_STARS; i++) {
            pos[2*i] = random() * W; pos[2*i+1] = random() * H;
            q[2*i] = random(); ids[2*i+1] = offset * MAX_STARS + i;
        }
        device.queue.writeBuffer(state['pos'+eye],0,pos);
        device.queue.writeBuffer(state['meta'+eye],0,meta);
    }
    device.queue.writeBuffer(state.counters,0,new Uint32Array([2*MAX_STARS,0]));
    return state;
}

/**
 * Command. Release an explicitly owned state; does not destroy its device.
 * @param {object} state - Result of createStarState.
 * @example destroyStarState(state) // returns undefined
 */
export function destroyStarState(state) {
    for (const buffer of Object.values(state)) buffer.destroy();
}

/** GPU command adapter for the upstream coordinate algorithm, independent of drawing. */
export class StarWarp {
    /**
     * Command. Allocate scratch resources and compile unchanged upstream compute kernels.
     * @param {GPUDevice} device - Device with at least nine storage bindings per stage.
     * @param {number} W - Motion width (pixels).
     * @param {number} H - Motion height (pixels).
     * @example new StarWarp(device, 1024, 1024) // reusable compute adapter
     */
    constructor(device, W, H) {
        if (!Number.isInteger(W) || !Number.isInteger(H) || W < 2 || H < 2) throw new Error('StarWarp requires integer W,H >= 2');
        this.device=device; this.W=W; this.H=H;
        this.pipelines = [starSplatWGSL,starScanRowsWGSL,starScanCdfWGSL,starUpdateWGSL,mergeSelectWGSL].map(code => device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code}),entryPoint:'main'}}));
        this.scratch = [W*H*4,W*H*4,H*4].map(size => device.createBuffer({size,usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST}));
        this.uniforms = [0,1].map(() => device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}));
    }

    /**
     * Command. Encode next state from explicit previous state, fields and seed; previous is untouched.
     * Not pure: records GPU work, writes uniforms/scratch/output. Atomic density sums and ID
     * allocation retain upstream GPU scheduling nondeterminism (not bitwise reproducible).
     * Submit this encoder before calling encode again: scratch/uniforms are reusable, not concurrent.
     * @param {GPUCommandEncoder} encoder - Caller-owned command stream.
     * @param {object} previous - Input coordinate state.
     * @param {object} output - Distinct output state, same allocation layout.
     * @param {object} fields - temporalL/R and crossL/R texture views, [H,W,4], rg=(dx,-dy)
     *   normalized by W,H. CrossL means L→R, crossR means R→L. Channels ba are unused.
     * @param {number} seed - Explicit frame seed; no hidden time/frame counter.
     * @param {boolean} stereo - Advance and merge both streams.
     * @param {object} settings - Explicit {numStars} active coordinate count.
     * @returns {object} Output coordinate state (not textures).
     * @example warp.encode(encoder, previous, next, fields, 42, true, {numStars:10000}) // next
     */
    encode(encoder, previous, output, fields, seed, stereo, settings) {
        const {device,W,H} = this;
        const {numStars:n} = settings;
        if (!Number.isInteger(n) || n < 1 || n > MAX_STARS) throw new Error('numStars must be an integer in [1, MAX_STARS]');
        if (Object.values(previous).some(buffer => Object.values(output).includes(buffer))) throw new Error('Previous and output star states must not alias');
        for (const name of ['posL','metaL','posR','metaR','counters']) encoder.copyBufferToBuffer(previous[name],0,output[name],0,name==='counters'?8:POSITION_BYTES);
        const [density,prefix,cdf] = this.scratch;
        /**
         * Pure function. Wrap a GPU buffer binding.
         * @param {GPUBuffer} value - Buffer to bind.
         * @returns {GPUBufferBinding} Binding descriptor.
         * @example buffer(density) // {buffer:density}
         */
        const buffer = value => ({buffer:value});
        /**
         * Command. Record one compute pass on the caller's encoder.
         * @param {number} index - Pipeline index.
         * @param {GPUBindingResource[]} resources - Binding-order resources.
         * @param {number} groups - Dispatch width.
         * @returns {undefined}
         * @example run(2, [u, buffer(prefix), buffer(cdf)], 1) // undefined; records CDF scan
         */
        const run = (index, resources, groups) => {
            const pipeline=this.pipelines[index];
            const bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:resources.map((resource,binding)=>({binding,resource}))});
            const pass=encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0,bindGroup); pass.dispatchWorkgroups(groups); pass.end();
        };
        const pixelGroups=Math.ceil(W*H/256);
        for (let eye=0;eye<(stereo?2:1);eye++) {
            const suffix=eye?'R':'L', u=buffer(this.uniforms[eye]);
            const frameSeed=eye ? (Math.imul(seed,2654435761)^0x9e3779b9)>>>0 : seed;
            device.queue.writeBuffer(this.uniforms[eye],0,new Uint32Array([W,H,frameSeed,n,eye,0,0,0]));
            encoder.clearBuffer(density);
            run(0,[u,fields['temporal'+suffix],buffer(density)],pixelGroups);
            run(1,[u,buffer(density),buffer(prefix)],Math.ceil(H/64));
            run(2,[u,buffer(prefix),buffer(cdf)],1);
            run(3,[u,fields['temporal'+suffix],buffer(density),buffer(prefix),buffer(cdf),buffer(output['pos'+suffix]),buffer(output['meta'+suffix]),buffer(output.counters)],Math.ceil(n/256));
        }
        if (stereo) {
            // Only active candidate bits and the three count words are consumed.
            encoder.clearBuffer(output.mask, 0, (2*n + 3)*4);
            for (let eye=0;eye<2;eye++) {
                const own=eye?'R':'L', other=eye?'L':'R', u=buffer(this.uniforms[eye]);
                encoder.clearBuffer(density);
                run(0,[u,fields['cross'+other],buffer(density)],pixelGroups);
                run(4,[u,fields['cross'+other],buffer(density),buffer(output['pos'+own]),buffer(output['meta'+own]),buffer(output['pos'+other]),buffer(output['meta'+other]),buffer(output['mergedPos'+own]),buffer(output['mergedMeta'+own]),buffer(output.mask)],Math.ceil(2*n/256));
            }
        }
        return output;
    }

    /**
     * Command. Destroy scratch buffers; caller retains ownership of states and device.
     * @example warp.destroy() // undefined; state buffers remain alive
     */
    destroy() { for (const buffer of [...this.scratch,...this.uniforms]) buffer.destroy(); }
}
