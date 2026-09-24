import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { MAX_STARS, StarWarp, createStarState, destroyStarState } from '../src/stars/star-warp.js';

const POSITION_BYTES = MAX_STARS * 8;

/**
 * Command. Create a recording mock; no GPU allocation, shader execution or numerical simulation.
 * @returns {object} Device/encoder doubles and their shared event log.
 * @example const gpu = mockGpu(); gpu.device.createBuffer({size:8}) // tracked fake buffer
 */
function mockGpu() {
    const events = [], buffers = [];
    let pipelineIndex = 0;
    const device = {
        /** Command. Register a fake allocation. @param {object} descriptor - Buffer size/usage. @returns {object} Fake buffer. @example createBuffer({size:8}) // {size:8,destroyed:false,destroy:…} */
        createBuffer(descriptor) {
            const buffer = { ...descriptor, destroyed:false,
                /** Command. Mark this fake allocation destroyed. @example buffer.destroy() // undefined; buffer.destroyed becomes true */
                destroy() { this.destroyed = true; },
            };
            buffers.push(buffer);
            return buffer;
        },
        /** Pure function. Return the supplied shader descriptor without compilation. @param {object} descriptor - WGSL descriptor. @returns {object} Descriptor. @example createShaderModule({code:'WGSL'}) // {code:'WGSL'} */
        createShaderModule(descriptor) { return descriptor; },
        /** Command. Assign a pipeline index. @returns {object} Fake pipeline. @example createComputePipeline() // first pipeline: {index:0,getBindGroupLayout:…} */
        createComputePipeline() {
            return { index:pipelineIndex++,
                /** Query. Return this pipeline's fake layout. @returns {number} Pipeline index. @example pipeline.getBindGroupLayout() // 0 for the first pipeline */
                getBindGroupLayout() { return this.index; },
            };
        },
        /** Pure function. Return the supplied bindings without GPU validation. @param {object} descriptor - Bindings. @returns {object} Descriptor. @example createBindGroup({layout:0,entries:[]}) // {layout:0,entries:[]} */
        createBindGroup(descriptor) { return descriptor; },
        queue: {
            /** Command. Snapshot upload content as a digest plus small u32 prefix. @param {object} target - Buffer. @param {number} offset - Byte offset. @param {ArrayBuffer|ArrayBufferView} data - Upload. @example writeBuffer(buffer,0,new Uint32Array([42])) // undefined; logs words:[42] */
            writeBuffer(target, offset, data) {
                const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength);
                events.push({kind:'write', target, offset, size:bytes.byteLength,
                    digest:createHash('sha256').update(bytes).digest('hex'),
                    words:Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, Math.min(8, bytes.byteLength / 4))),
                });
            },
        },
    };
    const encoder = {
        /** Command. Record a copy without executing it. @param {...*} args - WebGPU copy arguments. @example copyBufferToBuffer(a,0,b,0,8) // undefined; logs an 8-byte copy */
        copyBufferToBuffer(...args) { events.push({kind:'copy', args}); },
        /** Command. Record a clear without executing it. @param {...*} args - WebGPU clear arguments. @example clearBuffer(mask,0,12) // undefined; logs a 12-byte clear */
        clearBuffer(...args) { events.push({kind:'clear', args}); },
        /** Command. Begin a fake pass whose end publishes the recorded dispatch. @returns {object} Pass recorder. @example beginComputePass() // recorder with setPipeline, setBindGroup, dispatchWorkgroups, end */
        beginComputePass() {
            const event = {kind:'pass'};
            return {
                /** Command. Record pipeline. @param {object} pipeline - Fake pipeline. @example setPipeline({index:3}) // undefined; selects the update kernel */
                setPipeline(pipeline) { event.pipeline = pipeline.index; },
                /** Command. Record bindings. @param {number} index - Group index. @param {object} group - Bindings. @example setBindGroup(0,{entries:[{resource:{buffer:positions}}]}) // undefined; logs positions binding */
                setBindGroup(index, group) { assert.equal(index, 0); event.resources = group.entries.map(entry => entry.resource); },
                /** Command. Record dispatch width. @param {number} groups - Workgroups. @example dispatchWorkgroups(2) // undefined; records two workgroups */
                dispatchWorkgroups(groups) { event.groups = groups; },
                /** Command. Publish this completed pass. @example pass.end() // undefined; appends one pass event */
                end() { events.push(event); },
            };
        },
    };
    return {device, encoder, events, buffers};
}

/** Command. Verify upstream CPU initialization, ownership and command wiring; never runs WGSL. @example verifyStars() // undefined if all assertions pass; throws otherwise */
function verifyStars() {
    // Node has no WebGPU constants; the adapter only needs these standard flag values.
    const originalUsage = globalThis.GPUBufferUsage;
    globalThis.GPUBufferUsage = {COPY_SRC:4, COPY_DST:8, UNIFORM:64, STORAGE:128};
    try {
        const gpu = mockGpu();
        const {device, encoder, events} = gpu;
        const previous = createStarState(device, 8, 6);
        // SHA-256 of complete uploads produced by retained renderer._initStars at W=8,H=6.
        const expected = {
            posL:'f06deeaf50ddd51bf48260cd0e75d01bbe9d7a11d6e1070bbf6ca4bb130ab1a0',
            metaL:'6d92e8a0bc5f28f1c7d9e30e357bba6978a1603c12ce16d970272a3f0e18894a',
            posR:'6506a474e24912927f28700165317d1f85e247a352e43dae7ef253d6afc86fac',
            metaR:'fdc69d139338bd912b349df53505f56604d8bbd382dbd22ef8098af8be9ed70f',
            counters:'291f6b4fa86b5fc9fd6e8a420eb811c2bf16f2db788344ba3eb0993378773e51',
        };
        for (const [name, digest] of Object.entries(expected)) {
            assert.equal(events.find(event => event.target === previous[name]).digest, digest, name);
        }
        assert.deepEqual(events.at(-1).words, [2 * MAX_STARS, 0]);
        const output = createStarState(device, 8, 6, 19);
        assert.notEqual(events.find(event => event.target === output.posL).digest, expected.posL);
        assert.equal(new Set([...Object.values(previous), ...Object.values(output)]).size, 20);
        for (const state of [previous, output]) {
            for (const [name, buffer] of Object.entries(state)) {
                const size = name === 'mask' ? (2 * MAX_STARS + 4) * 4 : name === 'counters' ? 8 : name.startsWith('merged') ? POSITION_BYTES * 2 : POSITION_BYTES;
                assert.equal(buffer.size, size, name);
                assert.equal(buffer.usage, 140);
            }
        }
        const warp = new StarWarp(device, 8, 6);
        const allocations = gpu.buffers.length;
        const fields = {temporalL:{name:'temporalL'}, temporalR:{name:'temporalR'}, crossL:{name:'crossL'}, crossR:{name:'crossR'}};
        const previousRefs = {...previous};
        for (const stereo of [false, true]) {
            for (const n of [1, 257, MAX_STARS]) {
                events.length = 0;
                assert.equal(warp.encode(encoder, previous, output, fields, 42, stereo, {numStars:n}), output);
                assert.equal('numStars' in warp, false, 'active count belongs to each explicit call, not the adapter');
                assert.equal(gpu.buffers.length, allocations, 'encode allocates no GPU buffers');
                assert.deepEqual(previous, previousRefs);
                const copies = events.filter(event => event.kind === 'copy');
                assert.equal(copies.length, 5);
                for (const [index, name] of ['posL','metaL','posR','metaR','counters'].entries()) {
                    assert.deepEqual(copies[index].args, [previous[name], 0, output[name], 0, name === 'counters' ? 8 : POSITION_BYTES]);
                }
                assert.ok(events.slice(0, 5).every(event => event.kind === 'copy'));
                const uploads = events.filter(event => event.kind === 'write');
                assert.deepEqual(uploads.map(event => event.words), stereo
                    ? [[8,6,42,n,0,0,0,0], [8,6,(Math.imul(42,2654435761)^0x9e3779b9)>>>0,n,1,0,0,0]]
                    : [[8,6,42,n,0,0,0,0]]);
                const passes = events.filter(event => event.kind === 'pass');
                assert.deepEqual(passes.map(pass => pass.pipeline), stereo ? [0,1,2,3,0,1,2,3,0,4,0,4] : [0,1,2,3]);
                const clears = events.filter(event => event.kind === 'clear');
                assert.equal(clears.length, stereo ? 5 : 1);
                for (const pass of passes) {
                    assert.equal(pass.groups, pass.pipeline === 3 ? Math.ceil(n/256) : pass.pipeline === 4 ? Math.ceil(2*n/256) : 1);
                    assert.ok(pass.resources.every(resource => !Object.values(previous).includes(resource.buffer)), 'previous never bound writable');
                    if (pass.pipeline === 0) assert.deepEqual(events[events.indexOf(pass)-1], {kind:'clear',args:[warp.scratch[0]]});
                }
                for (let eye = 0; eye < (stereo ? 2 : 1); eye++) {
                    const suffix = eye ? 'R' : 'L';
                    const update = passes[eye*4+3];
                    assert.equal(update.resources[0].buffer, warp.uniforms[eye]);
                    assert.equal(update.resources[1], fields['temporal'+suffix]);
                    assert.equal(update.resources[5].buffer, output['pos'+suffix]);
                    assert.equal(update.resources[6].buffer, output['meta'+suffix]);
                    assert.equal(update.resources[7].buffer, output.counters);
                }
                if (stereo) {
                    assert.deepEqual(clears[2].args, [output.mask, 0, (2*n+3)*4]);
                    assert.ok(events.indexOf(clears[2]) > events.indexOf(passes[7]));
                    for (let eye = 0; eye < 2; eye++) {
                        const own = eye ? 'R' : 'L', other = eye ? 'L' : 'R';
                        const splat = passes[8+eye*2], merge = passes[9+eye*2];
                        assert.equal(splat.resources[1], fields['cross'+other]);
                        assert.equal(merge.resources[0].buffer, warp.uniforms[eye]);
                        assert.equal(merge.resources[1], fields['cross'+other]);
                        assert.deepEqual(merge.resources.slice(3).map(resource => resource.buffer), [output['pos'+own],output['meta'+own],output['pos'+other],output['meta'+other],output['mergedPos'+own],output['mergedMeta'+own],output.mask]);
                    }
                }
            }
        }
        events.length = 0;
        assert.throws(() => warp.encode(encoder, previous, previous, fields, 42, false, {numStars:257}), /must not alias/);
        assert.throws(() => warp.encode(encoder, previous, {...output, mask:previous.posL}, fields, 42, false, {numStars:257}), /must not alias/);
        for (const n of [0, -1, 1.5, NaN, MAX_STARS+1]) {
            assert.throws(() => warp.encode(encoder, previous, output, fields, 42, false, {numStars:n}), /numStars/);
        }
        assert.equal(events.length, 0, 'reject before recording or uploading');
        warp.encode(encoder, previous, output, fields, 42, false, {numStars:257});
        assert.equal(events.find(event => event.kind === 'write').words[3], 257, 'explicit count is uploaded');
        warp.destroy();
        assert.ok([...warp.scratch,...warp.uniforms].every(buffer => buffer.destroyed));
        assert.ok([...Object.values(previous),...Object.values(output)].every(buffer => !buffer.destroyed));
        destroyStarState(previous);
        destroyStarState(output);
        assert.ok(gpu.buffers.every(buffer => buffer.destroyed));
    } finally {
        if (originalUsage === undefined) delete globalThis.GPUBufferUsage;
        else globalThis.GPUBufferUsage = originalUsage;
    }
}

test('stars: upstream CPU seed uploads and mock GPU ownership/dispatch contract', verifyStars);
