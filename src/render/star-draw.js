import { starRenderWGSL } from './star-shaders.js';

/** Rasterization only: consumes coordinates without changing the star algorithm state. */
export class StarDraw {
    /**
     * Command. Create per-eye linear-light coverage textures and the upstream tent pipeline.
     * @param {GPUDevice} device - GPU owner.
     * @param {number} W - Width in pixels.
     * @param {number} H - Height in pixels.
     * @example new StarDraw(device,1024,1024) // rasterizer with two texture views
     */
    constructor(device,W,H) {
        this.device=device; this.W=W; this.H=H;
        this.starAAEnabled=true; this.starColorQEnabled=false; this.starSizeQEnabled=false;
        this.starSizeMaxPx=8; this.cullOrphansEnabled=false;
        this.textures=[0,1].map(()=>device.createTexture({size:[W,H],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}));
        this.views=this.textures.map(texture=>texture.createView());
        this.uniforms=[0,1].map(()=>device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));
        const module=device.createShaderModule({code:starRenderWGSL});
        const blend={srcFactor:'one',dstFactor:'one',operation:'add'};
        this.pipeline=device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format:'rgba16float',blend:{color:blend,alpha:blend}}]},primitive:{topology:'triangle-list'}});
    }

    /**
     * Command. Rasterize coordinate buffers into [H,W,4] linear-light RGBA coverage textures.
     * @param {GPUCommandEncoder} encoder - Caller-owned stream; submit before next encode.
     * @param {object} state - Star coordinate state; never mutated here.
     * @param {number} numStars - Active count in each source stream.
     * @param {boolean} stereo - Select candidate-indexed merged coordinates and visibility masks.
     * @returns {GPUTextureView[]} Left/right coverage views.
     * @example draw.encode(encoder,state,10000,true) // [leftView,rightView]
     */
    encode(encoder,state,numStars,stereo) {
        const {device,W,H}=this;
        for (let eye=0;eye<(stereo?2:1);eye++) {
            const data=new ArrayBuffer(64), u=new Uint32Array(data), f=new Float32Array(data);
            u[0]=W;u[1]=H;u[2]=numStars*(stereo?2:1);u[3]=this.starAAEnabled?1:0;
            f[4]=Math.max(1,Math.round(W/1024));f[5]=Math.max(0.5,W/2048);
            u[8]=this.starColorQEnabled?1:0;u[9]=this.starSizeQEnabled?1:0;
            f[10]=this.starSizeMaxPx*(W/1024);u[11]=stereo?eye+1:0;u[12]=this.cullOrphansEnabled?1:0;
            device.queue.writeBuffer(this.uniforms[eye],0,data);
            const suffix=eye?'R':'L', prefix=stereo?'merged':'';
            const pos=state[stereo?prefix+'Pos'+suffix:'posL'];
            const meta=state[stereo?prefix+'Meta'+suffix:'metaL'];
            const group=device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
                {binding:0,resource:{buffer:this.uniforms[eye]}},{binding:1,resource:{buffer:pos}},
                {binding:2,resource:{buffer:meta}},{binding:5,resource:{buffer:state.mask}},
            ]});
            const pass=encoder.beginRenderPass({colorAttachments:[{view:this.views[eye],loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
            pass.setPipeline(this.pipeline);pass.setBindGroup(0,group);pass.draw(u[2]*6);pass.end();
        }
        return this.views;
    }

    /** Command. Destroy owned textures and uniforms, not the shared device. */
    destroy() { for (const resource of [...this.textures,...this.uniforms]) resource.destroy(); }
}
