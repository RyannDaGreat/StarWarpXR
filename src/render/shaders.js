// Scene SBS and linear-light star coverage display; no fusion borders or convergence shift.
export const displayWGSL = /* wgsl */`
struct VsOut { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VsOut {
    var out: VsOut; out.position = vec4f(pos,0.0,1.0); out.uv = uv; return out;
}
struct Display { mode:u32, W:u32, H:u32, stereoMode:u32 }
@group(0) @binding(0) var<uniform> disp:Display;
@group(0) @binding(1) var sceneL:texture_2d<f32>;
@group(0) @binding(2) var sceneR:texture_2d<f32>;
@group(0) @binding(3) var starsL:texture_2d<f32>;
@group(0) @binding(4) var starsR:texture_2d<f32>;
fn srgbEncode(c:f32) -> f32 {
    if (c <= 0.0031308) { return 12.92*c; }
    return 1.055*pow(c,1.0/2.4)-0.055;
}
@fragment fn fs(in:VsOut) -> @location(0) vec4f {
    let width = select(disp.W,2u*disp.W,disp.stereoMode==2u);
    let x=min(u32(in.uv.x*f32(width)),width-1u);
    let y=min(u32(in.uv.y*f32(disp.H)),disp.H-1u);
    let right=x>=disp.W;
    let p=vec2u(x%disp.W,y);
    if (disp.mode==1u) {
        if (right) { return textureLoad(sceneR,p,0); }
        return textureLoad(sceneL,p,0);
    }
    var rgb:vec3f;
    if (right) { rgb=textureLoad(starsR,p,0).rgb; }
    else { rgb=textureLoad(starsL,p,0).rgb; }
    rgb=clamp(rgb,vec3f(0.0),vec3f(1.0));
    return vec4f(srgbEncode(rgb.r),srgbEncode(rgb.g),srgbEncode(rgb.b),1.0);
}
`;
