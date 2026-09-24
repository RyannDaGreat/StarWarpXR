/** Retained upstream tent/disc coordinate rasterization, without sprite-atlas branches. */
export const starRenderWGSL = /* wgsl */`
struct StarRenderUniforms {
    W:         u32,
    H:         u32,
    numStars:  u32,
    aa:        u32,
    radius:    f32,   // tent radius in texels (integer-valued for exactness)
    hardHalf:  f32,   // half-extent of the hard (non-AA) quad in texels
    _padding0: u32,
    _padding1: f32,
    colorQ:    u32,   // 1: tint tents by turbo(q) — blue fresh, red near death
    sizeQ:     u32,   // 1: scale every star's footprint by its strength q
    sizeMaxPx: f32,   // q-size mode: full width in texels of a fresh (q~0) star
    eyeBit:    u32,   // stereo merged render: 1 = left eye, 2 = right; 0 = mono (no mask)
    cullOrphans: u32, // 1: skip stars not visible in BOTH eyes (mask != 3)
}

// Smallest footprint scale in q-size mode: a newborn (q ~ 0) star must still
// cover at least a couple of texels or it disappears entirely.
const SIZE_Q_MIN = 0.15;

struct StarMeta {
    q:  f32,
    id: u32,
}

// Pure function. Google's Turbo approximation (Mikhailov 2019), t -> RGB.
// Example: turboQ(0.0) -> approximately (0.136, 0.091, 0.107).
fn turboQ(t: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    let v4 = vec4f(1.0, x, x * x, x * x * x);
    let v2 = vec2f(v4.w * x, v4.w * x * x);
    return clamp(vec3f(
        dot(v4, vec4f(0.13572138,  4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2f(-152.94239396,  59.28637943)),
        dot(v4, vec4f(0.09140261,  2.19418839,   4.84296658, -14.18503333)) + dot(v2, vec2f(   4.27729857,   2.82956604)),
        dot(v4, vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2f( -89.90310912,  27.34824973))),
        vec3f(0.0), vec3f(1.0));
}

@group(0) @binding(0) var<uniform> u: StarRenderUniforms;
@group(0) @binding(1) var<storage, read> stars: array<f32>;
@group(0) @binding(2) var<storage, read> starMeta: array<StarMeta>;
// Stereo merged rendering: per-candidate eye-visibility bits (1 = L, 2 = R)
// written by mergeSelect; mono renders bind it too but never read (eyeBit 0).
@group(0) @binding(5) var<storage, read> eyeMask: array<u32>;

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) starPos: vec2f,
    @location(2) @interpolate(flat) qv: f32,        // strength, for turbo tint
    @location(3) @interpolate(flat) effRadius: f32, // per-star tent radius (q-size)
}

// Query. Read coordinate/meta/mask buffers and construct a rasterization vertex.
@vertex fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
    let star = vid / 6u;
    let corner = vid % 6u;
    var out: VsOut;
    var skip = star >= u.numStars;
    if (!skip && u.eyeBit != 0u) {
        // Merged stereo buffers are candidate-indexed: draw only candidates
        // this eye saw — and in Cull Orphans mode, only those BOTH eyes see.
        let m = eyeMask[star];
        skip = (m & u.eyeBit) == 0u || (u.cullOrphans == 1u && (m & 3u) != 3u);
    }
    if (skip) {
        out.position = vec4f(2.0, 2.0, 0.0, 1.0);  // degenerate, off-screen
        out.starPos = vec2f(0.0);
        out.qv = 0.0;
        out.effRadius = 1.0;
        return out;
    }
    let pos = vec2f(stars[star * 2u], stars[star * 2u + 1u]);
    let q = starMeta[star].q;

    // Cover the full tent support (+0.5 reaches every participating texel center).
    // Q-size mode draws shrinking discs: sizeMaxPx * max(1-q, SIZE_Q_MIN).
    var half: f32;
    var rad = u.radius;
    if (u.sizeQ == 1u) {
        // Size = remaining life: fresh stars (q ~ 0) render at the slider's
        // full width and SHRINK as crowding erodes q toward death at 1.
        rad = max(0.5 * u.sizeMaxPx * max(1.0 - q, SIZE_Q_MIN), 0.25);
        half = rad;
        half += 1.0; // Room for the AA rim.
    } else {
        half = select(u.hardHalf, u.radius + 0.5, u.aa == 1u);
    }
    out.qv = q;
    out.effRadius = rad;
    // Triangle-list corners: (-1,-1) (1,-1) (-1,1) / (1,-1) (1,1) (-1,1)
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f( 1.0, -1.0), vec2f(1.0,  1.0), vec2f(-1.0, 1.0),
    );
    let corner_px = pos + offsets[corner] * half;

    // Pixel coords -> NDC (row 0 is the top => NDC y = +1).
    let ndc = vec2f(corner_px.x / f32(u.W) * 2.0 - 1.0,
                    1.0 - corner_px.y / f32(u.H) * 2.0);
    out.position = vec4f(ndc, 0.0, 1.0);
    out.starPos = pos;
    return out;
}

// Query. Read draw settings and evaluate the retained tent/disc coverage kernel.
@fragment fn fs(in: VsOut) -> @location(0) vec4f {

    var w: f32 = 1.0;
    if (u.sizeQ == 1u) {
        // Solid DISC with a ~1px antialiased rim. (Scaling the AA tent kernel
        // up instead renders the filter itself: a radial-gradient blob.)
        let d = distance(in.position.xy, in.starPos);
        w = clamp(in.effRadius - d + 0.5, 0.0, 1.0);
    } else if (u.aa == 1u) {
        // in.position.xy is the fragment's pixel-center coordinate — the same
        // space the star position lives in, so the tent needs no offsets.
        let d = abs(in.position.xy - in.starPos);
        w = max(0.0, 1.0 - d.x / in.effRadius) * max(0.0, 1.0 - d.y / in.effRadius);
    }
    // Alpha carries coverage; RGB and coverage accumulate additively.
    let tint = select(vec3f(1.0), turboQ(in.qv), u.colorQ == 1u);
    return vec4f(tint * w, w);
}
`;

