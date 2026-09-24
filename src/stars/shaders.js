const starCommonWGSL = /* wgsl */`
struct StarUniforms {
    W:         u32,
    H:         u32,
    frameSeed: u32,
    numStars:  u32,
    flags:     u32,   // bit 0: this merge's OWN stream is the right eye
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

// Per-star metadata: strength q (eroded by crowding, dies at 1) + identity.
struct StarMeta {
    q:  f32,
    id: u32,
}

fn pcg(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand01(state: ptr<function, u32>) -> f32 {
    *state = pcg(*state);
    return f32(*state) / 4294967296.0;
}
`;

export const starSplatWGSL = /* wgsl */`
${starCommonWGSL}

@group(0) @binding(0) var<uniform> u: StarUniforms;
@group(0) @binding(1) var motionTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> density: array<atomic<u32>>;

fn atomicAddF32(addr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old = atomicLoad(addr);
    loop {
        let newVal = bitcast<u32>(bitcast<f32>(old) + val);
        let result = atomicCompareExchangeWeak(addr, old, newVal);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= u.W * u.H) { return; }
    let row = idx / u.W;
    let col = idx % u.W;

    let m = textureLoad(motionTex, vec2u(col, row), 0);
    let fx = m.r * f32(u.W);
    let fy = -m.g * f32(u.H);

    // Pixel center (col+0.5, row+0.5) lands at center + flow; splat position in
    // texel-index space is that minus 0.5.
    let px = f32(col) + fx;
    let py = f32(row) + fy;
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let fxw = px - f32(x0);
    let fyw = py - f32(y0);

    for (var c: u32 = 0u; c < 4u; c++) {
        let dx = i32(c & 1u);
        let dy = i32(c >> 1u);
        let x = x0 + dx;
        let y = y0 + dy;
        let w = select(1.0 - fxw, fxw, dx == 1) * select(1.0 - fyw, fyw, dy == 1);
        if (x >= 0 && x < i32(u.W) && y >= 0 && y < i32(u.H) && w > 0.0) {
            atomicAddF32(&density[u32(y) * u.W + u32(x)], w);
        }
    }
}
`;

// Per-row inclusive prefix sums of the deficit max(1 - E, 0).
// Last column of each row is that row's total deficit.
export const starScanRowsWGSL = /* wgsl */`
${starCommonWGSL}

@group(0) @binding(0) var<uniform> u: StarUniforms;
@group(0) @binding(1) var<storage, read>       density:   array<f32>;
@group(0) @binding(2) var<storage, read_write> rowPrefix: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    if (row >= u.H) { return; }
    var acc: f32 = 0.0;
    for (var col: u32 = 0u; col < u.W; col++) {
        acc += max(1.0 - density[row * u.W + col], 0.0);
        rowPrefix[row * u.W + col] = acc;
    }
}
`;

// Inclusive prefix over row totals -> row CDF. Last entry = grand total deficit.
// Single thread: H is at most a few thousand; this pass is microseconds.
export const starScanCdfWGSL = /* wgsl */`
${starCommonWGSL}

@group(0) @binding(0) var<uniform> u: StarUniforms;
@group(0) @binding(1) var<storage, read>       rowPrefix: array<f32>;
@group(0) @binding(2) var<storage, read_write> rowCdf:    array<f32>;

@compute @workgroup_size(1)
fn main() {
    var acc: f32 = 0.0;
    for (var row: u32 = 0u; row < u.H; row++) {
        acc += rowPrefix[row * u.W + u.W - 1u];
        rowCdf[row] = acc;
    }
}
`;

// Per-star: advect along the flow, die out-of-frame or by strength exhaustion —
// each star's strength q (U[0,1) at birth) is multiplied by max(E,1) at the NEW
// position and the star dies at q >= 1. Deterministic, RNG-free death: exactly
// the coin rule "survive w.p. 1/max(E,1)" via inverse-CDF sampling of the death
// time (alive after n frames iff q < prod 1/max(E,1)). RNG is only used for
// respawns from the deficit CDF. Do NOT threshold a fixed q per frame instead
// of eroding: survivors would become immune to repeat thinning and persistent
// contraction collapses all stars into a clump.
export const starUpdateWGSL = /* wgsl */`
${starCommonWGSL}

@group(0) @binding(0) var<uniform> u: StarUniforms;
@group(0) @binding(1) var motionTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read>       density:   array<f32>;
@group(0) @binding(3) var<storage, read>       rowPrefix: array<f32>;
@group(0) @binding(4) var<storage, read>       rowCdf:    array<f32>;
@group(0) @binding(5) var<storage, read_write> stars:     array<f32>;
@group(0) @binding(6) var<storage, read_write> starMeta:  array<StarMeta>;
// counters[0] = fresh-birth id mint, counters[1] = cumulative deaths (diagnostics)
@group(0) @binding(7) var<storage, read_write> counters:  array<atomic<u32>>;

// Bilinear sample of the motion texture at continuous pixel position p,
// border-clamped in texel-index space.
fn sampleMotion(p: vec2f) -> vec2f {
    let maxIdx = vec2f(f32(u.W) - 1.0, f32(u.H) - 1.0);
    let q = clamp(p - 0.5, vec2f(0.0), maxIdx);
    let q0 = clamp(floor(q), vec2f(0.0), maxIdx - vec2f(1.0));
    let f = clamp(q - q0, vec2f(0.0), vec2f(1.0));
    let x0 = u32(q0.x); let y0 = u32(q0.y);
    let m00 = textureLoad(motionTex, vec2u(x0,      y0     ), 0).rg;
    let m10 = textureLoad(motionTex, vec2u(x0 + 1u, y0     ), 0).rg;
    let m01 = textureLoad(motionTex, vec2u(x0,      y0 + 1u), 0).rg;
    let m11 = textureLoad(motionTex, vec2u(x0 + 1u, y0 + 1u), 0).rg;
    return mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
}

// Bilinear sample of the density buffer at continuous pixel position p.
fn sampleDensity(p: vec2f) -> f32 {
    let maxIdx = vec2f(f32(u.W) - 1.0, f32(u.H) - 1.0);
    let q = clamp(p - 0.5, vec2f(0.0), maxIdx);
    let q0 = clamp(floor(q), vec2f(0.0), maxIdx - vec2f(1.0));
    let f = clamp(q - q0, vec2f(0.0), vec2f(1.0));
    let x0 = u32(q0.x); let y0 = u32(q0.y);
    let d00 = density[y0 * u.W + x0];
    let d10 = density[y0 * u.W + x0 + 1u];
    let d01 = density[(y0 + 1u) * u.W + x0];
    let d11 = density[(y0 + 1u) * u.W + x0 + 1u];
    return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

// Smallest index in [0, n) whose inclusive-prefix value exceeds t.
fn lowerBound(base: u32, n: u32, t: f32, isRowCdf: bool) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = n - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        var v: f32;
        if (isRowCdf) { v = rowCdf[base + mid]; } else { v = rowPrefix[base + mid]; }
        if (v > t) { hi = mid; } else { lo = mid + 1u; }
    }
    return lo;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.numStars) { return; }

    var pos = vec2f(stars[i * 2u], stars[i * 2u + 1u]);
    var q = starMeta[i].q;
    var id = starMeta[i].id;
    var rng: u32 = pcg(u.frameSeed * 104729u + i);  // births only; death is RNG-free

    // Advect: flow sampled at the OLD position.
    let m = sampleMotion(pos);
    pos += vec2f(m.x * f32(u.W), -m.y * f32(u.H));

    let domain = vec2f(f32(u.W), f32(u.H));
    var dead = pos.x < 0.0 || pos.x > domain.x || pos.y < 0.0 || pos.y > domain.y;

    if (!dead) {
        // Crowding at the NEW position erodes the star's strength; die at 1.
        // E <= 1 (uncrowded covered space) leaves q untouched — never kills.
        let e = sampleDensity(pos);
        q *= max(e, 1.0);
        if (q >= 1.0) { dead = true; }
    }

    if (dead) {
        atomicAdd(&counters[1], 1u);   // diagnostics: cumulative death count
        q = rand01(&rng);  // fresh strength at birth
        let total = rowCdf[u.H - 1u];
        if (total <= 1e-6) {
            // No deficit anywhere: respawn uniformly (the quota must hold).
            pos = vec2f(rand01(&rng) * domain.x, rand01(&rng) * domain.y);
            id = atomicAdd(&counters[0], 1u);
        } else {
            // Two-level inverse-CDF: row from the row CDF, column within the row.
            let t = rand01(&rng) * total;
            let row = lowerBound(0u, u.H, t, true);
            let tIn = t - select(0.0, rowCdf[row - 1u], row > 0u);
            let col = lowerBound(row * u.W, u.W, tIn, false);
            // Tent jitter within the cell keeps positions continuous while each
            // cell receives exactly its share of birth mass; reflect at borders.
            pos = vec2f(f32(col) + 0.5 + rand01(&rng) - rand01(&rng),
                        f32(row) + 0.5 + rand01(&rng) - rand01(&rng));
            pos = abs(pos);
            pos = domain - abs(domain - pos);
            id = atomicAdd(&counters[0], 1u);
        }
    }

    stars[i * 2u]      = pos.x;
    stars[i * 2u + 1u] = pos.y;
    starMeta[i]        = StarMeta(q, id);
}
`;

// ---------------------------------------------------------------------------
// Stereo merge select (one per eye per frame): the kept half of the user's
// hcat warp step, with the provably-dead birth stage skipped. Threads 0..N-1
// are OWN-stream stars (zero flow); threads N..2N-1 are OTHER-stream stars
// advected by the cross-eye flow. Survival is the deterministic v3 threshold
// q * E < 1 with E = 1 (own half's exact zero-flow self-splat) + the OTHER
// view's transported density (splatted by starSplat into `density` from the
// cross texture). Copies only — the per-eye streams are never mutated; shared
// q's couple both eyes' survival decisions through the same strength samples.
//
// Outputs are CANDIDATE-INDEXED (canon id = L-star j -> j, R-star j -> N+j;
// both eyes agree). Each eye writes its landing position/meta at [canon] and
// sets its bit in the shared mask; the render pass draws 2N quads and skips
// unset (or, in shared-only mode, non-both-eyes) candidates. mask tail:
// [2N] = left survivor count, [2N+1] = right, [2N+2] = shared (counted by the
// second merge pass via atomicOr's returned prior bits).
// ---------------------------------------------------------------------------
export const mergeSelectWGSL = /* wgsl */`
${starCommonWGSL}

@group(0) @binding(0) var<uniform> u: StarUniforms;
@group(0) @binding(1) var crossTex: texture_2d<f32>;   // other eye -> this eye flow
@group(0) @binding(2) var<storage, read> density:   array<f32>;   // transported other-view mass
@group(0) @binding(3) var<storage, read> ownPos:    array<f32>;
@group(0) @binding(4) var<storage, read> ownMeta:   array<StarMeta>;
@group(0) @binding(5) var<storage, read> otherPos:  array<f32>;
@group(0) @binding(6) var<storage, read> otherMeta: array<StarMeta>;
@group(0) @binding(7) var<storage, read_write> mergedPos:  array<f32>;
@group(0) @binding(8) var<storage, read_write> mergedMeta: array<StarMeta>;
@group(0) @binding(9) var<storage, read_write> mask: array<atomic<u32>>;

fn sampleCross(p: vec2f) -> vec2f {
    let maxIdx = vec2f(f32(u.W) - 1.0, f32(u.H) - 1.0);
    let q = clamp(p - 0.5, vec2f(0.0), maxIdx);
    let q0 = clamp(floor(q), vec2f(0.0), maxIdx - vec2f(1.0));
    let f = clamp(q - q0, vec2f(0.0), vec2f(1.0));
    let x0 = u32(q0.x); let y0 = u32(q0.y);
    let m00 = textureLoad(crossTex, vec2u(x0,      y0     ), 0).rg;
    let m10 = textureLoad(crossTex, vec2u(x0 + 1u, y0     ), 0).rg;
    let m01 = textureLoad(crossTex, vec2u(x0,      y0 + 1u), 0).rg;
    let m11 = textureLoad(crossTex, vec2u(x0 + 1u, y0 + 1u), 0).rg;
    return mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
}

fn sampleDensity(p: vec2f) -> f32 {
    let maxIdx = vec2f(f32(u.W) - 1.0, f32(u.H) - 1.0);
    let q = clamp(p - 0.5, vec2f(0.0), maxIdx);
    let q0 = clamp(floor(q), vec2f(0.0), maxIdx - vec2f(1.0));
    let f = clamp(q - q0, vec2f(0.0), vec2f(1.0));
    let x0 = u32(q0.x); let y0 = u32(q0.y);
    let d00 = density[y0 * u.W + x0];
    let d10 = density[y0 * u.W + x0 + 1u];
    let d01 = density[(y0 + 1u) * u.W + x0];
    let d11 = density[(y0 + 1u) * u.W + x0 + 1u];
    return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = u.numStars;
    if (i >= 2u * n) { return; }
    let ownIsRight = (u.flags & 1u) != 0u;
    let canon = select(i, (i + n) % (2u * n), ownIsRight);

    var pos: vec2f;
    var m: StarMeta;
    if (i < n) {
        pos = vec2f(ownPos[i * 2u], ownPos[i * 2u + 1u]);   // own half: zero flow
        m = ownMeta[i];
    } else {
        let j = i - n;
        var p = vec2f(otherPos[j * 2u], otherPos[j * 2u + 1u]);
        let f = sampleCross(p);
        pos = p + vec2f(f.x * f32(u.W), -f.y * f32(u.H));   // reproject into this eye
        m = otherMeta[j];
    }

    let domain = vec2f(f32(u.W), f32(u.H));
    if (pos.x < 0.0 || pos.x > domain.x || pos.y < 0.0 || pos.y > domain.y) { return; }

    let E = 1.0 + max(sampleDensity(pos), 0.0);
    if (m.q * E >= 1.0) { return; }                          // deterministic thin to uniform

    mergedPos[canon * 2u]      = pos.x;
    mergedPos[canon * 2u + 1u] = pos.y;
    mergedMeta[canon] = m;
    let eyeBit = select(1u, 2u, ownIsRight);
    let prior = atomicOr(&mask[canon], eyeBit);
    atomicAdd(&mask[2u * n + select(0u, 1u, ownIsRight)], 1u);
    if ((prior | eyeBit) == 3u) {                            // both eyes now: shared
        atomicAdd(&mask[2u * n + 2u], 1u);
    }
}
`;

