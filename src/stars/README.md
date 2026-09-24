# Standalone star coordinates

This module advances coordinate buffers, not images. It imports only its own
WGSL kernels: no scene, camera, renderer, clock, global random source or frame
counter. Motion generation and drawing belong to the caller.

`StarWarp.encode` is a **GPU command**, not a pure function. It records work,
uploads uniforms and mutates scratch/output buffers. Floating-point density
addition uses compare/exchange atomics whose order depends on GPU scheduling;
new IDs use atomic allocation. Explicit seeds do **not** promise bitwise GPU
reproducibility. Previous-state buffers are read only as copy sources.

## API and ownership

```js
import {StarWarp, createStarState, destroyStarState} from './star-warp.js';

const warp = new StarWarp(device, 1024, 768);
let previous = createStarState(device, 1024, 768, 777);
let next = createStarState(device, 1024, 768, 777);
const settings = {numStars:10000};
const encoder = device.createCommandEncoder();
warp.encode(encoder, previous, next, fields, 42, true, settings); // returns next
// Optionally encode consumption of next on this encoder.
device.queue.submit([encoder.finish()]);
[previous, next] = [next, previous];
// Reuse the two allocations for later frames; destroy after their final use.
destroyStarState(previous);
destroyStarState(next);
warp.destroy();
```

- `W,H` are fixed integer dimensions ≥ 2, shared by states and motion textures.
  Use the same device for every resource. It must support at least nine storage
  buffers per compute stage (the merge kernel).
- `seed` is an explicit unsigned 32-bit frame seed. `stereo` is an explicit boolean.
  `settings.numStars = N` must be an integer in `[1, MAX_STARS]`. There is no
  stored active count: the caller passes N to coordinate evolution and drawing.
- Allocate two independent states. No buffer in `previous` may occur anywhere in
  `next`; aliases are rejected before recording. Keep each state's allocation
  layout unchanged; do not replace members with aliased or undersized buffers.
- `encode` returns immediately, not a promise or CPU coordinate array. Output is
  valid after GPU execution. The caller owns encoding, submission and readback.
- **Submit before the next call on the same adapter.** Uniform buffers and scratch
  are reused. Two encodes before submission would overwrite the uniforms used by
  the first. Sequential queue submissions need no CPU `onSubmittedWorkDone` wait.
- Caller owns both states and input textures; `warp.destroy()` releases only its
  scratch/uniforms. `destroyStarState(state)` releases all ten state buffers.

## Inputs and axes

Positions are continuous pixel coordinates `(x,y)` with x rightward and y
downward; `(0,0)` is the upper-left boundary and pixel `(col,row)` has center
`(col+0.5,row+0.5)`. The kernels treat `[0,W] × [0,H]` as in-bounds.

`fields` contains texture views with logical shape **[H,W,4]**, e.g.
`[768,1024,4]`, readable via `texture_2d<f32>` (normally rgba32float):

| Member | Source → destination |
| --- | --- |
| `temporalL` | Previous left view → current left view |
| `temporalR` | Previous right view → current right view; stereo only |
| `crossL` | Current left view → current right view; stereo only |
| `crossR` | Current right view → current left view; stereo only |

At each source pixel, channels `(r,g) = (dx/W, -dy/H)`; b,a are unused.
For example moving 8 pixels right and 3 pixels down at 1024×768 uses
`(8/1024,-3/768,0,0)`. Motion sampling is bilinear with clamped borders.
Mono requires only `temporalL`; stereo requires all four views. Input fields
must be ready before their compute passes execute.

## State and output buffers

`MAX_STARS = 1 << 20` is capacity, **not** the active count N. All buffers have
STORAGE, COPY_SRC and COPY_DST usage. For example N=10000 means the first
10000 entries of each active stream are consumed, not the entire capacity.

| Members | Allocated layout | Meaning |
| --- | --- | --- |
| `posL`, `posR` | [MAX_STARS,2] float32 | `(x,y)` pixels |
| `metaL`, `metaR` | [MAX_STARS] 8-byte records | interleaved `{q:f32,id:u32}`; not two float32 values |
| `mergedPosL`, `mergedPosR` | [2×MAX_STARS,2] float32 | candidate-indexed `(x,y)` pixels |
| `mergedMetaL`, `mergedMetaR` | [2×MAX_STARS] 8-byte records | copied `{q:f32,id:u32}` for surviving candidates |
| `mask` | [2×MAX_STARS+4] uint32 | active visibility bits followed by counts at offsets based on **N** |
| `counters` | [2] uint32 | next fresh ID, cumulative temporal deaths; shared by L/R |

Initialization seeds the **entire capacity** using mulberry32, with L seed
`seed` and R seed `seed+1`. Each star consumes random values in x,y,q order;
q begins uniform in [0,1). IDs begin at `i` for L and `MAX_STARS+i` for R.
Counters start `[2×MAX_STARS,0]`. Merged buffers/mask start zero (WebGPU's
allocation initialization), but contain no merged result until a stereo step.

### Mono

Exactly N left-stream coordinates are updated. Right-stream data, inactive
stream tails, and counters are copied from previous; only temporal deaths
change counters. Consume `posL[0:N]` and `metaL[0:N]`. Merged buffers and mask
are **not written or valid mono output**; they may contain old stereo data.

### Stereo

Each temporal stream retains exactly N stars (dead stars are immediately
replaced). Both updates finish before either merge. The R temporal seed is
`(Math.imul(seed,2654435761) ^ 0x9e3779b9) >>> 0`; L uses `seed` directly.

Each eye considers **2N candidates**, not a packed list of 2N survivors:

- Canonical index `j` identifies left-stream star j.
- Canonical index `N+j` identifies right-stream star j, in both eyes.
- `mask[i] & 1` means candidate i survived in L; `mask[i] & 2` means R.
  `mask[i] === 3` means shared. Test the eye's bit **before** reading a merged
  position/meta: rejected entries are unwritten and may be stale.
- `mask[2N]` is the L survivor count; `[2N+1]` is R; `[2N+2]` is shared.
  The next allocated word is unused. Counts are not fixed at N: each is in
  `[0,2N]`, shared ≤ min(L,R), and union = L+R−shared.
- The mask's active candidate/count region is cleared once before both merges.
  Left merge uses `crossR` (other→own); right merge uses `crossL`. Neither merge
  modifies the temporal streams or counters. Rendering shared-only candidates
  is a downstream mask policy, not an additional simulation step.

## Retained numerical rules

The five exported WGSL kernels match the retained upstream implementation.
Temporal flow splats a unit-density pixel grid to estimate transported density
E. Row-prefix and row-CDF scans accumulate `max(1−E,0)` for births. Stars advect
using flow at their **old** positions; at the **new** positions their strength
becomes `q *= max(E,1)`. They die out-of-bounds or at `q >= 1`. Do not replace
this persistent erosion with repeated thresholds against a fixed birth q.
Births get a fresh q and ID and sample the deficit CDF with tent jitter and
border reflection; a total deficit ≤ 1e−6 selects uniform births instead.

Merging reprojects the other stream, uses `E = 1 + max(otherDensity,0)`, and
keeps candidates only when in-bounds and `q*E < 1`. It copies q unchanged:
merge selection must not erode the persistent stream's strength. Shared bits
and counts use atomics; there is no compacting or indirect-draw list.

## Allocation cost and verification limits

One state occupies about **104 MiB**; two states about **208 MiB**. Each adapter
adds `8WH + 4H + 64` bytes of scratch/uniforms. Initialization reuses a 16 MiB
CPU staging pair for both eyes; `writeBuffer` snapshots the data immediately.
Reuse states rather than constructing them every frame. Encode allocates bind
groups, but no GPU buffers or full-size CPU arrays.

Each step copies 32 MiB of temporal coordinates/metadata plus 8 counter bytes.
These full-capacity copies preserve inactive tails and the dormant right eye
when ping-ponging distinct explicit states. Copying only N would change behavior
when N increases or stereo resumes. Removing those copies needs a different
state/capacity contract; an in-place shortcut would violate previous immutability.
Merged arrays are neither copied nor cleared: the mask gates valid entries.

Run `node --test tests/stars.test.js`. Tests compare complete CPU initialization
uploads against retained-upstream digests, and use a recording mock device to
check allocation/ownership, no previous-state write bindings, seeds, count
validation, mono/stereo pass order, cross-eye routing, active mask clear size,
dispatch sizes and destruction. **The mock does not compile WGSL, execute GPU
work or validate GPU numerical results.** `npm run test:browser` additionally
executes the WGSL on a real browser WebGPU device: zero-flow mono coordinates and
metadata match their inputs byte-for-byte, previous coordinates remain unchanged,
and stereo mask bits match survivor counters (1019 shared candidates for the
1024-star/seed-777 fixture). This is not exhaustive verification of moving-field
density/strength evolution, GPU scheduling, or cross-device numerical parity.
