import test from 'node:test';
import assert from 'node:assert/strict';
import { mat4 } from 'gl-matrix';
import { worldPose, poseRay, aimFromHand, panelHit, panelPose } from '../src/xr/math.js';
import { sceneLights } from '../src/scenes/sandbox/index.js';
import { toGLProjection } from '../src/xr/projection.js';

/** Command. Verify native XR WebGPU depth maps to the renderer's GL convention without mutation. */
test('WebGPU XR projection converts near/far to GL clip depth exactly once', () => {
  const near = .1, far = 10000;
  const gpu = mat4.perspectiveZO(mat4.create(), Math.PI / 2, 1, near, far);
  const original = Array.from(gpu);
  const gl = toGLProjection(gpu);
  const reference = mat4.perspectiveNO(mat4.create(), Math.PI / 2, 1, near, far);
  for (let i = 0; i < 16; i++) assert.ok(Math.abs(gl[i] - reference[i]) < 1e-6);
  assert.deepEqual(Array.from(gpu), original);
  for (const [z, expected] of [[-near, -1], [-far, 1]]) {
    const depth = (gl[10] * z + gl[14]) / (gl[11] * z + gl[15]);
    assert.ok(Math.abs(depth - expected) < 1e-6);
  }
  assert.equal(toGLProjection(mat4.create())[14], -1);
});

/** Command. Check coordinate invariants and representative hand/panel interaction examples. */
test('XR coordinates: floor translation, aim, rigid head panel and rejection', () => {
  const pose = mat4.create();
  const world = worldPose(pose, [2, 0, 6]);
  assert.deepEqual(Array.from(pose), Array.from(mat4.create()));
  assert.equal(world[12], 2);
  assert.equal(world[14], 6);
  assert.deepEqual(poseRay(world).origin, [2, 0, 6]);
  assert.deepEqual(poseRay(world).direction.map(value => value || 0), [0, 0, -1]);
  assert.deepEqual(aimFromHand({ origin: [0, 0, 0], direction: [0, 0, -1] }, [0, 0, -1]), [0, 0, -1]);
  const aim = aimFromHand({ origin: [0, 1.6, 0], direction: [0, 0, -1] }, [0.3, 1.1, -0.2]);
  assert.ok(Math.abs(Math.hypot(...aim) - 1) < 1e-6);
  assert.ok(aim[0] < 0 && aim[1] > 0 && aim[2] < 0);
  const ray = { origin: [0, 0, 2], direction: [0, 0, -1] };
  assert.deepEqual(panelHit(ray, pose), [.5, .5]);
  assert.equal(panelHit({ ...ray, origin: [1, 0, 2] }, pose), null);
  assert.equal(panelHit({ ...ray, direction: [0, 0, 1] }, pose), null);
  assert.equal(panelHit({ ...ray, direction: [1, 0, 0] }, pose), null);
  const head = mat4.fromTranslation(mat4.create(), [2, 1.6, 6]);
  mat4.rotateY(head, head, Math.PI / 3);
  const panel = panelPose(head);
  const direction = [panel[12] - head[12], panel[13] - head[13], panel[14] - head[14]];
  assert.ok(Math.abs(panelHit({ origin: [head[12], head[13], head[14]], direction }, panel)[0] - .5) < 1e-5);
  assert.equal(panelPose(mat4.create())[14], -1.25);
});

/** Command. Verify the standalone scene-light transformation example and shader capacity. */
test('sandbox emissive object lights retain original priorities', () => {
  const ball = { pos: { x: 1, y: 2, z: 3 } };
  assert.deepEqual(sceneLights({ spheres: [ball], mushrooms: [] })[0].pos, [1, 2, 3]);
  assert.equal(sceneLights({ spheres: Array(40).fill(ball), mushrooms: [] }).length, 32);
  assert.equal(sceneLights({ spheres: [], mushrooms: [ball, ball, ball, ball] }).length, 2);
});
