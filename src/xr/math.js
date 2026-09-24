import { mat4, vec3 } from 'gl-matrix';

/**
 * Pure function. Translate a tracked pose into world space without altering it.
 * @param {ArrayLike<number>} pose - Column-major 4×4 local pose.
 * @param {number[]} origin - World offset [x,y,z], in metres.
 * @returns {Float32Array} World pose, shape (16,).
 * @example worldPose(mat4.create(), [2,0,6])[12] // 2
 */
export function worldPose(pose, origin) {
  const world = mat4.clone(pose);
  world[12] += origin[0]; world[13] += origin[1]; world[14] += origin[2];
  return world;
}

/**
 * Pure function. Extract an origin and normalized -Z ray from a rigid pose.
 * @param {ArrayLike<number>} pose - Column-major rigid 4×4 matrix.
 * @returns {{origin:number[], direction:number[]}} World xyz coordinates.
 * @example poseRay(mat4.create()) // {origin:[0,0,0], direction:[0,0,-1]}
 */
export function poseRay(pose) {
  return { origin: [pose[12], pose[13], pose[14]], direction: [-pose[8], -pose[9], -pose[10]] };
}

/**
 * Pure function. Aim a hand-launched ball at a point on the selection ray.
 * @param {{origin:number[], direction:number[]}} ray - Unit gaze/controller ray.
 * @param {number[]} hand - Hand world position [x,y,z].
 * @param {number} distance - Distance along the selection ray, in metres.
 * @returns {number[]} Unit direction [dx,dy,dz].
 * @example aimFromHand({origin:[0,0,0],direction:[0,0,-1]}, [0,0,-1], 10) // [0,0,-1]
 */
export function aimFromHand(ray, hand, distance = 30) {
  const target = vec3.scaleAndAdd(vec3.create(), ray.origin, ray.direction, distance);
  return Array.from(vec3.normalize(target, vec3.subtract(target, target, hand)));
}

/**
 * Pure function. Hit-test a transformed unit panel, returning its texture coordinates.
 * @param {{origin:number[],direction:number[]}} ray - World-space selection ray.
 * @param {ArrayLike<number>} panel - Matrix mapping unit square to world space.
 * @returns {number[]|null} [u,v] in [0,1), v increasing upward; null outside the panel.
 * @example panelHit({origin:[0,0,2],direction:[0,0,-1]}, mat4.create()) // [0.5,0.5]
 */
export function panelHit(ray, panel) {
  const inverse = mat4.invert(mat4.create(), panel);
  if (!inverse) throw new Error('Panel transform must be invertible');
  const start = vec3.transformMat4(vec3.create(), ray.origin, inverse);
  const end = vec3.transformMat4(vec3.create(), vec3.add(vec3.create(), ray.origin, ray.direction), inverse);
  const direction = vec3.subtract(end, end, start);
  if (direction[2] >= 0) return null;
  const distance = -start[2] / direction[2];
  if (distance < 0) return null;
  const hit = vec3.scaleAndAdd(start, start, direction, distance);
  return Math.abs(hit[0]) < 0.5 && Math.abs(hit[1]) < 0.5 ? [hit[0] + 0.5, hit[1] + 0.5] : null;
}

/**
 * Pure function. Place a level, head-position-following panel below the horizon.
 * @param {ArrayLike<number>} head - World head pose, column-major 4×4.
 * @returns {Float32Array} Unit-panel world transform (16,).
 * @example panelPose(mat4.create())[14] // -1.25
 */
export function panelPose(head) {
  const distance = 1.25, drop = 0.55, width = 1.4, height = 0.3;
  const yaw = Math.atan2(head[8], head[10]);
  const model = mat4.fromTranslation(mat4.create(), [head[12], head[13], head[14]]);
  mat4.rotateY(model, model, yaw);
  mat4.translate(model, model, [0, -drop, -distance]);
  return mat4.scale(model, model, [width, height, 1]);
}
