import { Ray, QueryFilterFlags } from '@dimforge/rapier3d-compat';
import { PhysicsWorld } from './physics.js';

const FIXED_STEP = 1 / 60;
const MAX_TICKS = 5;
const TELEPORT_RANGE = 30;
const MIN_UP_NORMAL = Math.cos(Math.PI / 4); // Maximum walkable slope: 45 degrees.
const LANDING_CLEARANCE = 0.02; // Keep the clearance capsule just above the floor.
const IDENTITY_ROTATION = { x: 0, y: 0, z: 0, w: 1 };

/**
 * Pure function. Collect the original ball and mushroom-cap point lights.
 * @param {object} data - Scene objects with xyz positions.
 * @returns {object[]} At most 32 shader-ready lights, prioritizing balls.
 * @example sceneLights({spheres:[{pos:{x:1,y:2,z:3}}],mushrooms:[]})[0].pos // [1,2,3]
 */
export function sceneLights(data) {
    const maxLights = 32;
    const balls = data.spheres.map(({ pos }) => ({ pos: [pos.x,pos.y,pos.z], color: [0.3,0.5,1], intensity: 3, radius: 15 }));
    const caps = data.mushrooms.filter((_, index) => index % 2 === 1)
        .map(({ pos }) => ({ pos: [pos.x,pos.y,pos.z], color: [0.2,0.8,0.4], intensity: 2, radius: 10 }));
    return [...balls, ...caps].slice(0, maxLights);
}

/**
 * Command. Loads Rapier and creates the unchanged legacy sandbox world.
 * @returns {Promise<object>} Scene adapter; spawn is a floor-level XR origin.
 * @example const scene = await createSandbox(); scene.spawn // [0, 0, 6]
 */
export async function createSandbox() {
    const physics = new PhysicsWorld();
    await physics.init();
    let accumulator = 0;
    let xrActive = false;

    /** Command. Restore the sandbox while retaining slow-motion and XR state. */
    function reset() {
        physics.reset();
        accumulator = 0;
        physics.playerBody.setEnabled(!xrActive);
    }

    return {
        xrControls: [{ label: 'Reset', run: reset }],
        physics,
        spawn: [0, 0, 6],

        /**
         * Command. Advances physics at 60 wall-clock ticks/second, dropping stall backlog.
         * Slow motion changes simulated time per tick, not the wall-clock accumulator.
         * @param {number} dt - Nonnegative elapsed wall time in seconds.
         * @param {object} [controls] - Desktop {forward:[x,z], right:[x,z], keys:{}}.
         * @example scene.update(1 / 90, {forward:[0,-1], right:[1,0], keys:{KeyW:true}}) // undefined
         */
        update(dt, controls = {}) {
            if (!Number.isFinite(dt) || dt < 0) throw new RangeError('dt must be finite and nonnegative');
            accumulator = Math.min(accumulator + dt, FIXED_STEP * MAX_TICKS);
            const { forward = [0, -1], right = [1, 0], keys = {} } = controls;
            // Epsilon prevents floating-point subtraction from dropping the fifth tick.
            for (let ticks = 0; ticks < MAX_TICKS && accumulator + Number.EPSILON >= FIXED_STEP; ticks++) {
                if (!xrActive) {
                    physics.movePlayer(...forward, ...right, keys);
                    if (keys.Space) physics.jump();
                }
                physics.step();
                accumulator = Math.max(0, accumulator - FIXED_STEP);
            }
        },

        /**
         * Command. Disables the desktop capsule in XR; never moves an XR camera.
         * @param {boolean} active - Whether an immersive XR session is active.
         * @example scene.setXR(true) // undefined; player collision disabled
         */
        setXR(active) {
            xrActive = active;
            physics.playerBody.setEnabled(!active);
        },

        /**
         * Query. Reads legacy render data from the current physics world.
         * @returns {object} Legacy geometry and transforms.
         * @example scene.renderData().dominoes.length // 60
         */
        renderData() {
            const data = physics.getSceneData();
            return { ...data, lights: sceneLights(data) };
        },

        /**
         * Command. Spawns a CCD projectile with hand or desktop muzzle clearance.
         * @param {number[]} origin - World [x,y,z] position.
         * @param {number[]} direction - Unit world [x,y,z] direction.
         * @param {boolean} [fromHand=false] - Use 0.35m rather than 1.5m clearance.
         * @example scene.shoot([0,1.6,6], [0,0,-1], true) // undefined
         */
        shoot(origin, direction, fromHand = false) {
            const [x, y, z] = origin;
            physics.shoot({ x, y, z }, ...direction, fromHand ? 0.35 : 1.5);
        },

        /**
         * Query. Finds a walkable floor hit within 30m with room for the player capsule.
         * Reads Rapier's query state from the latest physics tick; excludes player and sensors.
         * @param {number[]} origin - World [x,y,z] ray origin.
         * @param {number[]} direction - Nonzero world [x,y,z] ray direction (normalized here).
         * @returns {number[]|null} Floor [x,y,z], or null for a miss, steep hit, or obstruction.
         * @example scene.teleportTarget([10,2,6], [0,-1,0]) // [10,0,6] after a physics tick
         */
        teleportTarget(origin, direction) {
            const length = Math.hypot(...direction);
            if (!Number.isFinite(length) || length === 0) return null;
            const [x, y, z] = origin;
            const ray = new Ray({ x, y, z }, {
                x: direction[0] / length, y: direction[1] / length, z: direction[2] / length,
            });
            const hit = physics.world.castRayAndGetNormal(
                ray, TELEPORT_RANGE, true, QueryFilterFlags.EXCLUDE_SENSORS,
                undefined, undefined, physics.playerBody,
            );
            if (!hit || hit.normal.y < MIN_UP_NORMAL) return null;
            const landing = ray.pointAt(hit.timeOfImpact);
            const shape = physics.playerCollider.shape;
            const obstruction = physics.world.intersectionWithShape(
                { x: landing.x, y: landing.y + shape.halfHeight + shape.radius + LANDING_CLEARANCE, z: landing.z },
                IDENTITY_ROTATION, shape, QueryFilterFlags.EXCLUDE_SENSORS,
                undefined, undefined, physics.playerBody,
            );
            return obstruction ? null : [landing.x, landing.y, landing.z];
        },

        /**
         * Command. Resets legacy physics and pending time while retaining XR/slow-motion mode.
         * @example scene.reset() // undefined; projectiles removed, 60 dominoes restored
         */
        reset,

        /**
         * Command. Frees the Rapier world; the adapter must not be used afterward.
         * @example scene.destroy() // undefined
         */
        destroy() {
            physics.world.free();
        },
    };
}
