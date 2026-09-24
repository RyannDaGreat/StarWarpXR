import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { compile } from 'svelte/compiler';
import { ColliderDesc, RigidBodyDesc } from '@dimforge/rapier3d-compat';
import { createSandbox } from '../src/scenes/sandbox/index.js';

/** Command. Exercises the real Rapier adapter and frees its world after verification. */
async function verifySandbox() {
    const scene = await createSandbox();
    const { physics } = scene;
    try {
        assert.deepEqual(scene.spawn, [0, 0, 6]);
        assert.equal(scene.renderData().dominoes.length, 60);
        const step = physics.step.bind(physics);
        let ticks = 0;
        /** Command. Counts scheduled ticks without advancing the large world. */
        physics.step = function countTick() { ticks++; };
        for (const rate of [60, 72, 90, 120, 144]) {
            scene.reset();
            ticks = 0;
            for (let frame = 0; frame < rate; frame++) scene.update(1 / rate);
            assert.equal(ticks, 60, `${rate}Hz must schedule 60 ticks`);
        }
        ticks = 0;
        scene.update(10);
        assert.equal(ticks, 5);
        scene.update(0);
        assert.equal(ticks, 5, 'stall backlog must be discarded');
        assert.throws(() => scene.update(NaN), RangeError);
        scene.update(1 / 60, {forward:[0,-1], right:[1,0], keys:{KeyW:true}});
        assert.equal(physics.playerBody.linvel().z, -5);
        scene.setXR(true);
        assert.equal(physics.playerBody.isEnabled(), false);
        const before = physics.playerBody.translation();
        scene.update(1 / 60, {keys:{KeyD:true, Space:true}});
        assert.deepEqual(physics.playerBody.translation(), before);
        assert.equal(physics.playerBody.linvel().z, -5, 'XR ignores desktop input');
        scene.reset();
        assert.equal(physics.playerBody.isEnabled(), false);
        scene.setXR(false);
        assert.equal(physics.playerBody.isEnabled(), true);
        physics.step = step;
        scene.update(1 / 60);
        assert.ok(Math.abs(scene.teleportTarget([10,2,6], [0,-2,0])[1]) < 1e-5);
        assert.equal(scene.teleportTarget([10,31,6], [0,-1,0]), null);
        assert.equal(scene.teleportTarget([10,2,6], [0,0,0]), null);
        assert.equal(scene.teleportTarget([10,2,6], [0,1,0]), null);
        assert.ok(scene.teleportTarget([0,2,6], [0,-1,0]), 'ray excludes player');

        const wall = physics.world.createRigidBody(RigidBodyDesc.fixed().setTranslation(10,1,10));
        physics.world.createCollider(ColliderDesc.cuboid(1,1,0.1), wall);
        const ceiling = physics.world.createRigidBody(RigidBodyDesc.fixed().setTranslation(10,1.3,6));
        physics.world.createCollider(ColliderDesc.cuboid(1,0.1,1), ceiling);
        scene.update(1 / 60);
        assert.equal(scene.teleportTarget([10,1,8], [0,0,1]), null, 'wall is not walkable');
        assert.equal(scene.teleportTarget([10,1,6], [0,-1,0]), null, 'low ceiling blocks capsule');
        physics.world.removeRigidBody(wall);
        physics.world.removeRigidBody(ceiling);

        physics.playerBody.setTranslation({x:10,y:10,z:6}, true);
        scene.update(1 / 60);
        const airborneVelocity = physics.playerBody.linvel().y;
        physics.jump();
        assert.equal(physics.playerBody.linvel().y, airborneVelocity, 'no self-ray midair jump');
        physics.playerBody.setTranslation({x:10,y:0.81,z:6}, true);
        physics.playerBody.setLinvel({x:0,y:0,z:0}, true);
        scene.update(1 / 60);
        physics.jump();
        assert.ok(physics.playerBody.linvel().y > 0, 'grounded jump works');

        scene.shoot([10,2,6], [0,0,-1]);
        scene.shoot([10,2,6], [0,0,-1], true);
        assert.equal(physics.spheres[0].body.translation().z, 4.5);
        assert.ok(Math.abs(physics.spheres[1].body.translation().z - 5.65) < 1e-5);
        assert.equal(physics.spheres[1].body.isCcdEnabled(), true);
        physics.slowMo = true;
        scene.update(1 / 60);
        assert.ok(Math.abs(physics.world.timestep - 1 / 240) < 1e-8);
        scene.reset();
        assert.equal(physics.spheres.length, 0);
        assert.equal(physics.slowMo, true);
        assert.equal(scene.renderData().dominoes.length, 60);
    } finally {
        scene.destroy();
    }
}

test('sandbox adapter: timing, XR, teleport, projectiles, jump and reset', verifySandbox);

/** Command. Compiles scene-only controls and verifies their two native control bindings. */
function verifyControls() {
    const source = readFileSync(new URL('../src/scenes/sandbox/SceneControls.svelte', import.meta.url), 'utf8');
    const result = compile(source, {filename:'SceneControls.svelte'});
    assert.deepEqual(result.warnings, []);
    assert.match(source, /scene\.physics\.slowMo = event\.currentTarget\.checked/);
    assert.match(source, /onclick=\{onreset\}/);
}

test('sandbox controls compile without warnings', verifyControls);
