import { mat4 } from 'gl-matrix';
import { WebGPURenderer } from './render/renderer.js';
import { SceneManager } from './scenes/sandbox/instances.js';
import { createSandboxAssets } from './scenes/sandbox/assets.js';
import { createSandbox } from './scenes/sandbox/index.js';
import { startXR } from './xr/session.js';

const EYE_RESOLUTION = 768;
const LOOK_SENSITIVITY = 0.002;
const PITCH_LIMIT = Math.PI / 2 - 0.01;
const FOV = Math.PI / 3;
const CLIP_NEAR = 0.1, CLIP_FAR = 10000;

/**
 * Command. Assemble independent scene, renderer, and XR adapters; install desktop controls.
 * @param {object} options - WebGPU/WebGL canvases and status/error callbacks.
 * @returns {Promise<object>} Application control commands and active scene.
 */
export async function createApp({ canvas, xrCanvas, onerror, onstatus }) {
  if (!navigator.gpu) throw new Error('WebGPU is required. On Vision Pro, use visionOS 26 or later over trusted HTTPS.');
  const scene = await createSandbox();
  const renderer = new WebGPURenderer(canvas, EYE_RESOLUTION, EYE_RESOLUTION, createSandboxAssets());
  try { await renderer.init(); }
  catch (error) { scene.destroy(); renderer.destroy(); throw error; }
  const instances = new SceneManager();
  let mode = 1, yaw = 0, pitch = 0, lastTime = null, elapsed = 0, seed = 0;
  let previous = new Map(), immersive = false, xr = null, entering = null, alive = true, raf;
  const keys = {};
  const listeners = new AbortController();

  /** Command. Discard camera/object/coordinate history after a discontinuity. */
  function resetHistory() {
    previous.clear(); instances.prevTransforms.clear();
    renderer.resetHistory();
    lastTime = null;
  }

  /**
   * Command. Change display mode without changing physics or algorithm code.
   * @param {number} value - Internal mode 1 (Scene) or 6 (Stars).
   * @example setMode(6) // undefined; switch to Stars and reset coordinate history
   */
  function setMode(value) { mode = value; resetHistory(); onstatus({ mode }); }

  /** Command. Reset this scene and its rendering history. */
  function reset() { scene.reset(); resetHistory(); }

  /**
   * Query. Read desktop look direction from mutable yaw/pitch.
   * @returns {number[]} Unit world direction [dx,dy,dz].
   * @example look() // [0,0,-1] when yaw and pitch are zero
   */
  function look() {
    return [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
  }

  /**
   * Command. Advance physics and draw the tracked views with one shared simulation tick.
   * @param {number} now - Frame timestamp in milliseconds.
   * @param {object[]} views - GL matrices (16,), world xyz origins and directions per eye.
   * @param {object} head - World {origin:[x,y,z],direction:[dx,dy,dz]} for lighting.
   * @example render(now, xrViews, headRay) // undefined; stereo frame submitted
   */
  function render(now, views, head) {
    const dt = lastTime === null ? 0 : (now - lastTime) / 1000;
    lastTime = now; elapsed += dt;
    scene.update(dt, { keys, forward: [Math.sin(yaw), -Math.cos(yaw)], right: [Math.cos(yaw), Math.sin(yaw)] });
    const sceneData = scene.renderData();
    const counts = instances.buildInstances(sceneData);
    const left = views.find(view => view.eye === 'left') ?? views[0];
    const right = views.find(view => view.eye === 'right');
    const stereo = right ? {
      mode: 2, viewProjL: left.viewProj, viewProjR: right.viewProj,
      prevViewProjL: previous.get('left') ?? left.viewProj,
      prevViewProjR: previous.get('right') ?? right.viewProj,
      invViewProjL: left.invViewProj, invViewProjR: right.invViewProj,
      eyePosL: left.origin, eyePosR: right.origin, eyeDirL: left.direction, eyeDirR: right.direction,
    } : null;
    renderer.frame({ ...counts, viewProj: left.viewProj, invViewProj: left.invViewProj,
      prevViewProj: previous.get(left.eye) ?? left.viewProj, stereo,
      instanceData: instances.getActiveData(), displayMode: mode, frameSeed: seed++,
      elapsedSecs: elapsed, eyePos: head.origin, eyeDir: head.direction,
      lights: sceneData.lights,
    });
    for (const view of views) previous.set(view.eye, mat4.clone(view.viewProj));
  }

  /**
   * Command. Draw/schedule desktop frames; WebXR owns drawing while immersive.
   * @param {number} now - Window animation timestamp in milliseconds.
   * @example desktop(now) // undefined; renders unless immersed, then schedules next frame
   */
  function desktop(now) {
    if (!alive) return;
    try {
      if (!immersive) {
        const position = scene.physics.getPlayerEyePos();
        const origin = [position.x, position.y, position.z], direction = look();
        const target = origin.map((value, index) => value + direction[index]);
        const view = mat4.lookAt(mat4.create(), origin, target, [0, 1, 0]);
        const projection = mat4.perspective(mat4.create(), FOV, canvas.clientWidth / canvas.clientHeight, CLIP_NEAR, CLIP_FAR);
        const viewProj = mat4.multiply(mat4.create(), projection, view);
        render(now, [{ eye: 'mono', viewProj, invViewProj: mat4.invert(mat4.create(), viewProj), origin, direction }], { origin, direction });
      }
      raf = requestAnimationFrame(desktop);
    } catch (error) { onerror(error); }
  }

  /** Command. Begin XR inside a user gesture; report unavailable capabilities to the UI. */
  async function beginVR() {
    if (immersive || !alive) return;
    immersive = true;
    try {
      document.exitPointerLock();
      scene.setXR(true); resetHistory(); onstatus({ immersive: true });
      xr = await startXR({ canvas: xrCanvas, source: canvas, spawn: scene.spawn, render,
        onshoot: scene.shoot,
        onteleport(origin, direction) {
          const destination = scene.teleportTarget(origin, direction);
          if (destination) resetHistory();
          return destination;
        },
        controls: [
          { label: 'Scene', run: () => setMode(1) },
          { label: 'Stars', run: () => setMode(6) },
          ...scene.xrControls.map(control => ({ label: control.label, run() { control.run(); resetHistory(); } })),
        ],
        onend() {
          xr = null; immersive = false; scene.setXR(false);
          if (alive && !renderer.gpuError) resetHistory();
          onstatus({ immersive: false });
        }, onerror,
      });
    } catch (error) {
      immersive = false; scene.setXR(false); lastTime = null;
      onstatus({ immersive: false }); onerror(error);
    }
  }

  /** Command. Track pending session setup so HMR teardown waits for it before freeing resources. */
  function enterVR() {
    entering ??= beginVR().finally(() => { entering = null; });
    return entering;
  }

  /**
   * Command. Track movement keys, ignoring keydown inside forms but always releasing keys.
   * @param {KeyboardEvent} event - Window keydown/keyup event.
   * @example key(new KeyboardEvent('keyup', {code:'KeyW'})) // undefined; W is released
   */
  function key(event) {
    if (event.type === 'keydown' && event.target.matches('input, select, button, textarea')) return;
    keys[event.code] = event.type === 'keydown';
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.code)) event.preventDefault();
  }
  window.addEventListener('keydown', key, { signal: listeners.signal });
  window.addEventListener('keyup', key, { signal: listeners.signal });
  window.addEventListener('blur', () => { for (const key of Object.keys(keys)) delete keys[key]; }, { signal: listeners.signal });
  document.addEventListener('mousemove', event => {
    if (document.pointerLockElement !== canvas || immersive) return;
    yaw += event.movementX * LOOK_SENSITIVITY;
    pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pitch - event.movementY * LOOK_SENSITIVITY));
  }, { signal: listeners.signal });
  canvas.addEventListener('click', () => {
    if (immersive) return;
    if (document.pointerLockElement !== canvas) { canvas.requestPointerLock().catch(onerror); return; }
    const position = scene.physics.getPlayerEyePos();
    scene.shoot([position.x, position.y, position.z], look());
  }, { signal: listeners.signal });
  raf = requestAnimationFrame(desktop);
  return {
    scene, renderer, setMode, reset, enterVR,
    /** Command. Stop animation/listeners, end immersion, then release GPU and physics resources. */
    async destroy() {
      alive = false; cancelAnimationFrame(raf); listeners.abort();
      try {
        await entering;
        if (xr) await xr.end();
      } finally {
        scene.destroy(); renderer.destroy();
      }
    },
  };
}
