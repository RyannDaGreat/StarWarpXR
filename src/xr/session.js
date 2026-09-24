import { mat4 } from 'gl-matrix';
import { createPresenter } from './presenter.js';
import { aimFromHand, panelHit, panelPose, poseRay, worldPose } from './math.js';

const PANEL_WIDTH = 1200, PANEL_HEIGHT = 256, PANEL_GAP = 6;

/**
 * Command. Start an immersive session with native pinch input and actual per-eye poses.
 * @param {object} options - Presentation canvas, WebGPU source canvas, floor spawn [x,y,z],
 * render(now,views,head), onshoot(origin,direction,fromHand), onteleport(origin,direction),
 * controls [{label,run}], onend() and onerror(error). Scene controls are supplied, never imported.
 * @returns {Promise<object>} {session,end}; end() releases immersion idempotently.
 * @example await startXR(options) // {session: XRSession, end: Function}; requires a user gesture
 */
export async function startXR({ canvas, source, spawn, render, onshoot, onteleport, controls, onend, onerror }) {
  const actions = [...controls, ...['Shoot', 'Teleport', 'Exit'].map(label => ({ label }))];
  if (!isSecureContext) throw new Error('VR needs trusted HTTPS (localhost is only valid on the same device).');
  if (!navigator.xr) throw new Error('WebXR is unavailable. Use Safari on Vision Pro with WebXR enabled.');
  // This request must remain inside the original button gesture, before other awaits.
  const session = await navigator.xr.requestSession('immersive-vr', {
    requiredFeatures: ['local-floor'],
  });
  let presenter, ended = false, ending;
  const events = new AbortController();

  /** Command. Release presentation and listeners once, including external session termination. */
  function cleanup() {
    if (ended) return;
    ended = true; events.abort(); presenter?.destroy(); onend();
  }

  /** Command. End immersion once; cleanup also runs when the platform rejects termination. */
  function end() {
    if (ended) return Promise.resolve();
    return ending ??= session.end().finally(cleanup);
  }
  session.addEventListener('end', cleanup, { once: true });
  try {
    presenter = createPresenter(canvas);
    await presenter.gl.makeXRCompatible();
    if (ended) throw new Error('XR session ended during startup');
    const layer = new XRWebGLLayer(session, presenter.gl, { alpha: false, depth: false, antialias: false });
    session.updateRenderState({ baseLayer: layer, depthNear: 0.1, depthFar: 10000 });
    const reference = await session.requestReferenceSpace('local-floor');
    if (ended) throw new Error('XR session ended during startup');
    reference.addEventListener('reset', () => {
      onerror(new Error('Headset tracking origin changed. Re-enter VR to recalibrate safely.'));
      end().catch(onerror);
    }, { signal: events.signal });
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      onerror(new Error('WebXR presentation context lost. Reload to restore rendering.'));
      end().catch(onerror);
    }, { signal: events.signal });
    const panelCanvas = document.createElement('canvas');
    panelCanvas.width = PANEL_WIDTH; panelCanvas.height = PANEL_HEIGHT;
    const context = panelCanvas.getContext('2d');
    let origin = [...spawn], calibrated = false, panel = null, localHead = null;
    let tool = 'Shoot', notice = 'Look at a target and pinch. Controls are below you.';

    /** Command. Paint accessible, high-contrast labels into the in-VR control atlas. */
    function paintPanel() {
      const cell = PANEL_WIDTH / actions.length;
      const rowHeight = PANEL_HEIGHT / 2;
      context.fillStyle = '#0e1729'; context.fillRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT);
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.font = 'bold 32px system-ui';
      for (const [index, { label: action }] of actions.entries()) {
        context.fillStyle = action === tool ? '#225d5d' : '#23304a';
        context.fillRect(index * cell + PANEL_GAP, PANEL_GAP, cell - 2 * PANEL_GAP, rowHeight - 2 * PANEL_GAP);
        context.fillStyle = '#ffffff'; context.fillText(action, (index + .5) * cell, rowHeight / 2);
      }
      context.font = '26px system-ui'; context.fillStyle = '#bce6e1';
      context.fillText(notice, PANEL_WIDTH / 2, rowHeight * 1.5, PANEL_WIDTH - 24);
    }

    /**
     * Command. Handle transient/controller selection without assuming stable source indices.
     * @param {XRInputSourceEvent} event - Native selectstart with event-time poses.
     * @example select(event) // undefined; activates a control or shoots/teleports
     */
    function select(event) {
      try {
        const target = event.frame.getPose(event.inputSource.targetRaySpace, reference);
        if (!target || !calibrated) return; // Tracking unavailable, not an input error.
        const ray = poseRay(worldPose(target.transform.matrix, origin));
        const hit = panel ? panelHit(ray, panel) : null;
        if (hit) {
          const cell = PANEL_WIDTH / actions.length;
          const x = hit[0] * PANEL_WIDTH, y = (1 - hit[1]) * PANEL_HEIGHT;
          if (y < PANEL_GAP || y > PANEL_HEIGHT / 2 - PANEL_GAP || x % cell < PANEL_GAP || x % cell > cell - PANEL_GAP) return;
          const control = actions[Math.floor(x / cell)];
          const action = control.label;
          if (action === 'Exit') { end().catch(onerror); return; }
          if (action === 'Shoot' || action === 'Teleport') {
            tool = action;
            notice = tool === 'Teleport' ? 'Look at a walkable surface and pinch to teleport.' : 'Look at a target and pinch to launch from your hand.';
          } else {
            control.run();
            notice = `${action} selected`;
          }
          return;
        }
        if (tool === 'Teleport') {
          const landing = onteleport(ray.origin, ray.direction);
          if (landing && localHead) {
            origin = [landing[0] - localHead[12], landing[1], landing[2] - localHead[14]];
            panel = panelPose(worldPose(localHead, origin));
            notice = 'Teleported. Physical walking remains 1:1.';
          } else notice = 'No clear walkable landing. Try another surface.';
          return;
        }
        const grip = event.inputSource.gripSpace && event.frame.getPose(event.inputSource.gripSpace, reference);
        const hand = grip ? poseRay(worldPose(grip.transform.matrix, origin)).origin : ray.origin;
        onshoot(hand, aimFromHand(ray, hand), Boolean(grip));
        notice = grip ? 'Ball launched from your hand' : 'Ball launched from selection ray (no grip pose)';
      } catch (error) { onerror(error); }
    }

    /**
     * Command. Render one XR frame; skip drawing when tracking is unavailable.
     * @param {number} now - Runtime timestamp, milliseconds.
     * @param {XRFrame} xrFrame - Current tracked viewer/eye poses.
     * @example frame(timestamp, xrFrame) // undefined; submits stereo and schedules next frame
     */
    function frame(now, xrFrame) {
      if (ended) return;
      try {
        const viewer = xrFrame.getViewerPose(reference);
        if (viewer) {
          if (!calibrated) {
            origin = [spawn[0] - viewer.transform.position.x, spawn[1], spawn[2] - viewer.transform.position.z];
            calibrated = true;
          }
          localHead = viewer.transform.matrix;
          const head = worldPose(localHead, origin);
          panel = panelPose(head);
          const views = Array.from(viewer.views, xrView => {
            const pose = worldPose(xrView.transform.matrix, origin);
            const view = mat4.invert(mat4.create(), pose);
            const viewProj = mat4.multiply(mat4.create(), xrView.projectionMatrix, view);
            return { xrView, eye: xrView.eye, viewProj, invViewProj: mat4.invert(mat4.create(), viewProj), ...poseRay(pose) };
          });
          if (views.length !== 2) throw new Error('This stereo bridge requires a two-view immersive headset.');
          render(now, views, poseRay(head));
          paintPanel();
          presenter.upload(source, panelCanvas);
          presenter.present(layer, views, panel);
        }
        session.requestAnimationFrame(frame);
      } catch (error) {
        onerror(error);
        end().catch(onerror);
      }
    }
    session.addEventListener('selectstart', select, { signal: events.signal });
    session.requestAnimationFrame(frame);
    return { session, end };
  } catch (error) {
    try { await end(); }
    catch (cleanupError) { onerror(cleanupError); }
    throw error;
  }
}
