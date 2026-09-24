import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';
const browser = await puppeteer.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  page.on('console', message => console.log('BROWSER', message.type(), message.text()));
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle0' });
  const result = await page.evaluate(async () => {
    const { startXR } = await import('/src/xr/session.js');
    const { mat4 } = await import('/node_modules/.vite/deps/gl-matrix.js');
    const projection = mat4.perspective(mat4.create(), Math.PI / 3, 1, .1, 1000);
    const head = mat4.fromTranslation(mat4.create(), [0, 1.6, 0]);
    const reference = new EventTarget();
    const frame = {
      getViewerPose() {
        return { transform: { matrix: head, position: { x: 0, y: 1.6, z: 0 } }, views: ['left','right'].map(eye => ({
          eye, projectionMatrix: projection,
          transform: { matrix: mat4.fromTranslation(mat4.create(), [eye === 'left' ? -.032 : .032, 1.6, 0]) },
        })) };
      },
      getPose(space) { return { transform: { matrix: space.matrix } }; },
    };
    class Session extends EventTarget {
      updateRenderState() {}
      async requestReferenceSpace() { return reference; }
      requestAnimationFrame(callback) { window.requestAnimationFrame(now => { if (!this.ended) callback(now, frame); }); }
      async end() { this.ended = true; this.dispatchEvent(new Event('end')); }
    }
    const session = new Session();
    Object.defineProperty(navigator, 'xr', { configurable: true, value: { requestSession: async () => session } });
    WebGL2RenderingContext.prototype.makeXRCompatible = async () => {};
    globalThis.XRWebGLLayer = class {
      constructor() { this.framebuffer = null; }
      getViewport(view) { return { x: view.eye === 'left' ? 0 : 64, y: 0, width: 64, height: 64 }; }
    };
    const source = document.createElement('canvas'); source.width = 128; source.height = 64;
    const sourceContext = source.getContext('2d');
    sourceContext.fillStyle = 'red'; sourceContext.fillRect(0,0,64,64);
    sourceContext.fillStyle = 'green'; sourceContext.fillRect(64,0,64,64);
    const canvas = document.createElement('canvas'); canvas.width = 128; canvas.height = 64;
    let frames = 0, shots = [], actions = 0, ended = 0, errors = [], latestEyes;
    let firstFrame;
    const rendered = new Promise(resolve => { firstFrame = resolve; });
    const options = { canvas, source, spawn: [0,0,6],
      controls: [{ label: 'Custom', run() { actions++; } }],
      render(now, views) { frames++; latestEyes = views.map(view => view.origin); firstFrame(latestEyes); },
      onshoot(...args) { shots.push(args); }, onteleport() { return [10,0,8]; },
      onend() { ended++; }, onerror(error) { errors.push(error.message); },
    };
    const controller = await startXR(options);
    const eyeOrigins = await rendered;
    const event = new Event('selectstart');
    event.frame = frame;
    event.inputSource = {
      targetRaySpace: { matrix: head },
      gripSpace: { matrix: mat4.fromTranslation(mat4.create(), [.3, 1.1, -.2]) },
      targetRayMode: 'transient-pointer',
    };
    session.dispatchEvent(event);
    const panelEvent = new Event('selectstart'); panelEvent.frame = frame;
    // First of four cells: x=-.525; button center y=1.125,z=-1.25 in reference space.
    panelEvent.inputSource = { targetRaySpace: { matrix: mat4.targetTo(mat4.create(), [0,1.6,0], [-.525,1.125,-1.25], [0,1,0]) } };
    session.dispatchEvent(panelEvent);
    // The status row must consume the pinch without activating a hidden button or shooting.
    const statusEvent = new Event('selectstart'); statusEvent.frame = frame;
    statusEvent.inputSource = { targetRaySpace: { matrix: mat4.targetTo(mat4.create(), [0,1.6,0], [-.525,.975,-1.25], [0,1,0]) } };
    session.dispatchEvent(statusEvent);
    const toolEvent = new Event('selectstart'); toolEvent.frame = frame;
    toolEvent.inputSource = { targetRaySpace: { matrix: mat4.targetTo(mat4.create(), [0,1.6,0], [.175,1.125,-1.25], [0,1,0]) } };
    session.dispatchEvent(toolEvent);
    session.dispatchEvent(event);
    session.dispatchEvent(event);
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const teleportedEyes = latestEyes;
    reference.dispatchEvent(new Event('reset'));
    await controller.end();
    await controller.end();
    const earlySession = new Session();
    navigator.xr.requestSession = async () => earlySession;
    let resumeCompatibility, compatibilityEntered, earlyEnded = 0, startupError;
    const compatibilityStarted = new Promise(resolve => { compatibilityEntered = resolve; });
    WebGL2RenderingContext.prototype.makeXRCompatible = () => new Promise(resolve => {
      resumeCompatibility = resolve; compatibilityEntered();
    });
    const starting = startXR({ ...options, canvas: document.createElement('canvas'), onend() { earlyEnded++; } });
    await compatibilityStarted;
    await earlySession.end();
    resumeCompatibility();
    try { await starting; }
    catch (error) { startupError = error.message; console.log('Expected startup interruption:', startupError); }
    return { frames, shots, actions, ended, errors, eyeOrigins, teleportedEyes, earlyEnded, startupError };
  });
  assert.deepEqual(result.errors, ['Headset tracking origin changed. Re-enter VR to recalibrate safely.']);
  assert.equal(result.earlyEnded, 1);
  assert.equal(result.startupError, 'XR session ended during startup');
  assert.ok(result.frames >= 1);
  assert.equal(result.shots.length, 1);
  assert.equal(result.shots[0][2], true);
  assert.ok(Math.abs(result.shots[0][0][0] - .3) < 1e-6);
  assert.ok(Math.abs(result.shots[0][0][2] - 5.8) < 1e-6);
  assert.equal(result.actions, 1);
  assert.equal(result.ended, 1);
  assert.ok(result.eyeOrigins[0][0] < result.eyeOrigins[1][0]);
  assert.ok(Math.abs(result.teleportedEyes[0][0] - (10 - .032)) < 1e-5);
  assert.equal(result.teleportedEyes[0][2], 8);
  console.log('PASS: simulated XR lifecycle, stereo poses, transient pinch hand launch, spatial custom control, teardown', result);
} finally { await browser.close(); }
