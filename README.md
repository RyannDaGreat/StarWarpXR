# StarWarpXR

[Open the demo](https://ryanndagreat.github.io/StarWarpXR/) · [GitHub repository](https://github.com/RyannDaGreat/StarWarpXR)

A focused extraction of [web_demo_v3](https://ryanndagreat.github.io/infinite_resolution_integral_noise_warping_code/web_demo_v3/): **2 · Scene** and **7 · Stars**, with WebXR for Apple Vision Pro.

## Architecture contract

- `src/stars/`: standalone coordinate evolution. Explicit previous state, motion fields, and random seed → next star coordinates and metadata. No scene, physics, UI, WebXR, camera, or drawing dependencies. GPU allocation/dispatch is a command, not a mathematically pure function. Rendering those coordinates is a separate concern.
- `src/render/`: generic instanced rasterization, motion fields, and a separate `StarDraw` coordinate rasterizer. The bridge into the algorithm passes only numerical fields/state plus explicit count and seed.
- `src/scenes/sandbox/`: original world generation, meshes, material/sky shaders, instance packing, Rapier physics, and scene-specific controls. The scene supplies assets and draw batches; the renderer does not import the sandbox.
- `src/xr/`: headset poses, transient pinch events, and immersive presentation. No star-algorithm knowledge.
- Svelte supplies desktop UI; Vite supplies development hot reload.

## Development

```sh
npm ci
npm run dev
npm run build
npm test
npm run test:browser
```

WebGPU requires a secure context. Desktop localhost works. Use **visionOS 26.2+** for native WebGPU XR, WebXR support, and a trusted HTTPS URL. `npm run dev:https` provides local TLS, but its self-signed certificate must be trusted on the headset; merely dismissing a certificate warning is not a reliable secure-context setup. A trusted HTTPS tunnel to port 5173 is usually easier. If using a tunnel, start Vite with its exact hostname:

```sh
__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS=your-tunnel.example npm run dev
```

Open the tunnel's HTTPS URL in Safari on Vision Pro, then press **Enter VR**. Your reverse proxy must forward WebSockets for Vite hot reload. Full reload can end immersion; re-enter afterward. Serve production builds over HTTPS, not `file://`.

## Controls

- **Desktop:** click the scene to capture the mouse; WASD moves, Shift runs, Space jumps. Further clicks shoot. Esc releases the mouse.
- **Vision Pro:** walk physically, look at a target and pinch to launch a ball from the tracked hand. Uses native `transient-pointer` selection, not a controller-button assumption. When no grip pose is supplied, the in-VR notice explicitly identifies selection-ray launching.
- **In VR:** look down at the spatial control panel. Scene / Stars switches rendering; Reset is supplied by the sandbox. Select Shoot or Teleport, then look and pinch at a target. Teleport accepts clear walkable surfaces within 30 metres. Exit ends immersion.
- Physical movement is **1:1 and never clamped to virtual walls**. Teleport checks landing clearance, but real walking can pass through virtual objects. Keep your real space clear; this is not a physical-safety boundary system.

## Verification and limitations

Unit tests cover coordinate ownership/seed contracts, geometry, instance history, physics, projection conventions and XR math. `npm run test:browser` starts/stops Vite and runs Puppeteer checks (tracked harnesses in `.scratchpad/verification/`): both display modes, real WGSL zero-motion preservation and stereo masks, WebGPU→WebGL eye/orientation/color transfer, plus a **simulated** XR session with hand launch, panel selection, repeated teleport and teardown. Screenshots land in `.scratchpad/`.

**Native headset-compositor validation is still required.** When `XRGPUBinding` is available, the app requests an XR-compatible GPU adapter and the `webgpu` session feature, samples the renderer's SBS texture directly, and draws each eye into its native projection-layer texture. This avoids WebGPU→WebGL canvas transfer entirely. Native XR projections use [0,1] clip depth; a pure conversion supplies the scene/motion renderer's [-1,1] matrices without changing the native panel projection. The presenter respects each subimage's array layer and viewport, with no scissor operation.

Browsers without native WebGPU XR use a **legacy WebGL compatibility path**, with a fresh context per session and operation-specific error checks on every frame. That path has reported black screens and error 1282 on Vision Pro; it is **not considered headset-validated**. The precise device-side cause has not been reproduced locally. Errors offer a copyable diagnostic report containing build ID, chosen backend, browser, capability flags, and stack; it stays local unless you share it.

### Visual WebKit regression

On macOS, install Playwright's WebKit inside this dump and run:

```sh
PLAYWRIGHT_BROWSERS_PATH=.scratchpad/playwright npx playwright install webkit
npm run test:webkit
```

This runs the **actual app, WGSL shaders and GPU presentation** in Apple WebKit, but supplies a **simulated XR runtime/compositor**. It captures both eye layers to `.scratchpad/webkit-xr-*-scene.png` / `*-stars.png`, checks asymmetric red/yellow/green/blue markers for eye assignment and orientation, tests Scene → exit → Stars → exit → Scene, and exercises native pinch events with pointer-lock APIs absent. WebGL context creation is prohibited in this test to prove the native path does not cross that bridge. The saved images were visually reviewed: scene geometry, stars, controls, and launched balls are visible in both eyes.

**This is not a Vision Pro simulator or a claim of headset success.** It does not reproduce visionOS compositor internals, real tracking/permissions, or headset performance. Desktop Safari automation additionally requires Allow Remote Automation; the available native Safari driver rejected startup because that setting is disabled. The Playwright WebKit test runs without changing it. A desktop WebKit canvas-copy probe did not reproduce the reported 1282, so a successful desktop bridge test is not evidence that the legacy headset path works.

The default render resolution is 768×768 per eye. Two explicit star states consume about 208 MiB, plus scene and rendering resources. The retained GPU atomics mean explicit seeds do not imply bitwise reproducibility.

Tracking loss skips frames. A tracking-origin reset ends immersion with an explanatory error rather than unexpectedly shifting/rotating the world; re-enter to recalibrate. The panel is drawn as an overlay without scene-depth occlusion. No controller/hand meshes, grabbing, multiplayer, or passthrough AR are included.

## GitHub deployment

The public repository is `RyannDaGreat/StarWarpXR`, with **Settings → Pages → Source: GitHub Actions** and HTTPS enforced. Every push to `main` runs `.github/workflows/pages.yml`: unit tests, Vite build, then deployment of `dist/` to https://ryanndagreat.github.io/StarWarpXR/. Relative asset URLs support the project subpath. Headset development uses the Vite URL; GitHub Pages serves production builds without hot reload.

## Source provenance

Extracted from `ryanndagreat/infinite_resolution_integral_noise_warping_code`, commit `db80da363f4103c6470d05cbf850fdd55dcc3ed8`, directory `web_demo_v3`. Original license is retained in `LICENSE`. Physics layouts, geometry, scene shaders, wood textures, and point-process star warping originate there. Other display modes are intentionally excluded.

Apple input references: [natural input](https://webkit.org/blog/15162/introducing-natural-input-for-webxr-in-apple-vision-pro/), [WebXR on visionOS](https://developer.apple.com/videos/play/wwdc2024/10066/), [Safari 26 WebGPU](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/), [Safari 26.2 native WebGPU XR](https://webkit.org/blog/17640/webkit-features-for-safari-26-2/), [WebXR WebGPU binding](https://immersive-web.github.io/webxr-webgpu-binding/), [WebKit XR scissor fix](https://github.com/WebKit/WebKit/commit/eda478cddb5d40012b333ffcbf0b71eb200c49b0).
