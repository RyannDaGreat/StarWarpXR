# Noise Warp VR

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

WebGPU requires a secure context. Desktop localhost works. Vision Pro requires **visionOS 26+** (Safari WebGPU), WebXR support, and a trusted HTTPS URL. `npm run dev:https` provides local TLS, but its self-signed certificate must be trusted on the headset; merely dismissing a certificate warning is not a reliable secure-context setup. A trusted HTTPS tunnel to port 5173 is usually easier. If using a tunnel, start Vite with its exact hostname:

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

**Actual Vision Pro testing is still required.** The renderer uses WebGPU, while immersive presentation uses a WebGL `XRWebGLLayer`. A per-frame canvas copy connects them; this is not zero-copy interop, and Safari performance/compatibility has not been verified on hardware. The default render resolution is 768×768 per eye. Two explicit star states consume about 208 MiB, plus scene and rendering resources. The retained GPU atomics mean explicit seeds do not imply bitwise reproducibility.

Tracking loss skips frames. A tracking-origin reset ends immersion with an explanatory error rather than unexpectedly shifting/rotating the world; re-enter to recalibrate. The panel is drawn as an overlay without scene-depth occlusion. No controller/hand meshes, grabbing, multiplayer, or passthrough AR are included.

## GitHub deployment

The repository includes `.github/workflows/pages.yml`. Add the intended GitHub remote, enable **Settings → Pages → Source: GitHub Actions**, and push `main`. The workflow runs unit tests, builds Vite and deploys `dist/` to an HTTPS Pages URL. Relative asset URLs support a project subpath. Headset development uses the Vite URL; GitHub Pages serves production builds without hot reload.

## Source provenance

Extracted from `ryanndagreat/infinite_resolution_integral_noise_warping_code`, commit `db80da363f4103c6470d05cbf850fdd55dcc3ed8`, directory `web_demo_v3`. Original license is retained in `LICENSE`. Physics layouts, geometry, scene shaders, wood textures, and point-process star warping originate there. Other display modes are intentionally excluded.

Apple input references: [natural input](https://webkit.org/blog/15162/introducing-natural-input-for-webxr-in-apple-vision-pro/), [WebXR on visionOS](https://developer.apple.com/videos/play/wwdc2024/10066/), [Safari 26 WebGPU](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/).
