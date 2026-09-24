# Noise Warp VR

A focused extraction of [web_demo_v3](https://ryanndagreat.github.io/infinite_resolution_integral_noise_warping_code/web_demo_v3/): **2 · Scene** and **7 · Stars**, with WebXR for Apple Vision Pro.

## Architecture contract

- `src/stars/`: standalone coordinate evolution. Explicit previous state, motion fields, and random seed → next star coordinates and metadata. No scene, physics, UI, WebXR, camera, or drawing dependencies. GPU allocation/dispatch is a command, not a mathematically pure function. Rendering those coordinates is a separate concern.
- `src/render/`: scene rasterization, motion fields, and drawing star coordinates. The bridge into the algorithm passes only numerical fields/state.
- `src/scenes/sandbox/`: original world generation, Rapier physics, and scene-specific controls. Replace the scene without changing the star algorithm.
- `src/xr/`: headset poses, transient pinch events, and immersive presentation. No star-algorithm knowledge.
- Svelte supplies desktop UI; Vite supplies development hot reload.

## Development

```sh
npm ci
npm run dev
npm run build
npm test
```

WebGPU requires a secure context. Desktop localhost works. Vision Pro requires **visionOS 26+** (Safari WebGPU), WebXR support, and a trusted HTTPS URL. `npm run dev:https` provides local TLS, but its self-signed certificate must be trusted on the headset; a trusted HTTPS tunnel is usually easier. Hot reload reaches the headset through the same URL. Browser reload may end immersion; re-enter VR after a full reload.

## Source provenance

Extracted from `ryanndagreat/infinite_resolution_integral_noise_warping_code`, commit `db80da363f4103c6470d05cbf850fdd55dcc3ed8`, directory `web_demo_v3`. Original license is retained in `LICENSE`. Physics layouts, geometry, scene shaders, wood textures, and point-process star warping originate there. Other display modes are intentionally excluded.

Apple input references: [natural input](https://webkit.org/blog/15162/introducing-natural-input-for-webxr-in-apple-vision-pro/), [WebXR on visionOS](https://developer.apple.com/videos/play/wwdc2024/10066/), [Safari 26 WebGPU](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/).
