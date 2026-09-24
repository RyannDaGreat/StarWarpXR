<script>
  import { onMount } from 'svelte';
  import { createApp } from './app.js';
  import SceneControls from './scenes/sandbox/SceneControls.svelte';

  let canvas, xrCanvas;
  let app = $state(null), error = $state(''), ready = $state(false);
  let mode = $state(1), immersive = $state(false), supported = $state(false);
  let stars = $state(10000), antialias = $state(true), cull = $state(false);

  /** Command. Report runtime failures visibly and in the console. */
  function report(problem) { console.error(problem); error = problem.message ?? String(problem); }

  /** Command. Receive renderer/XR status without coupling those adapters to Svelte. */
  function status(state) {
    if ('mode' in state) mode = state.mode;
    if ('immersive' in state) immersive = state.immersive;
  }

  onMount(() => {
    let disposed = false;
    createApp({ canvas, xrCanvas, onerror: report, onstatus: status }).then(value => {
      if (disposed) { value.destroy().catch(report); return; }
      app = value; ready = true;
    }).catch(report);
    if (navigator.xr) navigator.xr.isSessionSupported('immersive-vr').then(value => { supported = value; }).catch(report);
    return () => { disposed = true; app?.destroy().catch(report); };
  });
</script>

<svelte:head>
  <title>Noise Warp VR · Scene / Stars</title>
  <meta name="description" content="Integral noise-warping stars and the original physics sandbox, with immersive WebXR." />
</svelte:head>

<main>
  <canvas bind:this={canvas} aria-label="Interactive physics sandbox" class:immersive></canvas>
  <canvas bind:this={xrCanvas} class="xr-surface" aria-hidden="true"></canvas>
  <header>
    <div><strong>NOISE WARP <span>VR</span></strong><small>Scene / Stars · WebXR</small></div>
    <button class="vr" disabled={!ready || !supported || immersive} onclick={() => app.enterVR()}>
      {immersive ? 'In headset' : supported ? 'Enter VR' : 'VR headset required'}
    </button>
  </header>
  <aside aria-label="Controls">
    <fieldset disabled={!ready}>
      <legend>Rendering</legend>
      <div class="modes">
        <button class:active={mode === 1} aria-pressed={mode === 1} onclick={() => app.setMode(1)}>2 · Scene</button>
        <button class:active={mode === 6} aria-pressed={mode === 6} onclick={() => app.setMode(6)}>7 · Stars</button>
      </div>
      {#if mode === 6}
        <label>Star coordinates / eye <output>{stars.toLocaleString()}</output>
          <input aria-label="Star count" type="range" min="1000" max="100000" step="1000" bind:value={stars}
            oninput={event => { app.renderer.numStars = event.currentTarget.valueAsNumber; }} />
        </label>
        <label><input type="checkbox" bind:checked={antialias} onchange={event => { app.renderer.starDraw.starAAEnabled = event.currentTarget.checked; }} /> Tent antialiasing</label>
        <label><input type="checkbox" bind:checked={cull} onchange={event => { app.renderer.starDraw.cullOrphansEnabled = event.currentTarget.checked; }} /> Binocular pairs only</label>
      {/if}
    </fieldset>
    {#if app}
      <fieldset><legend>Sandbox</legend><SceneControls scene={app.scene} onreset={() => app.reset()} /></fieldset>
    {/if}
    <details>
      <summary>Controls & headset setup</summary>
      <p><b>Desktop:</b> click the scene to look around. WASD moves, Shift runs, Space jumps. Click again to shoot. Esc releases the mouse.</p>
      <p><b>Vision Pro:</b> look and pinch to shoot from your hand. Look down for Scene / Stars, Shoot / Teleport, Reset and Exit controls. Physical walking is 1:1; keep your real space clear.</p>
      <p>Requires visionOS 26+, WebXR, and trusted HTTPS. Use the same Vite URL on the headset for hot reload. Full reload exits VR.</p>
    </details>
  </aside>
  {#if !ready && !error}<div class="notice" role="status">Loading sandbox & compiling shaders…</div>{/if}
  {#if immersive}<div class="notice">Look down in VR for controls. Pinch to select.</div>{/if}
  {#if error}<div class="error" role="alert"><b>Cannot continue</b><p>{error}</p><button onclick={() => location.reload()}>Reload</button></div>{/if}
  <footer>Original sandbox · standalone star coordinates · <a href="https://ryanndagreat.github.io/infinite_resolution_integral_noise_warping_code/web_demo_v3/">source demo</a></footer>
</main>
