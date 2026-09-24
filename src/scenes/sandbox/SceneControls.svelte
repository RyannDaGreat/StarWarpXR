<script>
    let { scene, onreset } = $props();
</script>

<label>
    <input type="checkbox" checked={scene.physics.slowMo} onchange={event => { scene.physics.slowMo = event.currentTarget.checked; }} />
    Slow motion
</label>
<button type="button" onclick={onreset}>Reset</button>
