import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';

const browser = await puppeteer.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  const uncaught = [];
  page.on('pageerror', error => { console.error(error); uncaught.push(error.message); });
  page.on('console', message => console.log('POINTER TEST', message.type(), message.text()));
  await page.evaluateOnNewDocument(() => {
    Object.defineProperty(document, 'exitPointerLock', { value: undefined, configurable: true });
    Object.defineProperty(Element.prototype, 'requestPointerLock', { value: undefined, configurable: true });
    globalThis.xrRequests = 0;
    Object.defineProperty(navigator, 'xr', { value: {
      isSessionSupported: async () => true,
      requestSession: async () => { globalThis.xrRequests++; throw new Error('TEST: reached native XR session request'); },
    }, configurable: true });
  });
  await page.goto(process.env.APP_URL ?? 'http://localhost:5173/', { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => !document.querySelector('.vr').disabled, { timeout: 120000 });
  // Legacy browsers return undefined, which must not cause a .catch TypeError.
  await page.$eval('canvas', canvas => {
    canvas.requestPointerLock = () => { globalThis.legacyLockCalled = true; };
    canvas.dispatchEvent(new MouseEvent('click'));
  });
  assert.equal(await page.evaluate(() => globalThis.legacyLockCalled), true);
  assert.equal(await page.$('[role=alert]'), null);
  await page.$eval('canvas', canvas => {
    delete canvas.requestPointerLock;
    canvas.dispatchEvent(new MouseEvent('click'));
  });
  await page.waitForFunction(() => document.querySelector('[role=alert]')?.innerText.includes('Desktop mouse-look is unavailable'));
  await page.$eval('.vr', button => button.click());
  await page.waitForFunction(() => document.querySelector('[role=alert]')?.innerText.includes('TEST: reached native XR session request'));
  assert.equal(await page.evaluate(() => globalThis.xrRequests), 1);
  assert.deepEqual(uncaught, []);
  console.log('PASS: XR entry without pointer-lock APIs, explicit unsupported desktop control, and legacy void requestPointerLock.');
} finally { await browser.close(); }
