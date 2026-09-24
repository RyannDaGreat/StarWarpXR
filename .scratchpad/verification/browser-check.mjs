import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';

const browser = await puppeteer.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  page.on('console', message => console.log('BROWSER', message.type(), message.text()));
  page.on('pageerror', error => console.error('PAGE ERROR', error));
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => !document.querySelector('.modes button').disabled, { timeout: 120000 });
  await new Promise(resolve => setTimeout(resolve, 2500));
  await page.screenshot({ path: '.scratchpad/scene.png' });
  console.log('ALERT:', await page.$eval('body', element => element.innerText));
  await page.click('.modes button:nth-child(2)');
  await new Promise(resolve => setTimeout(resolve, 2500));
  await page.screenshot({ path: '.scratchpad/stars.png' });
  assert.equal(await page.$('[role=alert]'), null);
} finally { await browser.close(); }
