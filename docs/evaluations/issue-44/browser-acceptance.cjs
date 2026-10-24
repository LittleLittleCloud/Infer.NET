// Independent browser acceptance by Codex; no production measurements generated.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const playwrightModule = process.env.PLAYWRIGHT_MODULE;
if (!playwrightModule || !process.env.CHROMIUM_PATH) throw new Error('Set PLAYWRIGHT_MODULE and CHROMIUM_PATH to your installed browser runtime.');
const {chromium} = require(playwrightModule);
const output = process.env.EVAL_OUTPUT || __dirname;
const checks = [], errors = [];
const production = process.env.PRODUCTION_URL || 'http://127.0.0.1:8746/';
const fixture = process.env.FIXTURE_URL || 'http://127.0.0.1:8747/';
async function check(name, run) {
  try { const details = await run(); checks.push({name,passed:true,details}); }
  catch(e) {checks.push({name,passed:false,error:e.message});}
}
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH});
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror',e=>errors.push(e.message));
  const badResponses=[];
  page.on('response',r=>{if(r.status()>=400)badResponses.push({url:r.url(),status:r.status()});});
  const wait = async()=>{await page.waitForFunction(()=>document.querySelectorAll('#tasks tbody tr').length===20);};
  await page.goto(production); await wait();
  await check('Four production panels, no former homepage sections',async()=>{
    assert.equal(await page.locator('.trend-grid section').count(),4);
    assert.equal(await page.locator('.hero,.stats,#device-topology,.milestones').count(),0);
  });
  await check('Default main rerun retains failed task and fixed denominator',async()=>{
    assert.equal(await page.locator('#run').inputValue(),'pi-main-rerun-20261023');
    assert.match(await page.locator('#pass-value').textContent(),/95%.*19\/20/);
    assert.equal(await page.locator('#tasks tbody tr').count(),20);
    assert.match(await page.locator('#tasks tbody tr').nth(4).textContent(),/HumanEval\/149.*error/);
  });
  await check('Missing history not plotted as zero',async()=>{
    for(const key of ['prefill','decode','ttft']){
      assert.match(await page.locator('#'+key+'-value').textContent(),/Unavailable|Not acquired/);
      assert.equal(await page.locator('#'+key+'-chart button').count(),0);
    }
    assert.match(await page.locator('#tasks tbody tr').first().textContent(),/Unavailable|Not acquired/);
  });
  await check('Original 20-task history matches source token totals and first row',async()=>{
    await page.selectOption('#run','pi-main-20261023');
    assert.match(await page.locator('#pass-value').textContent(),/100%.*20\/20/);
    const first=await page.locator('#tasks tbody tr').first().locator('th,td').allTextContents();
    assert.equal(first[0],'HumanEval/156 / 1');
    assert.equal(Number(first[1]),36.475);
    assert.equal(Number(first[2].replaceAll(',','')),10989);
    assert.equal(Number(first[3].replaceAll(',','')),1030);
    const totals=await page.locator('#tasks tfoot').textContent();
    assert.match(totals,/222,744/);assert.match(totals,/19,212/);
  });
  await check('Run selection permalink reload and browser Back',async()=>{
    const url=page.url();assert.match(url,/run=pi-main-20261023/);
    await page.reload();await wait();assert.equal(await page.locator('#run').inputValue(),'pi-main-20261023');
    await page.selectOption('#run','pi-main-rerun-20261023');await page.goBack();
    assert.equal(await page.locator('#run').inputValue(),'pi-main-20261023');
  });
  await check('Keyboard selection updates all details without losing focus',async()=>{
    const p=page.locator('#pass-chart button[data-run="pi-main-rerun-20261023"]');
    await p.focus();await page.keyboard.press('Enter');
    assert.equal(await page.locator('#run').inputValue(),'pi-main-rerun-20261023');
    assert.equal(await p.getAttribute('aria-pressed'),'true');assert.equal(await p.evaluate(e=>document.activeElement===e),true);
  });
  for(const width of [1280,768,390,320]){
    await page.setViewportSize({width,height:900}); await page.reload(); await wait();
    await check(`Responsive layout and readable chart labels at ${width}px`,async()=>{
      const m=await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth,
        labels:[...document.querySelectorAll('svg text')].map(t=>t.getBoundingClientRect().height),
        charts:[...document.querySelectorAll('.plot')].map(el=>({w:el.clientWidth,h:el.clientHeight}))}));
      assert.ok(m.width<=m.viewport+1,JSON.stringify(m));
      assert.ok(m.labels.every(v=>v>=11),'Chart labels too small: '+JSON.stringify(m.labels));return m;
    });
    if(width===1280||width===390||width===320)await page.screenshot({path:path.join(output,`production-${width}.png`),fullPage:true});
  }
  await page.setViewportSize({width:1280,height:900});await page.goto(fixture);await wait();
  await check('Synthetic fixture remains explicitly labeled',async()=>{assert.match(await page.locator('header').textContent(),/SYNTHETIC/);});
  await check('Candidate click synchronizes all four charts and metadata',async()=>{
    await page.locator('#prefill-chart button[data-run="fixture-candidate"]').click();
    assert.equal(await page.locator('#run').inputValue(),'fixture-candidate');
    assert.match(await page.locator('#run-state').textContent(),/candidate/);
    assert.match(await page.locator('#ttft-value').textContent(),/120 ms/);
    for(const key of ['prefill','decode','ttft','pass'])assert.equal(await page.locator('#'+key+'-chart button[data-run="fixture-candidate"]').getAttribute('aria-pressed'),'true');
  });
  await check('Failure, real zero and unknown assertions remain distinguishable',async()=>{
    await page.selectOption('#run','fixture-failure');assert.match(await page.locator('#prefill-value').textContent(),/Unavailable|Not acquired/);
    await page.selectOption('#run','fixture-zero');assert.equal(await page.locator('#prefill-value').textContent(),'0 token/s');assert.equal(await page.locator('#ttft-value').textContent(),'0 ms');
    await page.selectOption('#run','fixture-unverified');assert.match(await page.locator('#pass-value').textContent(),/Unverified/);
  });
  await check('Incompatible BF16 series never mixed into FP32 charts',async()=>{
    await page.selectOption('#configuration','fixture-incompatible-bf16');
    assert.deepEqual(await page.locator('#prefill-chart button').evaluateAll(es=>es.map(e=>e.dataset.run)),['fixture-bf16']);
    await page.goto(fixture+'#run=fixture-candidate');await wait();
    assert.notEqual(await page.locator('#configuration').inputValue(),'fixture-incompatible-bf16');
  });
  await check('Candidate fixture screenshot and same-SHA repeated runs preserved',async()=>{
    assert.equal(await page.locator('#prefill-chart button[data-run="fixture-main"]').count(),1);
    assert.equal(await page.locator('#prefill-chart button[data-run="fixture-repeat"]').count(),1);
    await page.screenshot({path:path.join(output,'fixture-candidate-1280.png'),fullPage:true});
  });
  for(const width of [390,320]){
    await page.setViewportSize({width,height:900});await page.reload();await wait();
    await check(`Measured fixture points selectable and readable at ${width}px`,async()=>{
      const labels=await page.locator('svg text').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().height));
      assert.ok(labels.every(v=>v>=11),'Small labels: '+JSON.stringify(labels));
      await page.selectOption('#run','fixture-main');await page.locator('#ttft-chart button[data-run="fixture-candidate"]').click();
      assert.equal(await page.locator('#run').inputValue(),'fixture-candidate');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    });
  }
  await check('Local resources load and browser has no runtime errors',async()=>{assert.deepEqual(errors,[]);assert.deepEqual(badResponses,[]);});
  const result={browserVersion:browser.version(),playwrightVersion:require(path.join(playwrightModule,'package.json')).version,
    productionRuns:2,plannedTasksPerRun:20,checks,passed:checks.filter(c=>c.passed).length,failed:checks.filter(c=>!c.passed).length,errors,badResponses};
  fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
  await browser.close();if(result.failed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
