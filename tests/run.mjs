import puppeteer from 'puppeteer-core';
import fs from 'fs';
const out='/tmp/shots68';fs.mkdirSync(out,{recursive:true});
const browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:'new',args:['--no-sandbox','--disable-dev-shm-usage','--hide-scrollbars']});
let fails=[];
const check=(name,ok,extra='')=>{console.log((ok?'PASS':'FAIL')+' '+name+(extra?' — '+extra:''));if(!ok)fails.push(name);};
const COMBOS=[['desktop','light',1440,900],['desktop','dark',1440,900],['mobile','light',390,844],['mobile','dark',390,844]];
const only=process.argv[2];
for(const [vname,theme,w,h] of COMBOS){
 if(only&&(`${vname}-${theme}`)!==only)continue;
 for(const v of [{name:vname,w,h}]){
 for(const theme2 of [theme]){
  const page=await browser.newPage();
  await page.setViewport({width:v.w,height:v.h,deviceScaleFactor:1});
  const errs=[];page.on('pageerror',e=>errs.push(String(e)));page.on('console',m=>{if(m.type()==='error')errs.push(m.text());});
  await page.evaluateOnNewDocument(t=>localStorage.setItem('theme',t),theme);
  await page.goto('http://localhost:8068',{waitUntil:'networkidle2'});
  // env physics sanity
  const env=await page.evaluate(()=>{let s=window._test.env.reset();let t=0;
    for(;t<500;t++){const r=window._test.env.step(s,Math.random()<.5?0:1);s=r.s;if(r.done)break;}
    return {randomSteps:t+1,vecLen:window._test.env.vec(s).length};});
  check(`${v.name}/${theme} env terminates on random play`,env.randomSteps<500,env.randomSteps+' steps');
  check(`${v.name}/${theme} state vector is 4d`,env.vecLen===4);
  if(v.name==='desktop'&&theme==='light'){
    // heavy ML training once: 60 episodes
    const hist=await page.evaluate(()=>window._test.trainSync(80,4));
    check(`${v.name}/${theme} trained 80 episodes`,hist.length===80);
    const early=hist.slice(0,10).reduce((a,b)=>a+b,0)/10, late=hist.slice(-10).reduce((a,b)=>a+b,0)/10;
    check(`${v.name}/${theme} balance time improves`,late>early,`early ${early.toFixed(0)} -> late ${late.toFixed(0)}`);
    const best=await page.evaluate(()=>window._test.eval());
    check(`${v.name}/${theme} trained agent beats random`,best>env.randomSteps+10,`best ${best} vs random ${env.randomSteps}`);
  }
  // canvas draws
  const drawn=await page.$eval('#cv',e=>{const c=e.getContext('2d');const d=c.getImageData(0,0,e.width,e.height).data;for(let i=3;i<d.length;i+=4)if(d[i]>0)return true;return false;});
  check(`${v.name}/${theme} cartpole canvas drawn`,drawn);
  // UI train button runs and re-enables
  await page.evaluate(()=>{document.getElementById('pEp').value='12';document.getElementById('btnTrain').click();});
  await page.waitForFunction(()=>!document.getElementById('btnTrain').disabled,{timeout:120000});
  const st=await page.evaluate(()=>window._test.state());
  check(`${v.name}/${theme} UI training completes`,st.trained===true&&st.hist===12,JSON.stringify(st));
  check(`${v.name}/${theme} watch button enabled`,await page.$eval('#btnWatch',e=>!e.disabled));
  const kpi=await page.$eval('#kpis',e=>e.textContent);
  check(`${v.name}/${theme} KPIs show 12/12`,/12\/12/.test(kpi),kpi.slice(0,40));
  // chart rendered
  check(`${v.name}/${theme} chart rendered`,await page.$eval('#chSteps',e=>e.width>0));
  await page.evaluate(()=>document.getElementById('watch').scrollIntoView({block:'center'}));
  await new Promise(s=>setTimeout(s,300));
  await page.screenshot({path:`${out}/sim-${v.name}-${theme}.png`});
  await page.evaluate(()=>document.getElementById('chSteps').scrollIntoView({block:'center'}));
  await new Promise(s=>setTimeout(s,250));
  await page.screenshot({path:`${out}/curve-${v.name}-${theme}.png`});
  check(`${v.name}/${theme} no console errors`,errs.length===0,errs.slice(0,3).join('|'));
  await page.close();
 }
 }
}
await browser.close();
if(fails.length){console.log('\nFAILURES: '+fails.join(', '));process.exit(1);}
console.log('\nALL PASS');
