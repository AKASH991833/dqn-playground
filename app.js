/* DQN Playground - Deep Q-Network on CartPole with TensorFlow.js */
(function(){
'use strict';
const $=s=>document.querySelector(s);

// ---- CartPole environment (classic dynamics) ----
const ENV={
  gravity:9.8,masscart:1.0,masspole:0.1,total_mass:1.1,length:0.5,force_mag:10.0,tau:0.02,
  theta_threshold:12*2*Math.PI/360,x_threshold:2.4,maxSteps:500,
  reset(){return {x:(Math.random()-.5)*.1,xd:(Math.random()-.5)*.1,th:(Math.random()-.5)*.1,thd:(Math.random()-.5)*.1};},
  step(s,a){
    const force=a===1?this.force_mag:-this.force_mag;
    const costh=Math.cos(s.th),sinth=Math.sin(s.th);
    const temp=(force+this.masspole*this.length*s.thd*s.thd*sinth)/this.total_mass;
    const thacc=(this.gravity*sinth-costh*temp)/(this.length*(4/3-this.masspole*costh*costh/this.total_mass));
    const xacc=temp-this.masspole*this.length*thacc*costh/this.total_mass;
    const ns={x:s.x+this.tau*s.xd,xd:s.xd+this.tau*xacc,th:s.th+this.tau*s.thd,thd:s.thd+this.tau*thacc};
    const done=Math.abs(ns.x)>this.x_threshold||Math.abs(ns.th)>this.theta_threshold;
    return {s:ns,done,reward:1};
  },
  vec:s=>[s.x/2.4,s.xd/3,s.th/0.21,s.thd/3]
};

// ---- DQN agent ----
function buildModel(units,lr){
  const m=tf.sequential();
  m.add(tf.layers.dense({inputShape:[4],units,activation:'relu'}));
  m.add(tf.layers.dense({units,activation:'relu'}));
  m.add(tf.layers.dense({units:2}));
  m.compile({optimizer:tf.train.adam(lr),loss:'meanSquaredError'});
  return m;
}
let online=null,target=null,replay=[],stepsHist=[],trained=false,training=false;

function act(stateVec,eps){
  if(Math.random()<eps)return Math.random()<0.5?0:1;
  return tf.tidy(()=>online.predict(tf.tensor2d([stateVec])).argMax(1).dataSync()[0]);
}
async function learn(batch,gamma){
  if(replay.length<batch)return;
  const batchArr=[];const seen=new Set();
  while(batchArr.length<batch){const i=Math.floor(Math.random()*replay.length);if(!seen.has(i)){seen.add(i);batchArr.push(replay[i]);}}
  const states=tf.tensor2d(batchArr.map(b=>b.s));
  const nexts=tf.tensor2d(batchArr.map(b=>b.ns));
  const qNext=tf.tidy(()=>target.predict(nexts).max(1));
  const qNextData=await qNext.data();qNext.dispose();nexts.dispose();
  const qs=await online.predict(states).array();states.dispose();
  batchArr.forEach((b,i)=>{
    qs[i][b.a]=b.done?b.r:b.r+gamma*qNextData[i];
  });
  const targetTensor=tf.tensor2d(qs);
  const statesTensor=tf.tensor2d(batchArr.map(b=>b.s));
  await online.fit(statesTensor,targetTensor,{epochs:1,verbose:0});
  statesTensor.dispose();targetTensor.dispose();
}

function drawPole(s){
  const cv=$('#cv'),ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.clearRect(0,0,W,H);
  const cartY=H-60,cartW=64,cartH=32;
  const cx=W/2+(s.x/2.4)*(W/2-60);
  const dark=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim()||'#0f172a';
  const accent=getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()||'#1d4ed8';
  ctx.strokeStyle=dark;ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(0,cartY+cartH+8);ctx.lineTo(W,cartY+cartH+8);ctx.stroke();
  ctx.fillStyle=dark;ctx.fillRect(cx-cartW/2,cartY,cartW,cartH);
  const px=cx+Math.sin(s.th)*120,py=cartY-Math.cos(s.th)*120;
  ctx.strokeStyle=accent;ctx.lineWidth=7;ctx.lineCap='round';
  ctx.beginPath();ctx.moveTo(cx,cartY);ctx.lineTo(px,py);ctx.stroke();
  ctx.fillStyle=accent;ctx.beginPath();ctx.arc(px,py,10,0,7);ctx.fill();
}

let chart=null;
function makeChart(){
  if(chart)chart.destroy();
  const ink=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim()||'#0f172a';
  chart=new Chart($('#chSteps'),{type:'line',data:{labels:[],datasets:[{label:'steps survived',data:[],borderColor:'#1d4ed8',tension:.3,pointRadius:0,borderWidth:2},{label:'moving avg (10)',data:[],borderColor:'#d97706',tension:.3,pointRadius:0,borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:ink}}},scales:{x:{ticks:{color:ink,maxTicksLimit:10}},y:{ticks:{color:ink},beginAtZero:true,max:500}}}});
}
function updateChart(){
  chart.data.labels=stepsHist.map((_,i)=>i+1);
  chart.data.datasets[0].data=stepsHist;
  chart.data.datasets[1].data=stepsHist.map((_,i)=>{const s=stepsHist.slice(Math.max(0,i-9),i+1);return s.reduce((a,b)=>a+b,0)/s.length;});
  chart.update('none');
}
function kpis(ep,total,eps,last){
  const avg=stepsHist.length?stepsHist.slice(-10).reduce((a,b)=>a+b,0)/Math.min(10,stepsHist.length):0;
  $('#kpis').innerHTML=[['Episode',`${ep}/${total}`],['Epsilon',eps.toFixed(2)],['Last run',last??'-'],['Avg (10)',avg.toFixed(0)]]
    .map(([l,v])=>`<div class="card kpi"><div class="small muted">${l}</div><div class="kpi-val" style="font-size:1.1rem">${v}</div></div>`).join('');
}

async function train(){
  if(training)return;
  training=true;$('#btnTrain').textContent='Training...';$('#btnTrain').disabled=true;
  const totalEp=+$('#pEp').value,lr=+$('#pLr').value,gamma=+$('#pGamma').value,batch=+$('#pBatch').value,units=+$('#pUnits').value,sync=+$('#pSync').value;
  if(online){online.dispose();target.dispose();}
  online=buildModel(units,lr);target=buildModel(units,lr);
  target.setWeights(online.getWeights());
  replay=[];stepsHist=[];
  for(let ep=1;ep<=totalEp;ep++){
    let s=ENV.reset(),eps=Math.max(0.05,1-ep/(totalEp*0.6)),steps=0;
    for(let t=0;t<ENV.maxSteps;t++){
      const sv=ENV.vec(s);
      const a=act(sv,eps);
      const {s:ns,done,reward}=ENV.step(s,a);
      replay.push({s:sv,a,r:reward,ns:ENV.vec(ns),done});
      if(replay.length>10000)replay.shift();
      await learn(batch,gamma);
      s=ns;steps++;
      if(done)break;
    }
    stepsHist.push(steps);
    if(ep%sync===0)target.setWeights(online.getWeights());
    kpis(ep,totalEp,eps,steps);updateChart();
    if(ep%3===0){drawPole(s);await new Promise(r=>setTimeout(r,0));}
  }
  trained=true;training=false;
  $('#btnTrain').textContent='Train';$('#btnTrain').disabled=false;$('#btnWatch').disabled=false;
  toast('Training done - watch the agent or retrain');
}
let watching=false;
$('#btnWatch').addEventListener('click',async()=>{
  if(watching||!trained)return;
  watching=true;$('#simStatus').textContent='Running greedy policy...';
  let s=ENV.reset();
  for(let t=0;t<ENV.maxSteps;t++){
    const a=act(ENV.vec(s),0);
    const r=ENV.step(s,a);s=r.s;
    drawPole(s);
    if(r.done){$('#simStatus').textContent=`Pole fell after ${t+1} steps - retrain or tune to do better`;watching=false;return;}
    await new Promise(r2=>setTimeout(r2,20));
  }
  $('#simStatus').textContent='Balanced for the full 500 steps - solved!';
  watching=false;
});
$('#btnTrain').addEventListener('click',()=>train().catch(e=>{training=false;$('#btnTrain').disabled=false;$('#btnTrain').textContent='Train';toast('Training error: '+e.message);console.error(e);}));
$('#btnReset').addEventListener('click',()=>{
  if(online){online.dispose();target.dispose();online=null;target=null;}
  replay=[];stepsHist=[];trained=false;$('#btnWatch').disabled=true;makeChart();kpis(0,+$('#pEp').value,1,null);drawPole(ENV.reset());toast('Reset');
});
document.addEventListener('themechange',()=>{makeChart();updateChart();});

makeChart();kpis(0,150,1,null);drawPole(ENV.reset());

window._test={
  env:ENV,buildModel,
  // headless training loop for tests (fewer episodes)
  trainSync:async(eps,learnEvery=2)=>{const lr=0.005,gamma=0.95,batch=32;
    if(online){online.dispose();target.dispose();}
    online=buildModel(24,lr);target=buildModel(24,lr);target.setWeights(online.getWeights());
    replay=[];stepsHist=[];
    // warmup: random transitions so early batches are diverse
    for(let i=0;i<600;i++){let s=ENV.reset();
      for(let t=0;t<60;t++){const sv=ENV.vec(s);const a=Math.random()<.5?0:1;
        const {s:ns,done,reward}=ENV.step(s,a);
        replay.push({s:sv,a,r:reward,ns:ENV.vec(ns),done});s=ns;if(done)break;}
      if(replay.length>=600)break;}
    let stepCount=0;
    for(let ep=1;ep<=eps;ep++){
      let s=ENV.reset(),es=Math.max(0.05,1-ep/(eps*0.6)),steps=0;
      for(let t=0;t<500;t++){const sv=ENV.vec(s);const a=act(sv,es);
        const {s:ns,done,reward}=ENV.step(s,a);
        replay.push({s:sv,a,r:reward,ns:ENV.vec(ns),done});if(replay.length>10000)replay.shift();
        if(++stepCount%learnEvery===0)await learn(batch,gamma);
        s=ns;steps++;if(done)break;}
      stepsHist.push(steps);
      if(ep%5===0)target.setWeights(online.getWeights());
    }
    trained=true;return stepsHist;
  },
  eval:()=>{let best=0;for(let k=0;k<3;k++){let s=ENV.reset(),t=0;for(;t<500;t++){const a=act(ENV.vec(s),0);const r=ENV.step(s,a);s=r.s;if(r.done)break;}best=Math.max(best,t+1);}return best;},
  state:()=>({trained,hist:stepsHist.length,replay:replay.length})};
})();
