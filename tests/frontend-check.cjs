const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.join(__dirname, '..', 'web');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const elements = new Map();
function element(id = '') {
  return { id, value: '', disabled: false, hidden: false, checked: false, open: false, children: [], listeners: {}, textContent: '', style: {setProperty(){}},
    classList: {toggle(){},add(){},remove(){}}, dataset: {},
    setAttribute(){}, addEventListener(name,callback){(this.listeners[name]||=[]).push(callback);}, querySelectorAll(){ return []; }, closest(){return null;},
    replaceChildren(...nodes){this.children=[...nodes];}, appendChild(node){this.children.push(node);return node;}, showModal(){this.open=true;}, close(){this.open=false;},
    clientWidth: 800, clientHeight: 500,
  };
}
for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1], element(match[1]));
for (const match of html.matchAll(/<input\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
  const attrs = Object.fromEntries([...match[0].matchAll(/([\w-]+)="([^"]*)"/g)].map(m => [m[1],m[2]]));
  Object.assign(elements.get(match[1]), attrs);
  elements.get(match[1]).checked = /\bchecked\b/.test(match[0]);
}
elements.get('diff-channel').value = 'rgb';
elements.get('test-target').value = 'cage';
const pendingImages = new Map();
let autoImage = false;
class FakeImage {
  constructor(){ this.naturalWidth=2; this.naturalHeight=1; }
  set src(value){ pendingImages.set(value,this); if(autoImage) queueMicrotask(()=>this.onload()); }
}
const store = new Map();
const downloads = [];
const context = {
  console, Uint8Array, Uint8ClampedArray, Map, Set, Promise, Number, Math, JSON, Object, Array,
  Image: FakeImage, Option: function(text,value){this.text=text;this.value=value;},
  crypto: {randomUUID:()=> 'preset-id'},
  localStorage: {getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},
  setTimeout:(callback,delay)=>{if(delay<=1000)queueMicrotask(callback);return 0;}, clearTimeout(){}, requestAnimationFrame:()=>0, cancelAnimationFrame(){},
  ResizeObserver: class {observe(){}}, window: {devicePixelRatio:1,addEventListener(){}},
  document: {
    getElementById:id=>{assert.ok(elements.has(id),`Missing HTML element ${id}`);return elements.get(id);},
    querySelectorAll:()=>[], querySelector:()=>null, body:{dataset:{},appendChild(){}},
    createElement(tag){
      if(tag === 'a') return {click(){downloads.push({url:this.href,name:this.download});},remove(){}};
      if(tag !== 'canvas') return element();
      const canvas={width:0,height:0};
      canvas.getContext=()=>({ createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}), putImageData:data=>{canvas.pixels=data.data;} });
      return canvas;
    },
  },
};
context.globalThis=context;
const source=fs.readFileSync(path.join(root,'app.js'),'utf8').replace(/\ninitialize\(\);\s*$/,'');
vm.runInNewContext(source+'\nglobalThis.test={state,settings,normalizedSettings,applySettings,diffImage,useSource,useResult,processImage,openSettingsDialog,closeSettingsDialog,saveSettingsDialog,updateCards,restoreSelection,watchJob,busy,syncMethodControls,readPresets,savePreset,selectPreset,downloadStage1,download,testEligible,startProtectionTest,watchTestJob,openTestDialog,closeTestDialog,renderTestResults,invalidateTest,resumeActiveJob,baselineGate,setVisualBaseline,changeBaselineCutoff};', context);
const t=context.test;
const descendants = node => [node,...(node.children||[]).flatMap(descendants)];
const renderedText = node => descendants(node).map(item=>item.textContent||'').join(' ');
async function until(predicate){for(let i=0;i<80;i++){if(predicate())return;await Promise.resolve();}assert.fail('Fixture did not reach the expected async stage');}

(async()=>{
  assert.equal(t.normalizedSettings().max_size,1280);
  assert.equal(t.state.testBaselineCutoff,.80,'Default baseline quality cutoff must be 0.80');
  assert.equal(elements.get('test-baseline-cutoff').value,'0.80');
  assert.match(html,/not a confidence percentage/);
  assert.doesNotMatch(html,/class="local-note"|Local processing · research use only/,'Removed footer must not leave an empty element');
  assert.match(html,/<dt>License<\/dt><dd>Research version:/,'License information stays available in Help');
  assert.equal(t.normalizedSettings().mode,'phantomseal_faceshield');
  assert.equal(t.normalizedSettings().phantom_iterations,1000);
  assert.equal(t.normalizedSettings({},true).mode,'faceshield','Old settings/presets must keep FaceShield-only behavior');
  assert.equal(t.normalizedSettings({mode:'phantomseal'},true).mode,'phantomseal');
  for(const value of [9,2001,10.5,NaN]) assert.throws(()=>t.normalizedSettings({phantom_iterations:value}));
  assert.throws(()=>t.normalizedSettings({mode:'invalid'}));
  assert.throws(()=>t.normalizedSettings({iterations:NaN}));
  assert.throws(()=>t.normalizedSettings({noise_clamp:1000}));
  t.state.initializing=true; t.updateCards();
  assert.equal(elements.get('file-input').disabled,true,'Startup must lock the picker');
  t.state.initializing=false;
  t.applySettings({});
  assert.equal(elements.get('phantom-iterations').disabled,false);
  assert.equal(elements.get('iterations').disabled,false);
  t.applySettings({mode:'phantomseal'});
  assert.equal(elements.get('phantom-iterations').disabled,false);
  for(const id of ['iterations','noise-clamp','step-size']) assert.equal(elements.get(id).disabled,true);
  t.applySettings({mode:'faceshield'});
  assert.equal(elements.get('phantom-iterations').disabled,true);
  assert.equal(elements.get('iterations').disabled,false);
  assert.equal(elements.get('max-size').disabled,false);
  t.state.ready=true; t.state.uploadedSource={id:'available'};
  t.state.engines={phantomseal:{ready:false,message:'PhantomSeal weights unavailable.'},faceshield:{ready:true,message:'Ready'}};
  t.applySettings({mode:'phantomseal_faceshield'});
  assert.equal(elements.get('process-button').disabled,true,'Missing PhantomSeal must block chain');
  assert.match(elements.get('method-readiness').textContent,/weights unavailable/);
  t.applySettings({mode:'faceshield'});
  assert.equal(elements.get('process-button').disabled,false,'Available FaceShield remains usable');
  t.state.engines={phantomseal:{ready:true,message:'PhantomSeal ensemble: ready'},faceshield:{ready:true}};
  t.applySettings({});
  assert.equal(elements.get('method-readiness').textContent,'PhantomSeal ensemble: ready · FaceShield: ready','Ready engine descriptions must use the actual API message');
  t.openSettingsDialog();
  assert.equal(elements.get('dialog-method-readiness').textContent,'PhantomSeal ensemble: ready · FaceShield: ready');
  t.closeSettingsDialog();
  t.busy(true);
  for(const id of ['method-select','phantom-iterations','iterations','max-size','dialog-method-select']) assert.equal(elements.get(id).disabled,true);
  t.busy(false);
  assert.equal(elements.get('method-select').disabled,false);
  t.openSettingsDialog();
  elements.get('dialog-method-select').value='phantomseal'; t.syncMethodControls('dialog-');
  assert.equal(elements.get('dialog-iterations').disabled,true);
  assert.equal(elements.get('dialog-phantom-iterations').disabled,false);
  elements.get('dialog-iterations').value='9999'; t.syncMethodControls('dialog-');
  assert.equal(Number(elements.get('dialog-iterations').value),30,'Inactive invalid fields must not prevent saving another method');
  elements.get('dialog-iterations').value='44';
  t.closeSettingsDialog();
  assert.equal(t.settings().iterations,30,'Cancel must discard draft');
  assert.equal(t.settings().mode,'phantomseal_faceshield','Cancel must discard method draft');
  t.openSettingsDialog();
  elements.get('dialog-method-select').value='phantomseal';
  elements.get('dialog-phantom-iterations').value='1200';
  elements.get('dialog-iterations').value='44';
  t.saveSettingsDialog();
  assert.equal(t.settings().iterations,44,'Save must apply draft');
  assert.equal(t.settings().mode,'phantomseal');
  assert.equal(t.settings().phantom_iterations,1200);
  store.set('faceshield.presets.v1',JSON.stringify({version:1,presets:[{id:'legacy',name:'Old FaceShield',params:{iterations:44}}]}));
  t.readPresets(); assert.equal(t.state.presets[0].params.mode,'faceshield','Legacy named presets must migrate to FaceShield');
  t.openSettingsDialog();
  elements.get('dialog-method-select').value='phantomseal_faceshield'; elements.get('dialog-phantom-iterations').value='1100';
  elements.get('preset-name').value='Combined'; t.savePreset(); t.readPresets();
  const combined=t.state.presets.find(p=>p.name==='Combined');
  assert.equal(combined.params.mode,'phantomseal_faceshield'); assert.equal(combined.params.phantom_iterations,1100);
  elements.get('preset-select').value=combined.id; t.selectPreset(); t.saveSettingsDialog();
  assert.equal(t.settings().mode,'phantomseal_faceshield'); assert.equal(t.settings().phantom_iterations,1100);

  t.state.uploadVersion=1;
  const old={id:'old',url:'/api/image/old',name:'old.png',width:2,height:1};
  const current={id:'current',url:'/api/image/current',name:'current.png',width:2,height:1};
  const older=t.useSource(old,true,1);
  t.state.uploadVersion=2;
  const newer=t.useSource(current,true,2);
  pendingImages.get(current.url).onload(); await newer;
  pendingImages.get(old.url).onload(); assert.equal(await older,false);
  assert.equal(t.state.source.id,'current','Late decode must not replace new upload');
  store.set('faceshield.selection.v1', JSON.stringify({uploadedSource:old,source:old,result:null}));
  const restore=t.restoreSelection(2);
  t.state.uploadVersion=3;
  pendingImages.get(old.url).onload(); await restore;
  assert.equal(t.state.uploadedSource.id,'current','Late startup restore must not replace a fresh upload');

  t.state.result={id:'result',name:'result.png',width:2,height:1};
  t.state.resultImage={naturalWidth:2,naturalHeight:1};
  t.state.pair={a:new Uint8Array([10,20,30,255,50,60,70,255]),b:new Uint8Array([13,20,29,255,50,60,70,255])};
  elements.get('diff-gain').value='2';
  let diff=t.diffImage();
  assert.equal(diff.width,2); assert.equal(diff.height,1);
  assert.deepEqual([...diff.pixels],[6,0,2,255,0,0,0,255]);
  elements.get('diff-channel').value='red';
  diff=t.diffImage();
  assert.deepEqual([...diff.pixels],[6,6,6,255,0,0,0,255]);
  t.state.diffKey=''; t.state.pair.b=t.state.pair.a;
  assert.deepEqual([...t.diffImage().pixels],[0,0,0,255,0,0,0,255]);

  const input={id:'full-original',url:'/api/image/full-original',name:'source.png',width:2400,height:1200};
  const baseline={id:'baseline',url:'/api/image/baseline',name:'source.png',width:1280,height:640};
  const result={id:'protected',url:'/api/image/protected',name:'protected.png',width:1280,height:640};
  const intermediate={id:'phantom',url:'/api/image/phantom',name:'phantom-stage.png',width:1280,height:640};
  t.state.uploadedSource=input; t.state.ready=true; t.state.result=null; autoImage=true;
  let calls=0, submitted;
  context.fetch=async(url,options)=>{
    calls++;
    if(url==='/api/process'){submitted=JSON.parse(options.body);return {ok:true,json:async()=>({id:'job'})};}
    assert.equal(url,'/api/job/job');
    return {ok:true,json:async()=>({id:'job',source_id:input.id,status:'done',progress:100,elapsed:1,source:baseline,result,stage1:intermediate,params:{mode:'phantomseal_faceshield'},metrics:{identical:false,ssim:0.9,psnr_db:30,changed_pct:2}})};
  };
  t.state.uploading=true; await t.processImage(); assert.equal(calls,0,'Pending upload must block processing');
  t.state.uploading=false; await t.processImage();
  assert.equal(submitted.source_id,'full-original','Rerun must use original upload');
  assert.equal(submitted.params.mode,'phantomseal_faceshield');
  assert.equal(submitted.params.phantom_iterations,1100);
  assert.equal(t.state.uploadedSource.id,'full-original');
  assert.equal(t.state.source.id,'baseline','Comparison must use normalized baseline');
  assert.equal(t.state.result.id,'protected');
  assert.equal(t.state.stage1.id,'phantom');
  assert.equal(elements.get('export-stage1-button').hidden,false);
  t.downloadStage1(); await t.download();
  assert.equal(downloads.at(-2).url,intermediate.url,'Separate export must download intermediate');
  assert.equal(downloads.at(-1).url,result.url,'Main export must still download final result');
  assert.equal(t.state.source.id,baseline.id,'Intermediate export must not change clean comparison source');
  assert.equal(t.state.result.id,result.id);
  assert.equal(t.state.busy,false,'Completion must release controls');
  t.state.uploadedSource=null; t.state.assetIndex.set(input.id,input);
  await t.watchJob('job');
  assert.equal(t.state.uploadedSource.id,'full-original','Active job resume must recover full original from asset index');
  context.fetch=async()=>({ok:false,status:400,json:async()=>({error:'Fixture error'})});
  await t.processImage(); assert.equal(t.state.busy,false,'Error must release controls');
  assert.equal(t.state.stage1,null,'A new attempt must clear stale intermediate');
  assert.equal(elements.get('export-stage1-button').hidden,true);
  assert.equal(elements.get('file-input').disabled,false);
  t.state.stage1=intermediate; await t.useSource(current);
  assert.equal(t.state.stage1,null,'A new photo must clear the old intermediate');
  assert.equal(elements.get('export-stage1-button').hidden,true);
  assert.equal(elements.get('test-jpeg95').checked,true,'JPEG95 must be enabled by default');
  t.state.source=baseline; t.state.result=result; t.state.stage1=intermediate;
  t.state.uploadedSource=input; t.state.testableResultIds.add(result.id);
  t.state.testing={ready:true,message:'SimSwap ready',targets:[{key:'cage'},{key:'wlh'},{key:'zxy'}]};
  t.updateCards(); assert.equal(elements.get('test-protection-button').disabled,false);
  elements.get('test-target').value='wlh';
  const testAsset={id:'swap',url:'/api/image/swap',name:'swap.png',width:224,height:224};
  const testData={target:testAsset,conditions:['png','jpeg95'].map(name=>({name,rows:[
    {key:'clean',label:'Clean original',swap:testAsset,score:.8,source_detected:true,swap_detected:true,status:'ok'},
    {key:'stage1',label:'PhantomSeal first stage',swap:testAsset,score:.45,source_detected:true,swap_detected:true,status:'ok'},
    {key:'final',label:'Earlier combined result',swap:testAsset,score:.46,source_detected:true,swap_detected:true,status:'ok'},
  ],delta_final_minus_stage1:.01})),note:'One model and target; results can vary.',metric:'FaceNet cosine'};
  let resolveFinal, testPayload, polls=0, testPosts=0;
  context.fetch=async(url,options)=>{
    if(url==='/api/test'){
      testPosts++;testPayload=JSON.parse(options.body);
      return {ok:true,json:async()=>({id:'test-job',kind:'test',status:'queued'})};
    }
    assert.equal(url,'/api/job/test-job'); polls++;
    if(polls===1)return {ok:true,json:async()=>({id:'test-job',kind:'test',result_id:result.id,status:'running',progress:55,elapsed:4,stage:'Running PNG swaps'})};
    return new Promise(resolve=>{resolveFinal=resolve;});
  };
  await t.openTestDialog();
  elements.get('test-baseline-cutoff').value='0.75';t.changeBaselineCutoff();
  const runningTest=t.startProtectionTest();
  await until(()=>Boolean(resolveFinal));
  assert.deepEqual(testPayload,{result_id:'protected',target_key:'wlh',jpeg95:true});
  assert.equal(t.state.busy,true); assert.equal(t.state.busyKind,'test');
  assert.equal(elements.get('process-label').textContent,'Testing…');
  for(const id of ['process-button','upload-button','file-input','export-button','export-stage1-button','run-test-button','test-target','test-jpeg95'])assert.equal(elements.get(id).disabled,true,`Test must lock ${id}`);
  assert.equal(elements.get('test-progress').value,55);
  assert.equal(elements.get('test-status').textContent,'Running PNG swaps');
  t.closeTestDialog();assert.equal(elements.get('test-dialog').open,false);
  assert.equal(t.state.busy,true,'Closing modal must not cancel or unlock the active test');
  await t.openTestDialog();assert.equal(elements.get('test-dialog').open,true);
  assert.equal(testPosts,1,'Reopen must not start another GPU test');
  elements.get('method-select').value='faceshield';
  resolveFinal({ok:true,json:async()=>({id:'test-job',kind:'test',result_id:result.id,status:'done',progress:100,elapsed:6,test:testData})});
  await runningTest;
  assert.equal(t.state.busy,false);assert.equal(t.state.testData,testData);
  assert.equal(t.state.source.id,baseline.id);assert.equal(t.state.result.id,result.id);assert.equal(t.state.stage1.id,intermediate.id);
  assert.equal(elements.get('run-test-button').disabled,false);assert.equal(elements.get('file-input').disabled,false);
  let text=renderedText(elements.get('test-results'));
  assert.match(text,/Earlier combined result/,'Result labels must come from the test job, not current settings');
  assert.match(text,/PNG/);assert.match(text,/JPEG95/);assert.match(text,/0\.8000/);
  assert.match(text,/confirm the baseline appearance/);
  assert.doesNotMatch(text,/Chain − PhantomSeal cosine/,'A numeric gate alone must not expose the effect delta');
  const confirmationLabels=descendants(elements.get('test-results')).filter(node=>node.className==='test-visual-confirmation');
  assert.equal(confirmationLabels.length,2);
  const pngConfirmation=confirmationLabels[0].children[0],jpegConfirmation=confirmationLabels[1].children[0];
  assert.equal(pngConfirmation.checked,false);assert.equal(pngConfirmation.disabled,false,'Completion must unlock eligible visual checks');
  pngConfirmation.checked=true;pngConfirmation.listeners.change[0]();
  text=renderedText(elements.get('test-results'));
  assert.match(text,/Baseline accepted for comparison/);assert.match(text,/\+0\.0100/);
  assert.equal(jpegConfirmation.checked,false,'PNG confirmation must not confirm JPEG95');
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,1);
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-reference-image')[0].src,baseline.url,'Header must show actual clean source separately from demo target');
  assert.match(text,/Unprotected swap/);
  assert.doesNotMatch(text,/\b(?:SAFE|PASS)\b/);
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-swap-card').length,6);
  t.closeTestDialog(); await t.openTestDialog(); assert.equal(t.state.testData,testData,'Completed results must survive close/reopen');

  // A single-method job supplies exactly clean/final cards; failed detection stays unmeasured.
  t.state.testData={target:testAsset,conditions:[{name:'png',rows:[testData.conditions[0].rows[0],{key:'final',label:'FaceShield only',swap:null,score:null,source_detected:true,swap_detected:false,status:'swap_detection_failed'}],delta_final_minus_stage1:null}]};
  t.renderTestResults();
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-swap-card').length,2);
  text=renderedText(elements.get('test-results'));
  assert.match(text,/Swap face not detected/);assert.doesNotMatch(text,/Chain − PhantomSeal cosine/);

  // Startup must inspect kind and leave all main assets untouched for a test.
  context.fetch=async url=>({ok:true,json:async()=>({id:'resume-test',kind:'test',result_id:result.id,status:'done',progress:100,
    source:{id:'wrong-source'},result:{id:'wrong-result'},stage1:{id:'wrong-stage'},test:testData})});
  await t.resumeActiveJob('resume-test');
  assert.equal(t.state.source.id,baseline.id);assert.equal(t.state.result.id,result.id);assert.equal(t.state.stage1.id,intermediate.id);
  assert.equal(t.state.testData,testData);assert.equal(t.state.busy,false);

  // An HTTP start error and a worker error both release locks and stay inline.
  context.fetch=async()=>({ok:false,status:409,json:async()=>({error:'Fixture GPU busy'})});
  await t.startProtectionTest();assert.equal(t.state.busy,false);
  assert.equal(elements.get('test-error').hidden,false);assert.match(elements.get('test-error').textContent,/Fixture GPU busy/);
  context.fetch=async url=>({ok:true,json:async()=>url==='/api/test'
    ? {id:'failed-test',kind:'test'}
    : {id:'failed-test',kind:'test',result_id:result.id,status:'error',error:'Fixture inference failed'}});
  await t.startProtectionTest();assert.equal(t.state.busy,false);
  assert.match(elements.get('test-error').textContent,/Fixture inference failed/);
  assert.equal(elements.get('test-target').disabled,false);assert.equal(elements.get('export-button').disabled,false);

  // A late result for a replaced source must never populate its test panel.
  let resolveLate;
  context.fetch=async url=>url==='/api/test'?{ok:true,json:async()=>({id:'late-test',kind:'test'})}:new Promise(resolve=>{resolveLate=resolve;});
  const late=t.startProtectionTest();await until(()=>Boolean(resolveLate));
  await t.useSource(current);
  resolveLate({ok:true,json:async()=>({id:'late-test',kind:'test',result_id:result.id,status:'done',test:testData})});
  await late;
  assert.equal(t.state.testData,null);assert.equal(t.state.result,null);assert.equal(t.state.source.id,current.id);
  assert.equal(elements.get('test-results').hidden,true);assert.equal(elements.get('test-protection-button').disabled,true);
  assert.equal(t.state.busy,false,'Stale completion must still release its own lock');

  // Readiness text remains actionable without pretending the test succeeded.
  t.state.result=result;t.state.testableResultIds.add(result.id);
  t.state.testing={ready:false,message:'SimSwap weights unavailable.'};t.updateCards();
  assert.equal(elements.get('run-test-button').disabled,true);
  assert.match(elements.get('test-readiness').textContent,/weights unavailable/);

  // Poor baselines never expose a protection effect, even if the server supplies a delta.
  const condition=(name,score)=>({name,rows:[
    {...testData.conditions[0].rows[0],score},
    {...testData.conditions[0].rows[1]},
    {...testData.conditions[0].rows[2]},
  ],delta_final_minus_stage1:999});
  t.state.testResultId=result.id;t.state.testJobId='quality-job';
  elements.get('test-baseline-cutoff').value='0.80';t.changeBaselineCutoff();
  const gated={target:testAsset,conditions:[condition('png',.68),condition('jpeg95',.9)]};
  t.state.testData=gated;t.renderTestResults();
  let sections=descendants(elements.get('test-results')).filter(node=>node.className==='test-condition');
  assert.match(renderedText(sections[0]),/Inconclusive — weak baseline/);
  assert.match(renderedText(sections[0]),/0\.6800/);assert.match(renderedText(sections[0]),/0\.80/);
  assert.doesNotMatch(renderedText(sections[0]),/Chain − PhantomSeal cosine|999/);
  assert.equal(t.setVisualBaseline(gated.conditions[0],true),false,'A visual check cannot bypass a weak numeric baseline');
  assert.equal(t.baselineGate(gated.conditions[1]).accepted,false,'A high score still needs visual confirmation');
  assert.match(renderedText(sections[1]),/confirm the baseline appearance/);
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,0);
  t.setVisualBaseline(gated.conditions[1],true);t.renderTestResults();
  sections=descendants(elements.get('test-results')).filter(node=>node.className==='test-condition');
  assert.doesNotMatch(renderedText(sections[0]),/Chain − PhantomSeal cosine/);
  assert.match(renderedText(sections[1]),/Chain − PhantomSeal cosine: \+0\.0100/);
  assert.doesNotMatch(renderedText(sections[1]),/999/,'Accepted delta must be recomputed from measured rows');

  // The conditions are accepted independently; changing the cutoff clears every visual decision.
  elements.get('test-baseline-cutoff').value='0.85';t.changeBaselineCutoff();
  assert.equal(t.state.testVisualConfirmations.size,0);
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,0);
  assert.equal(t.baselineGate(gated.conditions[1]).numericPassed,true);
  assert.equal(t.baselineGate(gated.conditions[1]).visuallyConfirmed,false);
  const allHigh={target:testAsset,conditions:[condition('png',.9),condition('jpeg95',.91)]};
  t.state.testData=allHigh;t.renderTestResults();
  assert.equal(t.baselineGate(allHigh.conditions[0]).accepted,false);
  t.setVisualBaseline(allHigh.conditions[0],true);t.renderTestResults();
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,1);
  assert.equal(t.baselineGate(allHigh.conditions[1]).accepted,false);
  t.setVisualBaseline(allHigh.conditions[1],true);t.renderTestResults();
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,2);

  // Null, NaN, missing/failed detection and invalid cutoffs all stay inconclusive.
  for(const score of [null,NaN,Infinity]){
    const invalid={target:testAsset,conditions:[condition('png',score)]};
    t.state.testData=invalid;t.renderTestResults();
    assert.equal(t.baselineGate(invalid.conditions[0]).numericPassed,false);
    assert.match(renderedText(elements.get('test-results')),/Inconclusive — weak baseline/);
    assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,0);
  }
  const undetected={target:testAsset,conditions:[condition('png',.95)]};
  undetected.conditions[0].rows[0].swap_detected=false;
  t.state.testData=undetected;t.renderTestResults();
  assert.equal(t.baselineGate(undetected.conditions[0]).numericPassed,false);
  elements.get('test-baseline-cutoff').value='1.01';t.changeBaselineCutoff();
  assert.equal(t.state.testBaselineCutoff,null);
  t.state.testData=allHigh;t.renderTestResults();
  assert.equal(t.baselineGate(allHigh.conditions[0]).numericPassed,false);

  // Cached results are gated on open without rerunning or changing the server schema.
  elements.get('test-baseline-cutoff').value='0.80';t.changeBaselineCutoff();
  await t.openTestDialog();
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,0);
  assert.match(renderedText(elements.get('test-results')),/confirm the baseline appearance/);
  t.setVisualBaseline(allHigh.conditions[0],true);t.renderTestResults();
  context.fetch=async()=>({ok:true,json:async()=>({id:'new-quality-job',kind:'test',result_id:result.id,status:'done',test:allHigh})});
  await t.watchTestJob('new-quality-job',result.id);
  assert.equal(t.state.testVisualConfirmations.size,0,'A new job must require new visual confirmation');
  assert.equal(descendants(elements.get('test-results')).filter(node=>node.className==='test-delta').length,0);
  t.setVisualBaseline(allHigh.conditions[0],true);
  await t.useSource(current);
  assert.equal(t.state.testVisualConfirmations.size,0,'A new photo must clear visual confirmation');
  assert.equal(t.state.testData,null);
  console.log('PASS: existing editor/test regressions; unchanged POST payload with client-only cutoff; poor/high-unconfirmed/confirmed baseline gates; independent PNG/JPEG checks; raw scores and original thumbnail; null/NaN/detection rejection; invalid cutoff; cached-result recheck; cutoff/new-job/new-photo confirmation resets.');
})().catch(error=>{console.error(error);process.exitCode=1;});
