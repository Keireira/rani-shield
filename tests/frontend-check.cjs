const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.join(__dirname, '..', 'web');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const elements = new Map();
function element(id = '', tag = '') {
  let storedId=id;
  const node={ tag, value: '', disabled: false, hidden: false, checked: false, open: false, children: [], listeners: {}, textContent: '', style: {setProperty(){}},
    classList: {toggle(){},add(){},remove(){}}, dataset: {}, attributes:{},
    setAttribute(name,value){this.attributes[name]=value;}, addEventListener(name,callback){(this.listeners[name]||=[]).push(callback);},
    querySelectorAll(selector){const tags=selector.split(',').map(value=>value.trim());return descendants(this).slice(1).filter(child=>tags.includes(child.tag));}, closest(){return null;},
    replaceChildren(...nodes){this.children=[...nodes];}, appendChild(node){this.children.push(node);return node;}, showModal(){this.open=true;}, close(){this.open=false;}, focus(){this.focused=true;},
    clientWidth: 800, clientHeight: 500,
  };
  Object.defineProperty(node,'id',{get:()=>storedId,set(value){storedId=value;elements.set(value,node);}});
  return node;
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
      if(tag !== 'canvas') return element('',tag);
      const canvas={width:0,height:0};
      canvas.getContext=()=>({ createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}), putImageData:data=>{canvas.pixels=data.data;} });
      return canvas;
    },
  },
};
context.globalThis=context;
const source=fs.readFileSync(path.join(root,'app.js'),'utf8').replace(/\ninitialize\(\);\s*$/,'');
vm.runInNewContext(source+'\nglobalThis.test={state,settings,normalizedSettings,applySettings,diffImage,useSource,useResult,processImage,openSettingsDialog,closeSettingsDialog,saveSettingsDialog,updateCards,restoreSelection,watchJob,busy,syncMethodControls,readPresets,savePreset,selectPreset,downloadStage1,download,testEligible,startProtectionTest,watchTestJob,openTestDialog,closeTestDialog,renderTestResults,invalidateTest,resumeActiveJob,baselineGate,setVisualBaseline,changeBaselineCutoff,changeTestBackend,addStage,removeStage,moveStage,changeStageType,setStageParameter,renderPipeline,readDialogSettings,downloadIntermediate};', context);
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
  const recipe=(stages,max_size=1280)=>({mode:'custom',max_size,stages});
  const stage=(type,params={})=>({type,params});
  const plain=value=>JSON.parse(JSON.stringify(value));
  assert.equal(t.normalizedSettings().mode,'custom');
  assert.deepEqual(plain(t.normalizedSettings().stages),[
    stage('phantomseal',{iterations:1000}),stage('faceshield',{iterations:30,noise_clamp:12,step_size:1,analysis_size:512})]);
  assert.deepEqual(plain(t.normalizedSettings({},true).stages),[stage('faceshield',{iterations:30,noise_clamp:12,step_size:1,analysis_size:512})],'Mode-less old presets stay FaceShield-only');
  const migrated=t.normalizedSettings({mode:'phantomseal_faceshield',phantom_iterations:1100,iterations:44,noise_clamp:9,step_size:.5,analysis_size:640},true);
  assert.deepEqual(plain(migrated.stages),[stage('phantomseal',{iterations:1100}),stage('faceshield',{iterations:44,noise_clamp:9,step_size:.5,analysis_size:640})]);
  assert.deepEqual(plain(t.normalizedSettings({mode:'nullswap'},true).stages),[stage('nullswap',{strength:1,seed:0})]);
  assert.deepEqual(plain(t.normalizedSettings({mode:'nullswap',nullswap_strength:2.75,nullswap_seed:42},true).stages),[stage('nullswap',{strength:2.75,seed:42})]);
  assert.throws(()=>t.normalizedSettings({mode:'invalid'}));
  assert.throws(()=>t.normalizedSettings({mode:'custom'}));
  assert.throws(()=>t.normalizedSettings({mode:['custom'],stages:[stage('nullswap')]}));
  assert.throws(()=>t.normalizedSettings(recipe([stage(['nullswap'])])));
  assert.throws(()=>t.normalizedSettings(recipe([stage('nullswap',null)])));
  assert.throws(()=>t.normalizedSettings(recipe([{type:'nullswap',unknown:true}])));
  for(const stages of [[],Array.from({length:9},()=>stage('nullswap')),{},null,[stage('invalid')],[stage('nullswap',{oops:1})]]) assert.throws(()=>t.normalizedSettings(recipe(stages)));
  for(const value of [-.25,4.01,NaN,Infinity,'1',true,null]) assert.throws(()=>t.normalizedSettings(recipe([stage('nullswap',{strength:value})])));
  for(const value of [-1,4294967296,.5,NaN,Infinity,'0',false]) assert.throws(()=>t.normalizedSettings(recipe([stage('nullswap',{seed:value})])));
  for(const value of [0,.25,2.75,4]) assert.equal(t.normalizedSettings(recipe([stage('nullswap',{strength:value})])).stages[0].params.strength,value);
  for(const value of [0,1,4294967295]) assert.equal(t.normalizedSettings(recipe([stage('nullswap',{seed:value})])).stages[0].params.seed,value);
  for(const value of [9,2001,10.5,NaN]) assert.throws(()=>t.normalizedSettings(recipe([stage('phantomseal',{iterations:value})])));
  for(const value of [0,201,1.5,NaN]) assert.throws(()=>t.normalizedSettings(recipe([stage('faceshield',{iterations:value})])));
  for(const value of [1,200]) assert.equal(t.normalizedSettings(recipe([stage('faceshield',{iterations:value})])).stages[0].params.iterations,value);
  for(const value of [63,129,513,1281]) assert.throws(()=>t.normalizedSettings(recipe([stage('faceshield',{analysis_size:value})])));
  for(const value of [64,512,1280]) assert.equal(t.normalizedSettings(recipe([stage('faceshield',{analysis_size:value})])).stages[0].params.analysis_size,value);
  assert.throws(()=>t.normalizedSettings(recipe([stage('faceshield',{noise_clamp:1000})])));
  assert.throws(()=>t.normalizedSettings(recipe([stage('faceshield',{step_size:.1})])));
  assert.throws(()=>t.normalizedSettings(recipe([stage('nullswap')],127)));
  assert.throws(()=>t.normalizedSettings(recipe([stage('nullswap')],1281)));
  assert.throws(()=>t.normalizedSettings(recipe([stage('nullswap')],NaN)));
  t.state.initializing=true;t.applySettings({});t.updateCards();
  assert.equal(elements.get('file-input').disabled,true,'Startup locks picker');
  assert.equal(elements.get('stage-0-iterations').disabled,true,'Startup locks stage parameters');
  t.state.initializing=false;t.applySettings({});
  assert.equal(elements.get('pipeline-stages').children.length,2);
  assert.equal(elements.get('stage-0-up').disabled,true);
  assert.equal(elements.get('stage-1-down').disabled,true);
  assert.equal(elements.get('stage-0-iterations').value,1000);
  assert.equal(elements.get('stage-1-analysis_size').value,512);
  t.state.ready=true;t.state.uploadedSource={id:'available'};
  t.state.engines={phantomseal:{ready:false,message:'PhantomSeal weights unavailable.'},faceshield:{ready:true,message:'Ready'},nullswap:{ready:true}};
  t.updateCards();assert.equal(elements.get('process-button').disabled,true);
  assert.match(elements.get('method-readiness').textContent,/weights unavailable/);
  t.applySettings({mode:'faceshield'});assert.equal(elements.get('process-button').disabled,false);
  t.addStage('phantomseal');assert.equal(elements.get('process-button').disabled,true,'Every stage must be ready');
  t.removeStage(1);assert.equal(elements.get('process-button').disabled,false);
  t.state.engines={phantomseal:{ready:true,message:'PhantomSeal ensemble: ready'},faceshield:{ready:true},nullswap:{ready:true}};
  t.applySettings({});assert.equal(elements.get('method-readiness').textContent,'PhantomSeal ensemble: ready · FaceShield: ready');
  // Repeated filters keep separate parameters through reordering and removal.
  t.addStage('nullswap');t.setStageParameter(2,'strength',2.75);t.setStageParameter(2,'seed',42);
  t.addStage('nullswap');t.setStageParameter(3,'strength',.25);t.setStageParameter(3,'seed',4294967295);
  assert.deepEqual(plain(t.settings().stages.map(item=>item.type)),['phantomseal','faceshield','nullswap','nullswap']);
  t.moveStage(3,-1);assert.equal(t.settings().stages[2].params.seed,4294967295);assert.equal(t.settings().stages[3].params.seed,42);
  assert.equal(elements.get('stage-2-seed').value,4294967295);
  t.moveStage(0,-1);t.moveStage(3,1);assert.equal(t.settings().stages.length,4,'Out-of-range moves are ignored');
  t.removeStage(1);assert.deepEqual(plain(t.settings().stages.map(item=>item.type)),['phantomseal','nullswap','nullswap']);
  t.changeStageType(0,'faceshield');assert.equal(t.settings().stages[0].params.iterations,30);assert.equal(t.settings().stages[1].params.seed,4294967295);
  for(let i=0;i<6;i++)t.addStage('nullswap');
  assert.equal(t.settings().stages.length,8);assert.equal(elements.get('add-stage').disabled,true);assert.equal(elements.get('add-stage-type').disabled,true);
  t.addStage('faceshield');assert.equal(t.settings().stages.length,8);
  for(let i=0;i<8;i++)t.removeStage(0);
  assert.equal(t.settings().stages.length,1);assert.equal(elements.get('stage-0-remove').disabled,true,'Last stage cannot be removed');
  t.applySettings({mode:'nullswap',nullswap_strength:2.75,nullswap_seed:4294967295});
  assert.match(renderedText(elements.get('pipeline-stages')),/22\/255 before PNG rounding/);
  // Exercise actual input callbacks, including blank seed and validation lock.
  const seedInput=elements.get('stage-0-seed');seedInput.value='';seedInput.listeners.input[0]();
  assert.throws(()=>t.settings(),/Seed/);assert.equal(elements.get('process-button').disabled,true);
  assert.match(elements.get('pipeline-error').textContent,/Seed/);
  seedInput.value='42';seedInput.listeners.input[0]();assert.equal(t.settings().stages[0].params.seed,42);
  assert.equal(elements.get('process-button').disabled,false);
  t.busy(true);const lockedRecipe=JSON.stringify(t.state.pipeline);
  for(const id of ['add-stage','add-stage-type','stage-0-type','stage-0-strength','stage-0-seed','max-size']) assert.equal(elements.get(id).disabled,true,`Busy lock ${id}`);
  t.addStage('faceshield');t.removeStage(0);t.changeStageType(0,'faceshield');t.setStageParameter(0,'seed',123);
  assert.equal(JSON.stringify(t.state.pipeline),lockedRecipe,'Busy operations must not mutate the recipe');
  t.busy(false);assert.equal(elements.get('stage-0-seed').disabled,false);
  t.openSettingsDialog();assert.equal(elements.get('dialog-stage-0-seed').value,42);
  t.addStage('faceshield','dialog-');t.moveStage(1,-1,'dialog-');t.setStageParameter(1,'seed',123,'dialog-');
  assert.equal(t.settings().stages.length,1,'Draft changes do not touch the live recipe');
  t.closeSettingsDialog();assert.equal(t.settings().stages[0].params.seed,42);
  t.openSettingsDialog();t.setStageParameter(0,'seed',1.5,'dialog-');t.saveSettingsDialog();
  assert.equal(elements.get('settings-dialog').open,true);assert.match(elements.get('settings-message').textContent,/Seed.*whole number/);
  assert.equal(t.settings().stages[0].params.seed,42);
  t.setStageParameter(0,'seed',123,'dialog-');t.addStage('faceshield','dialog-');t.moveStage(1,-1,'dialog-');
  t.setStageParameter(0,'analysis_size',640,'dialog-');elements.get('dialog-max-size').value='1024';
  t.saveSettingsDialog();assert.deepEqual(plain(t.settings().stages.map(item=>item.type)),['faceshield','nullswap']);
  assert.equal(t.settings().stages[0].params.analysis_size,640);assert.equal(t.settings().stages[1].params.seed,123);assert.equal(t.settings().max_size,1024);
  store.set('faceshield.presets.v1',JSON.stringify({version:1,presets:[{id:'legacy',name:'Old FaceShield',params:{iterations:44}},{id:'old-ns',name:'Old NullSwap',params:{mode:'nullswap',nullswap_strength:.25,nullswap_seed:8}}]}));
  t.readPresets();assert.equal(t.state.presets[0].params.stages[0].type,'faceshield');assert.equal(t.state.presets[0].params.stages[0].params.iterations,44);
  assert.equal(t.state.presets[1].params.stages[0].params.seed,8);
  t.openSettingsDialog();t.addStage('nullswap','dialog-');t.setStageParameter(2,'seed',999,'dialog-');
  elements.get('preset-name').value='Repeated filters';t.savePreset();t.readPresets();
  const repeated=t.state.presets.find(item=>item.name==='Repeated filters');
  assert.equal(repeated.params.stages.length,3);assert.equal(repeated.params.stages[1].params.seed,123);assert.equal(repeated.params.stages[2].params.seed,999);
  t.changeStageType(0,'phantomseal','dialog-');elements.get('preset-select').value=repeated.id;t.selectPreset();
  assert.equal(t.state.settingsDraft.stages[0].type,'faceshield');t.saveSettingsDialog();
  assert.deepEqual(plain(t.settings()),plain(repeated.params));
  assert.deepEqual(JSON.parse(store.get('faceshield.settings.v1')),plain(repeated.params));
  t.applySettings({});assert.equal(t.settings().stages.length,2);assert.equal(t.settings().stages[0].params.iterations,1000);assert.equal(t.settings().max_size,1280,'Reset restores complete default recipe');
  const combined={params:migrated};t.applySettings(combined.params);

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
  assert.equal(submitted.params.mode,'custom');
  assert.deepEqual(submitted.params,plain(migrated));
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
  assert.deepEqual(testPayload,{result_id:'protected',target_key:'wlh',backend:'simswap',jpeg95:true});
  assert.equal(t.state.busy,true); assert.equal(t.state.busyKind,'test');
  assert.equal(elements.get('process-label').textContent,'Testing…');
  for(const id of ['process-button','upload-button','file-input','export-button','export-stage1-button','run-test-button','test-target','test-jpeg95'])assert.equal(elements.get(id).disabled,true,`Test must lock ${id}`);
  assert.equal(elements.get('test-progress').value,55);
  assert.equal(elements.get('test-status').textContent,'Running PNG swaps');
  t.closeTestDialog();assert.equal(elements.get('test-dialog').open,false);
  assert.equal(t.state.busy,true,'Closing modal must not cancel or unlock the active test');
  await t.openTestDialog();assert.equal(elements.get('test-dialog').open,true);
  assert.equal(testPosts,1,'Reopen must not start another GPU test');
  t.applySettings({mode:'faceshield'});
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
  // Backend changes clear previous results/visual approvals and check only selected weights.
  t.setVisualBaseline(allHigh.conditions[0],true);
  t.state.testing={ready:true,backends:{simswap:{ready:true},diffswap:{ready:false,message:'Missing diffusion weights'},uniface:{ready:true}}};
  elements.get('test-backend').value='diffswap';t.changeTestBackend();
  assert.equal(t.state.testVisualConfirmations.size,0);
  assert.equal(t.state.testData,null);
  assert.equal(elements.get('run-test-button').disabled,true);
  assert.match(elements.get('test-readiness').textContent,/Missing diffusion weights/);
  elements.get('test-backend').value='uniface';t.changeTestBackend();
  assert.equal(elements.get('run-test-button').disabled,false);
  assert.equal(elements.get('run-test-button').textContent,'Run UniFace test');
  t.state.testData={...allHigh,backend:'diffswap',backend_label:'DiffSwap'};t.renderTestResults();
  assert.match(renderedText(elements.get('test-results')),/DiffSwap comparison/,'Saved results keep their actual backend label');
  t.state.engines={nullswap:{ready:true,message:'Reconstruction ready'},phantomseal:{ready:false},faceshield:{ready:false}};
  t.applySettings({mode:'nullswap',nullswap_strength:2.25,nullswap_seed:4294967295});
  assert.equal(elements.get('stage-0-strength').disabled,false);assert.equal(elements.get('max-size').disabled,false);
  assert.match(renderedText(elements.get('pipeline-stages')),/reconstruction/);
  t.addStage('nullswap');t.setStageParameter(1,'seed',77);t.setStageParameter(1,'strength',.5);
  t.state.uploadedSource=input;t.state.ready=true;
  let nullswapPayload;
  const outputs=[{index:1,type:'nullswap',asset:intermediate},{index:2,type:'nullswap',asset:result}];
  context.fetch=async(url,options)=>{
    if(url==='/api/process'){
      nullswapPayload=JSON.parse(options.body);
      assert.equal(elements.get('stage-0-strength').disabled,true);
      assert.equal(elements.get('stage-1-seed').disabled,true);
      return {ok:true,json:async()=>({id:'nullswap-job'})};
    }
    assert.equal(url,'/api/job/nullswap-job');
    return {ok:true,json:async()=>({id:'nullswap-job',source_id:input.id,status:'done',progress:100,source:baseline,result,params:nullswapPayload.params,intermediate_assets:outputs})};
  };
  await t.processImage();
  assert.equal(nullswapPayload.params.mode,'custom');
  assert.deepEqual(nullswapPayload.params.stages,[stage('nullswap',{strength:2.25,seed:4294967295}),stage('nullswap',{strength:.5,seed:77})]);
  assert.equal(elements.get('stage-0-strength').disabled,false);assert.equal(elements.get('stage-1-seed').disabled,false);
  assert.match(elements.get('result-summary').textContent,/Custom pipeline/);
  assert.equal(elements.get('stage-output-controls').hidden,false);
  assert.equal(elements.get('stage-output-select').children.length,2);
  elements.get('stage-output-select').value='1';t.downloadIntermediate();assert.equal(downloads.at(-1).url,intermediate.url);
  assert.equal(t.state.source.id,baseline.id);assert.equal(t.state.result.id,result.id,'Intermediate export must not replace viewer result');
  elements.get('stage-output-select').value='2';t.downloadIntermediate();assert.equal(downloads.at(-1).url,result.url);
  assert.equal(JSON.parse(store.get('faceshield.selection.v1')).intermediateAssets.length,2);
  t.state.assetIndex.set(intermediate.id,intermediate);t.state.assetIndex.set(result.id,result);
  t.state.intermediateAssets=[];await t.restoreSelection();
  assert.equal(t.state.intermediateAssets.length,2,'Reload restores verified stage outputs');
  assert.equal(elements.get('stage-output-controls').hidden,false);
  assert.equal(t.state.result.id,result.id);
  t.busy(true);assert.equal(elements.get('export-stage-button').disabled,true);assert.equal(elements.get('stage-output-select').disabled,true);t.busy(false);
  await t.useSource(current);
  assert.equal(t.state.testVisualConfirmations.size,0,'A new photo must clear visual confirmation');
  assert.equal(t.state.testData,null);
  assert.equal(t.state.intermediateAssets.length,0);assert.equal(elements.get('stage-output-controls').hidden,true);
  console.log('PASS: existing editor/test regressions; backend dispatch/readiness/labels/confirmation reset; ordered pipeline add/remove/reorder/repeats; per-stage validation and readiness; full recipe draft/presets/legacy migration; stage exports; client-only cutoff; poor/high-unconfirmed/confirmed baseline gates; independent PNG/JPEG checks; raw scores and original thumbnail; null/NaN/detection rejection; invalid cutoff; cached-result recheck; cutoff/new-job/new-photo confirmation resets.');
})().catch(error=>{console.error(error);process.exitCode=1;});
