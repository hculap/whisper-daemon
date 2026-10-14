const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
function fixture(){
  let receive; const messages=[],tabs=[];
  const noop=()=>{};
  const chrome={storage:{local:{setAccessLevel:noop,get:async()=>({pairingSecret:'stored-key'})}},
    runtime:{id:'ext',getURL:page=>`chrome-extension://ext/${page}`,
      onMessage:{addListener(fn){receive=fn;}},getContexts:async()=>[{}],sendMessage:async msg=>messages.push(msg)},
    action:{setBadgeText:noop,setBadgeBackgroundColor:noop},
    notifications:{create:noop,onClicked:{addListener:noop}},
    tabs:{onRemoved:{addListener:noop},query:async()=>[{id:1},{id:2}],sendMessage:async(id,msg)=>tabs.push({id,msg})}};
  const ctx=vm.createContext({chrome,console});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../background.js'),'utf8'),ctx);
  const deliver=async(msg,sender)=>{
    let response;receive(msg,sender,r=>response=r);
    await new Promise(setImmediate);return response;
  };
  return {messages,tabs,receive:(msg,id)=>deliver(msg,id?{tab:{id}}:{}),deliver};
}
test('settings reads from two tabs never claim recording ownership',async()=>{
  const f=fixture();
  await f.receive({type:'WD_GET_SETTINGS'},1);
  await f.receive({type:'WD_GET_SETTINGS'},2);
  assert.equal(f.messages.filter(m=>m.type==='WD_GET_SETTINGS').length,2);
  await f.receive({type:'SETTINGS',settings:{capture_mic:true}});
  assert.deepEqual(f.tabs.filter(x=>x.msg.type==='WD_SETTINGS').map(x=>x.id),[1,2]);
  assert.equal((await f.receive({type:'GET_STATE'})).meetTabId,null);
});
test('terminal idle releases ownership so another tab can record',async()=>{
  const f=fixture();
  await f.receive({type:'WD_START'},1);
  await f.receive({type:'CAPTURE_STARTED'});
  await f.receive({type:'STATUS',state:'idle'});
  assert.equal((await f.receive({type:'WD_START'},2)).ok,true);
});
test('a native recording remains controllable without browser tab ownership',async()=>{
  const f=fixture();
  await f.receive({type:'STATUS',state:'recording',sessionId:null});
  await f.receive({type:'WD_PAUSE_MEETING'},2);
  await f.receive({type:'WD_STOP_MEETING'},2);
  assert.deepEqual(f.messages.map(m=>m.type),['WD_PAUSE_MEETING','WD_STOP_MEETING']);
});
test('idle statuses keep the joined call, so leaving it still tears the socket down',async()=>{
  const f=fixture();
  await f.receive({type:'MEET_JOINED',title:'t',url:'u'},1);
  await f.receive({type:'STATUS',state:'idle',snapshot:true});
  assert.deepEqual({...await f.receive({type:'GET_STATE'})},{state:'detected',meetTabId:1,meetTitle:'t',meetUrl:'u'});
  await f.receive({type:'WD_START'},1);
  await f.receive({type:'STATUS',state:'idle',sessionId:'s'});
  assert.equal((await f.receive({type:'GET_STATE'})).meetTabId,1);
  await f.receive({type:'MEET_LEFT'},1);
  assert.ok(f.messages.some(m=>m.type==='STOP_CAPTURE'));
  assert.ok(f.tabs.some(x=>x.id===1&&x.msg.type==='WD_MEET_LEFT'));
});
test('a saving state never refuses a start and is released when the daemon goes away',async()=>{
  const f=fixture();
  await f.receive({type:'MEET_JOINED'},1);
  await f.receive({type:'STATUS',state:'transcribing',sessionId:'s'});
  await f.receive({type:'CONTROL_CLOSED'});
  assert.equal((await f.receive({type:'GET_STATE'})).state,'detected');
  await f.receive({type:'STATUS',state:'transcribing',sessionId:'s'});
  assert.equal((await f.receive({type:'WD_START'},1)).ok,true);
  assert.ok(f.messages.some(m=>m.type==='WD_START'));
});
test('only the offscreen document can read the pairing key',async()=>{
  const f=fixture();
  const ask={type:'GET_PAIRING_SECRET'};
  assert.equal((await f.deliver(ask,{id:'ext',url:'chrome-extension://ext/offscreen.html'})).secret,'stored-key');
  assert.equal((await f.deliver(ask,{id:'ext',url:'chrome-extension://ext/popup.html'})).secret,null);
  assert.equal((await f.deliver(ask,{id:'ext',url:'chrome-extension://ext/offscreen.html',tab:{id:1}})).secret,null);
  assert.equal((await f.deliver(ask,{id:'other',url:'chrome-extension://ext/offscreen.html'})).secret,null);
});
