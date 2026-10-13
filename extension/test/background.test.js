const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
function fixture(){
  let receive; const messages=[],tabs=[];
  const noop=()=>{};
  const chrome={storage:{local:{setAccessLevel:noop}},
    runtime:{onMessage:{addListener(fn){receive=fn;}},getContexts:async()=>[{}],sendMessage:async msg=>messages.push(msg)},
    action:{setBadgeText:noop,setBadgeBackgroundColor:noop},
    notifications:{create:noop,onClicked:{addListener:noop}},
    tabs:{onRemoved:{addListener:noop},query:async()=>[{id:1},{id:2}],sendMessage:async(id,msg)=>tabs.push({id,msg})}};
  const ctx=vm.createContext({chrome,console});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../background.js'),'utf8'),ctx);
  return {messages,tabs,receive:async(msg,id)=>{
    let response;receive(msg,id?{tab:{id}}:{},r=>response=r);
    await new Promise(setImmediate);return response;
  }};
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
