const test=require('node:test'), assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm'), path=require('node:path');
function fixture(){
  class Node{
    constructor(){this.children=[];this.value='';this.dataset={};this.currentTime=0;}
    append(...nodes){this.children.push(...nodes);} prepend(n){this.children.unshift(n);}
    replaceChildren(...nodes){this.children=nodes;}setAttribute(){}
  }
  const ids={}; const doc={getElementById:id=>ids[id]??=new Node(),createElement:()=>new Node(),querySelectorAll:()=>[]};
  const initial={title:'test',audio:[],session:{},transcript:{speakers:[{id:0,label:'original'}],segments:[]},notes:{summary:'',tasks:[{text:'original',owner:'',due:'',start:0}]}};
  const posts=[],presence=[],timers=[];let pending;
  const ctx=vm.createContext({document:doc,setInterval:(fn,ms)=>timers.push({fn,ms}),fetch:async(url,opts)=>{
    if(url==='data')return{ok:true,json:async()=>structuredClone(initial)};
    if(url==='presence'){presence.push(JSON.parse(opts.body).dirty);return{ok:true,json:async()=>({ok:true})};}
    const body=JSON.parse(opts.body);posts.push(body);
    await new Promise(resolve=>pending=resolve);
    const saved=structuredClone(initial); saved.transcript.speakers[0].label=body.names[0];saved.notes.tasks=body.tasks;
    return{ok:true,json:async()=>saved};
  }});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../../src/whisper_daemon/review_assets/review.js'),'utf8'),ctx);
  return{ids,posts,presence,timers,finish:()=>pending(),ready:()=>new Promise(setImmediate),
    edit(name,text){const n=ids.speakers.children[0].children[0];n.value=name;n.oninput();const t=ids.tasks.children[0].children[1];t.value=text;t.oninput();}};
}
test('second save persists newly edited speaker and task',async()=>{
  const f=fixture();await f.ready();f.edit('first','first');let save=f.ids.save.onclick();await f.ready();f.finish();await save;
  f.edit('second','second');save=f.ids.save.onclick();await f.ready();f.finish();await save;
  assert.equal(f.posts[1].names[0],'second');assert.equal(f.posts[1].tasks[0].text,'second');
});
test('edits made while saving remain pending and survive the next save',async()=>{
  const f=fixture();await f.ready();f.edit('first','first');let save=f.ids.save.onclick();await f.ready();f.edit('second','second');f.finish();await save;
  assert.match(f.ids.status.textContent,/Niezapisane/);
  save=f.ids.save.onclick();await f.ready();f.finish();await save;
  assert.equal(f.posts[1].names[0],'second');assert.equal(f.posts[1].tasks[0].text,'second');
});
test('unsaved edits are reported to the daemon until a save covers them',async()=>{
  const f=fixture();await f.ready();
  f.edit('first','first');
  assert.deepEqual(f.presence,[true]);  // reported once, on the first edit
  const heartbeat=f.timers.find(t=>t.ms<=60000);assert.ok(heartbeat);
  let save=f.ids.save.onclick();await f.ready();f.edit('second','second');f.finish();await save;
  assert.deepEqual(f.presence,[true,true]);  // edits made during the save are still unsaved
  heartbeat.fn();assert.deepEqual(f.presence,[true,true,true]);
  save=f.ids.save.onclick();await f.ready();f.finish();await save;
  assert.deepEqual(f.presence,[true,true,true,false]);
  heartbeat.fn();assert.equal(f.presence.length,4);  // a clean page sends no heartbeat
});
