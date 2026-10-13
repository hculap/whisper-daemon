const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

function fixture() {
  const messages=[],nodes=[],contexts=[];let receive,release;
  class Context {
    constructor(){contexts.push(this);this.audioWorklet={addModule:()=>new Promise(r=>{release=r;})};}
    createGain(){return {gain:{value:0},connect(){},disconnect(){}};}
    resume(){return Promise.resolve();}
    close(){this.closed=true;return Promise.resolve();}
  }
  class Node {constructor(){this.port={};nodes.push(this);}disconnect(){}}
  const window={addEventListener:(name,fn)=>{receive=fn;},postMessage:m=>messages.push(m)};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../rtc-tap.js'),'utf8'),{
    window,location:{origin:'https://meet.google.com'},console,AudioContext:Context,
    AudioWorkletNode:Node,Blob,URL:{createObjectURL:()=>'',revokeObjectURL(){}},Map,
  });
  return {messages,nodes,contexts,release:()=>release(),send:type=>receive({source:window,data:{source:'wd',type}})};
}

test('RTC start completes and forwards positioned PCM',async()=>{
  const f=fixture();f.send('WD_RTC_START');f.release();await new Promise(setImmediate);
  assert.equal(f.nodes.length,1);
  f.nodes[0].port.onmessage({data:{samples:new Float32Array([.1,.2]),position:128}});
  assert.equal(f.messages.at(-1).position,128);
  f.send('WD_RTC_STOP');assert.equal(f.contexts[0].closed,true);
});

test('RTC stop during module loading cannot revive capture',async()=>{
  const f=fixture();f.send('WD_RTC_START');f.send('WD_RTC_STOP');f.release();await new Promise(setImmediate);
  assert.equal(f.nodes.length,0);
});
