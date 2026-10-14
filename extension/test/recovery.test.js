const test=require('node:test'),assert=require('node:assert/strict');
const {recoveryWav}=require('../recovery.js');
function packet(seq,pos,values){const buffer=new ArrayBuffer(16+values.length*4),v=new DataView(buffer);v.setUint32(0,0x31414457,true);v.setUint32(4,seq,true);v.setBigUint64(8,BigInt(pos),true);values.forEach((x,i)=>v.setFloat32(16+i*4,x,true));return buffer;}
test('recovery WAV keeps original offset and missing-packet silence',async()=>{
  const {blob,offset}=recoveryWav([packet(4,16000,[.1,.2]),packet(6,16005,[.5])]);
  const data=new DataView(await blob.arrayBuffer());
  assert.equal(offset,1);assert.equal(data.getUint32(40,true),12);
  assert.deepEqual(Array.from({length:6},(_,i)=>data.getInt16(44+i*2,true)),[3277,6553,0,0,0,16384]);
});
