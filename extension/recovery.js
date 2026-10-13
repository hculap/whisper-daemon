// Keep a failed transport tail useful even if its original session has ended.
// The WAV begins at the first retained packet; its filename carries that offset.
function recoveryWav(packets) {
  if (!packets.length) throw new Error('No buffered audio');
  const first = Number(new DataView(packets[0]).getBigUint64(8, true));
  const last = packets.at(-1);
  const end = Number(new DataView(last).getBigUint64(8, true)) + (last.byteLength - 16) / 4;
  const header = new ArrayBuffer(44), view = new DataView(header);
  const label = (at, value) => { for (let i=0;i<value.length;i++) view.setUint8(at+i,value.charCodeAt(i)); };
  label(0,'RIFF');view.setUint32(4,36+(end-first)*2,true);label(8,'WAVE');label(12,'fmt ');
  view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);
  view.setUint32(24,16000,true);view.setUint32(28,32000,true);view.setUint16(32,2,true);view.setUint16(34,16,true);
  label(36,'data');view.setUint32(40,(end-first)*2,true);
  const pieces=[header];let cursor=first;
  const silence=new Uint8Array(32000);
  for(const packet of packets){
    const input=new DataView(packet),position=Number(input.getBigUint64(8,true));
    let missing=(position-cursor)*2;
    while(missing>0){const size=Math.min(missing,silence.length);pieces.push(silence.subarray(0,size));missing-=size;}
    const output=new ArrayBuffer((packet.byteLength-16)/2),pcm=new DataView(output);
    for(let i=16;i<packet.byteLength;i+=4)pcm.setInt16((i-16)/2,Math.round(Math.max(-1,Math.min(1,input.getFloat32(i,true)))*32767),true);
    pieces.push(output);cursor=position+(packet.byteLength-16)/4;
  }
  return {blob:new Blob(pieces,{type:'audio/wav'}),offset:first/16000};
}
if(typeof module!=='undefined')module.exports={recoveryWav};
