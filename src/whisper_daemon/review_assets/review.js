'use strict';
const $ = id => document.getElementById(id);
let data, assignments = {};
const time = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2,'0')}`;
function element(tag, text, cls) { const node=document.createElement(tag); if(text!==undefined) node.textContent=text; if(cls) node.className=cls; return node; }
function play(start) { $('audio').currentTime=start; $('audio').play().catch(() => {$('status').textContent='Wybierz dostępne źródło audio.';}); }
function renderTranscript() {
  const query=$('search').value.toLocaleLowerCase();
  $('segments').replaceChildren();
  let shown=0;
  for(const seg of data.transcript.segments) {
    const speaker=data.transcript.speakers.find(s=>s.id===(assignments[seg.id]??seg.speaker));
    if(query && !(seg.text+' '+speaker?.label).toLocaleLowerCase().includes(query)) continue;
    shown++;
    const row=element('article',undefined,'segment'); row.dataset.start=seg.start; row.dataset.end=seg.end;
    const seek=element('button',time(seg.start),'time'); seek.title='Odtwórz od tej wypowiedzi'; seek.onclick=()=>play(seg.start); row.append(seek);
    const body=element('div'); const select=element('select'); select.setAttribute('aria-label','Mówca wypowiedzi');
    for(const s of data.transcript.speakers) { const option=element('option',s.label);option.value=s.id;select.append(option); }
    select.value=assignments[seg.id]??seg.speaker;
    select.onchange=()=>{assignments[seg.id]=Number(select.value);dirty();};
    body.append(select,element('p',seg.text));row.append(body);$('segments').append(row);
  }
  $('count').textContent=`${shown} / ${data.transcript.segments.length}`;
}
function dirty() {$('status').textContent='Niezapisane zmiany';}
function renderTasks() {
  $('tasks').replaceChildren();
  data.notes.tasks.forEach((task,index)=>{
    const card=element('article',undefined,'task');
    const seek=element('button',`Źródło ${time(task.start)}`,'secondary'); seek.onclick=()=>play(task.start);
    const text=element('textarea');text.rows=2;text.value=task.text;text.setAttribute('aria-label','Treść zadania');text.oninput=()=>{task.text=text.value;dirty();};
    const fields=element('div',undefined,'fields');
    for(const [key,label] of [['owner','Osoba'],['due','Termin']]) {const input=element('input');input.placeholder=label;input.setAttribute('aria-label',label);input.value=task[key];input.oninput=()=>{task[key]=input.value;dirty();};fields.append(input);}
    const checks=element('div',undefined,'checks');
    for(const [key,label] of [['confirmed','Sprawdzone'],['done','Wykonane']]) {const wrap=element('label',label);const check=element('input');check.type='checkbox';check.checked=task[key];check.onchange=()=>{task[key]=check.checked;dirty();};wrap.prepend(check);checks.append(wrap);}
    const remove=element('button','Usuń','secondary');remove.onclick=()=>{data.notes.tasks.splice(index,1);renderTasks();dirty();};
    card.append(seek,text,fields,checks,remove);$('tasks').append(card);
  });
}
function renderSpeakers() {
  $('speakers').replaceChildren();
  for(const speaker of data.transcript.speakers) {const label=element('label','Nazwa mówcy / źródła');const input=element('input');input.value=speaker.label;input.maxLength=80;input.dataset.speaker=speaker.id;input.oninput=()=>{speaker.label=input.value;dirty();};input.onchange=renderTranscript;label.append(input);$('speakers').append(label);}
}
function render() {
  $('meta').textContent=data.title;
  $('source').replaceChildren();
  for(const file of data.audio) {const option=element('option',{'recording.wav':'Całość','mic.wav':'Mikrofon','browser.wav':'Uczestnicy'}[file]);option.value=file;$('source').append(option);}
  if(data.audio.length) $('audio').src=data.audio[0];
  else $('status').textContent='Brak audio. Dla przerwanego nagrania uruchom odzyskiwanie.';
  const gaps=data.session.gaps||[];
  $('gaps').textContent=gaps.length ? `Oznaczone przerwy: ${gaps.map(g=>`${g.source} ${time(g.start)}–${time(g.end)}`).join(', ')}` : 'Brak wykrytych luk w źródłach.';
  renderSpeakers();
  $('summary').value=data.notes.summary;
  renderTranscript();renderTasks();
}
$('search').oninput=renderTranscript;
$('summary').oninput=()=>{data.notes.summary=$('summary').value;dirty();};
$('source').onchange=()=>{const current=$('audio').currentTime;$('audio').src=$('source').value;$('audio').addEventListener('loadedmetadata',()=>{$('audio').currentTime=current;},{once:true});};
$('audio').ontimeupdate=()=>{for(const row of document.querySelectorAll('.segment')) row.classList.toggle('active',$('audio').currentTime>=+row.dataset.start&&$('audio').currentTime<+row.dataset.end);};
$('addTask').onclick=()=>{data.notes.tasks.push({text:'',owner:'',due:'',start:$('audio').currentTime||0,done:false,confirmed:false});renderTasks();dirty();};
$('save').onclick=async()=>{
  $('save').disabled=true;
  try {
    const names=Object.fromEntries(data.transcript.speakers.map(s=>[s.id,s.label]));
    const response=await fetch('save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names,assignments,summary:$('summary').value,tasks:data.notes.tasks})});
    const saved=await response.json();if(!response.ok)throw new Error(saved.error||'Nie udało się zapisać');
    data=saved;assignments={};$('status').textContent='Zapisano nazwy, notatki i zadania. Eksporty transkrypcji zaktualizowane.';
  } catch(error) {$('status').textContent=error.message;} finally {$('save').disabled=false;}
};
fetch('data').then(r=>{if(!r.ok)throw new Error('Brak dostępu do spotkania');return r.json();}).then(value=>{data=value;render();}).catch(error=>{$('status').textContent=error.message;});

$('addSpeaker').onclick=()=>{const id=Math.max(-1,...data.transcript.speakers.map(s=>s.id))+1;data.transcript.speakers.push({id,label:'Nowy mówca'});renderSpeakers();renderTranscript();dirty();};
