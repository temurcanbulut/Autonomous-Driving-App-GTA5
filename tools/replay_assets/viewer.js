const $ = selector => document.querySelector(selector);
const svgNS = 'http://www.w3.org/2000/svg';
const colors = {person:'#ff8493', rider:'#ffa483', bicycle:'#d1ee91', car:'#8dbeff', motorcycle:'#bca5ff'};
let index = 0;
let trace = replay.traces[0];
function node(tag, text) {const el = document.createElement(tag); if (text !== undefined) el.textContent = text; return el;}
function shape(tag, attributes, text) {const el = document.createElementNS(svgNS, tag); for (const [key,value] of Object.entries(attributes)) el.setAttribute(key, String(value)); if (text !== undefined) el.textContent = text; return el;}
function stat(value, label) {const el = node('div'); el.className = 'stat'; el.append(node('strong', value), node('span', label)); return el;}
function selected() {
  trace = replay.traces[Number($('#sequence').value)]; index = 0;
  const s = trace.summary;
  $('#stats').replaceChildren(stat(String(s.frames),'RECORDED FRAMES'), stat(`${s.span_s.toFixed(1)} s`,'FIRST → LAST TIMESTAMP'), stat(s.latency_median_ms === null ? '—' : `${s.latency_median_ms.toFixed(1)} ms`,'MEDIAN RECORDED LATENCY'), stat(String(s.dry_brake_frames),'DRY-RUN BRAKE FRAMES'));
  $('#input-state').textContent = {emitted:'TRACE REPORTS GAME INPUT',recorded_false:'ALL ROWS RECORD: NO GAME INPUT',not_recorded:'GAME INPUT STATUS NOT FULLY RECORDED'}[s.input_evidence];
  $('#frame').max = String(trace.frames.length - 1);
  $('#source').textContent = `${trace.name} · ${trace.bytes} input bytes · SHA256 ${trace.sha256}. Diagnostic target matched: ${s.diagnostic_frames ? `${s.diagnostic_matched}/${s.diagnostic_frames} frames` : 'not recorded'}. Latency scope comes from the trace producer; it is not GTA FPS.`;
  const chart = $('#chart'); chart.replaceChildren();
  const values = trace.frames.map(row => row.latency).filter(value => value !== null);
  const ceiling = Math.max(1, ...values);
  let segment = [];
  const flush = () => {if (segment.length) chart.append(shape('polyline',{points:segment.join(' '),fill:'none',stroke:'#d1ee91','stroke-width':1.5})); segment = [];};
  trace.frames.forEach((row,i) => {
    const x = i / Math.max(1, trace.frames.length - 1) * 960;
    if (row.latency !== null) segment.push(`${x},${75 - row.latency / ceiling * 65}`); else flush();
    if (row.brake === true) chart.append(shape('circle',{cx:x,cy:82,r:3,fill:'#ffba7b'}));
  }); flush();
  chart.append(shape('line',{id:'cursor',x1:0,y1:0,x2:0,y2:90,stroke:'#edf2ed','stroke-width':2}));
  draw();
}
function draw() {
  const row = trace.frames[index], scene = $('#scene');
  $('#frame').value = String(index);
  $('#frame-number').textContent = `${index + 1} / ${trace.frames.length}`;
  $('#time').textContent = `${row.time.toFixed(3)} s`;
  $('#state').textContent = row.state;
  $('#reason').textContent = row.reason;
  $('#brake').textContent = row.brake === null ? 'not recorded' : row.brake ? 'YES (dry-run)' : 'no';
  $('#matched').textContent = row.target_detected === null ? 'not recorded' : row.target_detected ? 'yes' : 'no';
  $('#latency').textContent = row.latency === null ? 'not recorded' : `${row.latency.toFixed(2)} ms`;
  $('#previous').disabled = index === 0; $('#next').disabled = index === trace.frames.length - 1;
  const hasNextBrake = trace.frames.slice(index + 1).some(frame => frame.brake === true);
  $('#next-brake').disabled = !hasNextBrake;
  scene.replaceChildren();
  for (let x=0;x<=960;x+=120) scene.append(shape('line',{x1:x,y1:0,x2:x,y2:540,stroke:'#203137'}));
  for (let y=0;y<=540;y+=90) scene.append(shape('line',{x1:0,y1:y,x2:960,y2:y,stroke:'#203137'}));
  scene.append(shape('text',{x:20,y:30,fill:'#83989f'},`FRAME ${index + 1} · ${row.time.toFixed(3)} s`));
  $('#actors').replaceChildren(...row.detections.map(item => {
    const [x,y,w,h] = item.box, color = colors[item.label] || '#bdccd1';
    scene.append(shape('rect',{x:x*960,y:y*540,width:w*960,height:h*540,stroke:color,'stroke-width':2,fill:color,'fill-opacity':0.1}));
    scene.append(shape('text',{x:Math.min(730, x*960+5),y:Math.max(55, y*540-8),fill:color},`${item.label} ${(item.confidence*100).toFixed(0)}%`));
    return node('li',`${item.label} · confidence ${item.confidence.toFixed(3)}`);
  }));
  if (!row.detections.length) $('#actors').append(node('li','No recorded actor components.'));
  const x = index / Math.max(1, trace.frames.length - 1) * 960;
  $('#cursor').setAttribute('x1',String(x)); $('#cursor').setAttribute('x2',String(x));
}
$('#context').textContent = replay.context;
replay.traces.forEach((item,i) => {const option = node('option',item.name.replaceAll('_',' ')); option.value = String(i); $('#sequence').append(option);});
$('#sequence').addEventListener('change',selected);
$('#frame').addEventListener('input',event => {index = Number(event.target.value); draw();});
$('#previous').addEventListener('click',() => {index = Math.max(0,index-1); draw();});
$('#next').addEventListener('click',() => {index = Math.min(trace.frames.length-1,index+1); draw();});
$('#next-brake').addEventListener('click',() => {const next = trace.frames.findIndex((row,i) => i>index && row.brake === true); if(next>=0) {index=next; draw();}});
selected();
