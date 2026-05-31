/* TruthNote 真相公示墙（社会辟谣大厅）— 插件内打包版逻辑
 * 来源：demo舞台/公开辟谣大厅.html 的内联脚本，逐行搬出为外部文件（MV3 扩展页 CSP 禁内联 <script>）。
 * 逻辑零改动，仅：① 从内联搬到此文件；② 表单提交防刷新从 HTML 内联 onsubmit 改为本文件 addEventListener。
 * 数据：优先 fetch 后端 /api/board，失败用内嵌种子（插件内 fetch 落到 chrome-extension 源会失败 → 走种子，离线照常展示）。
 */
/* ── 内嵌兜底数据（与后端 public_board.seed 对齐；file:// 直接打开也能跑）── */
const SEED = [
  {claim:"紧急通知！银行最新规定，个人存款超过5万元部分需缴纳20%利息税，本月起执行",verdict:"谣言",category:"政策法规",status:"已上报国家平台",heat:1342,reported_to:"中国互联网联合辟谣平台",evidence_urls:["https://www.pbc.gov.cn/","https://www.chinatax.gov.cn/"],min_ago:2},
  {claim:"冒充银行客服：您的账户存在风险，请将资金转入指定安全账户保护",verdict:"谣言",category:"诈骗",status:"已上报国家平台",heat:906,reported_to:"国家反诈中心",evidence_urls:["https://www.12321.cn/"],min_ago:5},
  {claim:"某某地今早发生8.5级强震，震波将在6小时内波及全国多省，请立即转移",verdict:"谣言",category:"灾害恐慌",status:"已上报国家平台",heat:2117,reported_to:"中国地震台网",evidence_urls:["https://www.ceic.ac.cn/"],min_ago:8},
  {claim:"微波炉加热的食物有辐射，长期吃会致癌，赶快转给家人",verdict:"谣言",category:"健康养生",status:"已辟谣",heat:588,reported_to:"",evidence_urls:["https://www.who.int/zh"],min_ago:13},
  {claim:"马云最新演讲金句：未来10年不懂AI的人将全部失业（附课程链接）",verdict:"谣言",category:"AI伪造",status:"已辟谣",heat:421,reported_to:"",evidence_urls:["https://www.piyao.org.cn/"],min_ago:21},
  {claim:"国家发放2026民生补贴每人最高2000元，扫码实名认证即可领取",verdict:"谣言",category:"诈骗",status:"已上报国家平台",heat:1755,reported_to:"国家反诈中心",evidence_urls:["https://www.12321.cn/"],min_ago:27},
  {claim:"每天喝一勺米醋，三个月软化血管降血压，比吃药还管用",verdict:"大部分不实",category:"健康养生",status:"已辟谣",heat:312,reported_to:"",evidence_urls:["https://www.nhc.gov.cn/"],min_ago:34},
  {claim:"螃蟹和柿子一起吃会产生砒霜，已有多人中毒入院",verdict:"大部分不实",category:"健康养生",status:"已辟谣",heat:264,reported_to:"",evidence_urls:["https://www.piyao.org.cn/"],min_ago:41},
  {claim:"某城市地铁X号线今早高峰发生严重踩踏事故，正在封锁现场",verdict:"无法核实",category:"旧闻翻炒",status:"待权威定论",heat:487,reported_to:"",evidence_urls:[],min_ago:1},
  {claim:"今天刚拍的！某地遭遇百年不遇洪灾，大量房屋被淹，紧急求助",verdict:"无法核实",category:"灾害恐慌",status:"待权威定论",heat:633,reported_to:"",evidence_urls:[],min_ago:4},
  {claim:"23点到1点是肝脏排毒时间，必须在23点前入睡否则肝脏无法排毒",verdict:"部分属实",category:"健康养生",status:"已标注",heat:178,reported_to:"",evidence_urls:["https://www.nhc.gov.cn/"],min_ago:52},
  {claim:"红果短剧宣布投入5亿元力挺真人短剧，AI抢不走演员饭碗",verdict:"属实",category:"综合",status:"已核实属实",heat:95,reported_to:"",evidence_urls:["https://www.xinhuanet.com/"],min_ago:67},
];

/* 实时模拟池：每隔几秒"刚刚有人遇到 → 自动核查"滑入顶部 */
const LIVE_POOL = [
  {claim:"扫码领取话费充值优惠，输入银行卡和验证码即可到账",verdict:"谣言",category:"诈骗",status:"已辟谣",reported_to:"",evidence_urls:["https://www.12321.cn/"]},
  {claim:"喝隔夜水会致癌，亚硝酸盐严重超标",verdict:"谣言",category:"健康养生",status:"已辟谣",reported_to:"",evidence_urls:["https://www.piyao.org.cn/"]},
  {claim:"明天起全国油价大涨3元，赶紧去加满",verdict:"无法核实",category:"政策法规",status:"待权威定论",reported_to:"",evidence_urls:[]},
  {claim:"某明星今日凌晨因病去世（配图）",verdict:"无法核实",category:"AI伪造",status:"待权威定论",reported_to:"",evidence_urls:[]},
  {claim:"五行缺水的人多喝水就能转运招财",verdict:"谣言",category:"综合",status:"已辟谣",reported_to:"",evidence_urls:["https://www.piyao.org.cn/"]},
  {claim:"国家正式宣布延迟退休到65岁，明年执行",verdict:"大部分不实",category:"政策法规",status:"已辟谣",reported_to:"",evidence_urls:["https://www.mohrss.gov.cn/"]},
  {claim:"绿豆汤包治百病，能根治高血压糖尿病",verdict:"谣言",category:"健康养生",status:"已辟谣",reported_to:"",evidence_urls:["https://www.nhc.gov.cn/"]},
];

const VERDICT_STYLE = {
  "谣言":{cls:"v-red",badge:"b-red",big:"谣言"},
  "大部分不实":{cls:"v-red",badge:"b-red",big:"谣言"},
  "误导性信息":{cls:"v-blue",badge:"b-blue",big:"存疑"},
  "部分属实":{cls:"v-blue",badge:"b-blue",big:"存疑"},
  "属实":{cls:"v-green",badge:"b-green",big:"属实"},
  "无法核实":{cls:"v-amber",badge:"b-amber",big:"待定"},
};
const STATUS_BADGE = {
  "已辟谣":"b-red","已上报国家平台":"b-red","已核实属实":"b-green",
  "待权威定论":"b-amber","已标注":"b-blue",
};

let items = [];      // 当前 feed
let serverStats = null;  // 后端返回的全量统计（官方辟谣库可达数千条）
let filter = "全部";
const CATS = ["全部","诈骗","健康养生","政策法规","灾害恐慌","AI伪造","旧闻翻炒","综合"];

function esc(s){const d=document.createElement("div");d.textContent=s==null?"":String(s);return d.innerHTML;}
// 仅放行 http(s)，并对引号/尖括号做属性上下文转义——防 evidence_urls 接后端 feed 后用畸形 URL 逃出 href="" 注入
function safeUrl(u){
  if(!/^https?:\/\//i.test(u||""))return"#";
  return String(u).replace(/"/g,"%22").replace(/'/g,"%27").replace(/</g,"%3C").replace(/>/g,"%3E");
}
function agoText(min){
  if(min<1)return"刚刚";if(min<60)return min+"分钟前";
  const h=Math.floor(min/60);return h+"小时前";
}
function nowMinusMin(min){return new Date(Date.now()-min*60000).toISOString();}

function normalizeSeed(s){
  return {...s, created_at: nowMinusMin(s.min_ago)};
}

function render(){
  // 统计：优先用后端全量统计（官方辟谣库数千条）；离线 file:// 时按当前 feed 估算
  let total, debunked, awaiting, reported, heat;
  if(serverStats){
    total=serverStats.total_checked; debunked=serverStats.debunked;
    awaiting=serverStats.awaiting_authority; reported=serverStats.reported_to_platform;
    heat=serverStats.total_heat;
  }else{
    debunked=items.filter(x=>x.status==="已辟谣"||x.status==="已上报国家平台").length;
    awaiting=items.filter(x=>x.status==="待权威定论").length;
    reported=items.filter(x=>x.status==="已上报国家平台").length;
    heat=items.reduce((a,x)=>a+(x.heat||0),0);
    total=items.length;
  }
  const S=[
    {n:total.toLocaleString(),l:"累计核查",c:""},
    {n:debunked.toLocaleString(),l:"已辟谣",c:"red"},
    {n:awaiting,l:"待核实·未定论",c:"amber"},
    {n:reported,l:"已上报国家平台",c:"red"},
    {n:heat.toLocaleString(),l:"累计触达人次",c:"green"},
  ];
  document.getElementById("stats").innerHTML = S.map(s=>
    `<div class="stat ${s.c}"><div class="n">${esc(s.n)}</div><div class="l">${esc(s.l)}</div></div>`).join("");

  // 分类 chips
  document.getElementById("chips").innerHTML = CATS.map(c=>
    `<div class="chip ${c===filter?"on":""}" data-cat="${esc(c)}">${esc(c)}</div>`).join("");
  document.querySelectorAll(".chip").forEach(el=>el.onclick=()=>{filter=el.dataset.cat;render();});

  // feed
  const list = items.filter(x=>filter==="全部"||x.category===filter)
                    .slice().sort((a,b)=>b.created_at.localeCompare(a.created_at));
  document.getElementById("feedCount").textContent = `实时核查流 · ${list.length} 条`;
  document.getElementById("feed").innerHTML = list.map(renderCard).join("");
  document.querySelectorAll(".expand").forEach(el=>el.onclick=()=>{
    const e=el.closest(".body").querySelector(".evi"); if(e)e.classList.toggle("open");
    el.textContent = e&&e.classList.contains("open")?"收起证据 ▲":"查看证据链 ▼";
  });
}

function renderCard(x){
  const v=VERDICT_STYLE[x.verdict]||VERDICT_STYLE["无法核实"];
  const min=Math.max(0,Math.round((Date.now()-new Date(x.created_at).getTime())/60000));
  const statusCls=STATUS_BADGE[x.status]||"b-amber";
  const reportHtml = x.status==="已上报国家平台"
    ? `<span class="report">⬆ 已上报 ${esc(x.reported_to||"国家辟谣平台")}</span>` : "";
  const sourceHtml = (x.status!=="已上报国家平台" && x.reported_to)
    ? `<span class="src-tag">来源 · ${esc(x.reported_to)}</span>` : "";
  const eviLinks = (x.evidence_urls||[]).length
    ? (x.evidence_urls.map(u=>`<a href="${safeUrl(u)}" target="_blank" rel="noopener">${esc((u||"").replace(/^https?:\/\//,"").replace(/\/$/,""))}</a>`).join(""))
    : `<span class="src-tag">权威源尚未表态 —— 已订阅，结论出现后自动回填</span>`;
  const expandable = x.verdict!=="无法核实";
  return `<div class="card ${x._flash?"flash":""}">
    <div class="stamp ${v.cls}"><div class="big">${esc(v.big)}</div><div class="conf">${esc(x.verdict)}</div></div>
    <div class="body">
      <p class="claim">${esc(x.claim)}</p>
      <div class="meta">
        <span class="cat">${esc(x.category)}</span>
        <span class="badge ${statusCls}">${esc(x.status)}</span>
        ${reportHtml}
        ${sourceHtml}
        <span class="heat">🔥 ${(x.heat||0).toLocaleString()} 人遇到</span>
        <span class="ago">${agoText(min)}</span>
      </div>
      <div class="expand">${expandable?"查看证据链 ▼":"查看说明 ▼"}</div>
      <div class="evi">${eviLinks}</div>
    </div>
  </div>`;
}

function toast(msg){
  const t=document.getElementById("toast");t.textContent=msg;t.classList.add("show");
  clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove("show"),2600);
}

/* 实时模拟：每 4-6 秒，要么新核查滑入，要么已有条目热度上涨 */
let livePtr=0;
function liveTick(){
  if(Math.random()<0.62){
    const tpl=LIVE_POOL[livePtr%LIVE_POOL.length];livePtr++;
    const it=normalizeSeed({...tpl,heat:Math.floor(20+Math.random()*180),min_ago:0,_flash:true});
    items.unshift(it);
    if(items.length>60)items.pop();
    if(serverStats){  // 实时累加真实总数，让大屏数字"在动"
      serverStats.total_checked++;
      serverStats.total_heat+=it.heat;
      if(serverStats.today_new!=null)serverStats.today_new++;
      if(it.status==="已辟谣"||it.status==="已上报国家平台")serverStats.debunked++;
      if(it.status==="待权威定论")serverStats.awaiting_authority++;
    }
    render();
    toast(`刚刚有 ${it.heat} 位用户遇到这条，已自动核查 → ${it.verdict}`);
    setTimeout(()=>{it._flash=false;},1800);
  }else{
    // 热度上涨：随机挑一条 +N
    const pick=items[Math.floor(Math.random()*Math.min(8,items.length))];
    if(pick){pick.heat=(pick.heat||0)+Math.floor(5+Math.random()*40);render();}
  }
}

function submitOne(){
  const inp=document.getElementById("submitInput");
  const txt=(inp.value||"").trim();if(!txt)return;
  inp.value="";
  const pending=normalizeSeed({claim:txt,verdict:"无法核实",category:"综合",
    status:"核查中…",heat:1,reported_to:"",evidence_urls:[],min_ago:0,_flash:true});
  pending._pending=true;items.unshift(pending);render();
  toast("已提交，进入核查队列…");
  setTimeout(()=>{
    // 模拟核查完成（雏形：随机给个确定判定）
    const r=Math.random();
    if(r<0.6){pending.verdict="谣言";pending.status="已辟谣";pending.evidence_urls=["https://www.piyao.org.cn/"];}
    else if(r<0.8){pending.verdict="无法核实";pending.status="待权威定论";}
    else{pending.verdict="属实";pending.status="已核实属实";pending.evidence_urls=["https://www.xinhuanet.com/"];}
    pending._pending=false;pending._flash=true;render();
    toast(`你提交的内容核查完成 → ${pending.verdict}`);
    setTimeout(()=>{pending._flash=false;},1800);
  },2200);
}

function startClock(){
  const c=document.getElementById("clock");
  setInterval(()=>{const d=new Date();
    c.textContent=d.toLocaleTimeString("zh-CN",{hour12:false})+" · "+d.toLocaleDateString("zh-CN");},1000);
}

async function boot(){
  startClock();
  // 优先取后端真实 feed；失败用内嵌种子（插件内 fetch 落到 chrome-extension 源会失败 → 走种子）
  try{
    const res=await fetch("/api/board?limit=60");
    if(res.ok){
      const data=await res.json();
      if(data.items&&data.items.length){
        items=data.items.map(it=>({
          claim:it.claim_text,verdict:it.verdict,category:it.category,status:it.status,
          heat:it.heat,reported_to:it.reported_to,evidence_urls:it.evidence_urls,
          created_at:it.created_at||nowMinusMin(1),
        }));
        serverStats=data.stats||null;
        const tot=serverStats?serverStats.total_checked.toLocaleString():"";
        const nw=serverStats&&serverStats.today_new?` · 今日新增 ${serverStats.today_new}`:"";
        document.getElementById("dataSrc").textContent=
          `数据源：官方辟谣库 ${tot} 条真实数据 + 实时核查${nw}`;
      }
    }
  }catch(e){/* file:// 或后端未起 → 用内嵌种子 */}
  if(!items.length){
    items=SEED.map(normalizeSeed);
    document.getElementById("dataSrc").textContent="数据源：内嵌种子样本（离线雏形）";
  }
  render();
  document.getElementById("submitBtn").onclick=submitOne;
  document.getElementById("submitInput").addEventListener("keydown",e=>{if(e.key==="Enter")submitOne();});
  // CSP：原 HTML 内联 onsubmit="return false" 被 MV3 拦，改在此处阻止表单默认提交（防回车刷新页面）
  const _form=document.getElementById("submitForm");
  if(_form)_form.addEventListener("submit",e=>e.preventDefault());
  setInterval(liveTick,4800);
}
boot();
