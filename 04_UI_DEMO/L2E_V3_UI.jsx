import { useState } from "react";

const LEARNERS = [
  { id: "LNR-202600002", name: "Piywdi D.", title: "Programmer", skills: ["Python", "Java", "JavaScript", "Git"], exp: 3, salary: 57000 },
  { id: "LNR-202600003", name: "Suthinanth P.", title: "UX/UI Designer", skills: ["Figma", "UX Design", "Wireframing", "Prototyping"], exp: 2, salary: 74000 },
  { id: "LNR-202600005", name: "Phiphiththn K.", title: "DevOps Engineer", skills: ["Docker", "Kubernetes", "CI/CD", "AWS"], exp: 5, salary: 36000 },
];

const JOBS = [
  { id: "JOB-202600001", title: "Cybersecurity Analyst", company: "Bangkok Bank", type: "Full-time", location: "On-site", province: "กรุงเทพมหานคร", salaryMin: 65000, salaryMax: 102000, skills: ["JavaScript", "Git", "Java", "Docker"], category: "ไอที เทคโนโลยีสื่อสาร" },
  { id: "JOB-202600002", title: "DevOps Engineer", company: "PTT Digital", type: "Full-time", location: "On-site", province: "กรุงเทพมหานคร", salaryMin: 40000, salaryMax: 62000, skills: ["CI/CD", "Linux", "Docker", "Kubernetes"], category: "ไอที เทคโนโลยีสื่อสาร" },
  { id: "JOB-202600003", title: "Data Scientist", company: "SCG", type: "Full-time", location: "Hybrid", province: "นนทบุรี", salaryMin: 55000, salaryMax: 86000, skills: ["Statistics", "Machine Learning", "SQL", "Tableau"], category: "ไอที เทคโนโลยีสื่อสาร" },
  { id: "JOB-202600004", title: "Data Analyst", company: "Grab Thailand", type: "Full-time", location: "Hybrid", province: "กรุงเทพมหานคร", salaryMin: 50000, salaryMax: 78000, skills: ["SQL", "Machine Learning", "Statistics", "Tableau"], category: "ไอที เทคโนโลยีสื่อสาร" },
  { id: "JOB-202600005", title: "UX/UI Designer", company: "LINE Thailand", type: "Full-time", location: "Hybrid", province: "กรุงเทพมหานคร", salaryMin: 45000, salaryMax: 70000, skills: ["Figma", "UX Design", "Prototyping", "User Research"], category: "ดิจิทัล/ครีเอทีฟ" },
];

const POOL_JOB = { id: "JOB-202601570", title: "Data Analyst", company: "Sansiri", salaryMin: 45000, salaryMax: 70000, skills: ["Python", "Machine Learning", "SQL", "Tableau", "Presentation"] };

const POOL_CANDIDATES = [
  { id: "LNR-202604978", name: "Oaaeoesaa W.", title: "Data Scientist", source: "direct_apply", status: "hired", score: 91, matched: ["Python", "ML", "SQL"], missing: ["Tableau"] },
  { id: "LNR-202605199", name: "Phrebyya S.", title: "Programmer", source: "pool", status: "offer", score: 73, matched: ["Python", "Java"], missing: ["SQL", "ML"] },
  { id: "LNR-202603720", name: "Oophas T.", title: "Civil Engineer", source: "recommendation", status: "applied", score: 38, matched: ["Presentation"], missing: ["Python", "SQL", "ML"] },
  { id: "LNR-202601894", name: "Thara N.", title: "UX/UI Designer", source: "direct_apply", status: "applied", score: 25, matched: [], missing: ["Python", "SQL", "ML", "Tableau"] },
  { id: "LNR-202605649", name: "Suprani K.", title: "Digital Marketer", source: "direct_apply", status: "screening", score: 18, matched: [], missing: ["Python", "SQL", "ML"] },
  { id: "LNR-202609177", name: "Khmkchy R.", title: "Civil Engineer", source: "direct_apply", status: "rejected", score: 12, matched: [], missing: ["Python", "SQL", "ML", "Tableau"] },
];

function getMatchScore(learnerSkills, jobSkills) {
  const m = jobSkills.filter(s => learnerSkills.includes(s));
  const base = Math.round((m.length / jobSkills.length) * 60 + Math.random() * 20 + 10);
  return Math.min(base, 99);
}

function getMatched(ls, js) { return js.filter(s => ls.includes(s)); }
function getMissing(ls, js) { return js.filter(s => !ls.includes(s)); }

function ScoreBadge({ score }) {
  const color = score >= 75 ? "text-green-600" : score >= 50 ? "text-amber-500" : "text-red-500";
  return <span className={`text-xl font-bold ${color}`}>{score}%</span>;
}

function SkillTag({ name, type }) {
  if (type === "match") return <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">{name} ✓</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-500">{name} ↗</span>;
}

function SourceBadge({ source }) {
  if (source === "recommendation") return <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700">🤖 AI</span>;
  if (source === "direct_apply") return <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">📝 สมัครเอง</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">👆 HR เพิ่ม</span>;
}

function StatusDot({ status }) {
  const map = { hired: "bg-green-500", offer: "bg-amber-400", interview: "bg-blue-500", screening: "bg-sky-400", applied: "bg-gray-400", rejected: "bg-red-400" };
  return <span className="flex items-center gap-1.5 text-xs text-gray-500"><span className={`w-2 h-2 rounded-full ${map[status] || "bg-gray-300"}`}></span>{status}</span>;
}

// ─── LEARNER VIEW ────────────────────────────────────────────────
function LearnerView() {
  const [learner, setLearner] = useState(LEARNERS[0]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  function search() {
    setLoading(true);
    setSelected(null);
    setTimeout(() => {
      const r = JOBS.map(j => ({
        ...j,
        score: getMatchScore(learner.skills, j.skills),
        matched: getMatched(learner.skills, j.skills),
        missing: getMissing(learner.skills, j.skills),
      })).sort((a, b) => b.score - a.score);
      setResults(r);
      setLoading(false);
    }, 900);
  }

  return (
    <div className="flex h-[calc(100vh-112px)]">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="text-xs text-gray-400 mb-1">ผู้เรียน</div>
          <select className="w-full text-sm border rounded-lg px-2 py-1.5" onChange={e => { setLearner(LEARNERS.find(l => l.id === e.target.value)); setResults(null); setSelected(null); }}>
            {LEARNERS.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
          </select>
        </div>
        <div className="p-4 flex-1">
          <div className="text-xs text-gray-400 mb-2">โปรไฟล์</div>
          <div className="font-semibold">{learner.name}</div>
          <div className="text-sm text-gray-500">{learner.title}</div>
          <div className="text-sm text-gray-500 mt-1">ประสบการณ์ {learner.exp} ปี</div>
          <div className="mt-3 flex flex-wrap gap-1">
            {learner.skills.map(s => <span key={s} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">{s}</span>)}
          </div>
        </div>
        <div className="p-4 border-t">
          <button onClick={search} className="w-full bg-[#1a3a2a] text-white text-sm font-semibold py-2.5 rounded-xl hover:bg-green-800 transition">
            {loading ? "กำลังค้นหา..." : "🔍 หางานที่เหมาะ"}
          </button>
        </div>
      </div>

      {/* Job List */}
      <div className="flex-1 overflow-y-auto bg-gray-50 p-4">
        {!results && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="text-4xl mb-3">💼</div>
            <div className="text-sm">กดปุ่มหางานเพื่อให้ AI แนะนำ</div>
          </div>
        )}
        {loading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="w-8 h-8 border-2 border-gray-200 border-t-green-500 rounded-full animate-spin mb-3"></div>
            <div className="text-sm">AI กำลังวิเคราะห์...</div>
          </div>
        )}
        {results && (
          <div>
            <div className="text-sm text-gray-500 mb-3">งานที่เหมาะกับคุณที่สุด ({results.length} ตำแหน่ง)</div>
            <div className="grid grid-cols-2 gap-3">
              {results.map(job => (
                <div key={job.id} onClick={() => setSelected(job)}
                  className={`bg-white rounded-xl p-4 cursor-pointer border-2 transition ${selected?.id === job.id ? "border-green-500 shadow-md" : "border-transparent hover:border-green-200"}`}>
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gray-100 flex items-center justify-center text-lg flex-shrink-0">🏢</div>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm truncate">{job.title}</div>
                      <div className="text-xs text-gray-500 truncate">{job.company}</div>
                      <div className="flex gap-1.5 mt-1.5">
                        <span className="text-xs bg-green-50 text-green-700 px-2 py-0.5 rounded-full">{job.location}</span>
                        <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full truncate">{job.category}</span>
                      </div>
                    </div>
                    <ScoreBadge score={job.score} />
                  </div>
                  <div className="text-xs text-gray-400 mt-2 truncate">{job.province} · {job.type}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Detail Panel */}
      {selected && (
        <div className="w-72 bg-white border-l overflow-y-auto p-5">
          <div className="w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center text-2xl mb-3">🏢</div>
          <div className="font-bold text-base">{selected.title}</div>
          <div className="text-sm text-gray-500 mb-4">{selected.company}</div>
          <div className="flex items-center justify-between bg-gray-50 rounded-xl p-3 mb-4">
            <span className="text-sm text-gray-500">คุณ Match กับงานนี้</span>
            <ScoreBadge score={selected.score} />
          </div>
          <button className="w-full bg-green-500 text-white text-sm font-semibold py-2.5 rounded-xl mb-3 hover:bg-green-600">สมัครงาน</button>
          <div className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-y-2 text-xs">
              <div className="text-gray-400">ประเภทการจ้าง</div><div className="text-green-700 font-medium">{selected.type}</div>
              <div className="text-gray-400">สถานที่</div><div className="text-green-700 font-medium">{selected.location}</div>
              <div className="text-gray-400">จังหวัด</div><div className="font-medium">{selected.province}</div>
              <div className="text-gray-400">เงินเดือน</div><div className="font-medium">{selected.salaryMin.toLocaleString()}–{selected.salaryMax.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1.5">ทักษะ</div>
              <div className="flex flex-wrap gap-1">
                {selected.matched.map(s => <SkillTag key={s} name={s} type="match" />)}
                {selected.missing.map(s => <SkillTag key={s} name={s} type="missing" />)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── EMPLOYER VIEW ───────────────────────────────────────────────
function EmployerView() {
  const [job, setJob] = useState(JOBS[0]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  function search() {
    setLoading(true);
    setSelected(null);
    setTimeout(() => {
      const r = LEARNERS.concat([
        { id: "LNR-202600010", name: "Somchai T.", title: "Security Engineer", skills: ["JavaScript", "Git", "Docker", "Java"], exp: 5, salary: 80000 },
        { id: "LNR-202600011", name: "Pranee K.", title: "IT Specialist", skills: ["JavaScript", "Java", "Git"], exp: 3, salary: 60000 },
        { id: "LNR-202600012", name: "Wanchai P.", title: "Network Engineer", skills: ["Git", "Linux"], exp: 2, salary: 45000 },
      ]).map(l => ({
        ...l,
        score: getMatchScore(l.skills, job.skills),
        matched: getMatched(l.skills, job.skills),
        missing: getMissing(l.skills, job.skills),
      })).sort((a, b) => b.score - a.score);
      setResults(r);
      setLoading(false);
    }, 900);
  }

  return (
    <div className="flex h-[calc(100vh-112px)]">
      <div className="w-64 bg-white border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="text-xs text-gray-400 mb-1">ประกาศงาน</div>
          <select className="w-full text-sm border rounded-lg px-2 py-1.5" onChange={e => { setJob(JOBS.find(j => j.id === e.target.value)); setResults(null); setSelected(null); }}>
            {JOBS.map(j => <option key={j.id} value={j.id}>{j.title}</option>)}
          </select>
        </div>
        <div className="p-4 flex-1">
          <div className="text-xs text-gray-400 mb-2">รายละเอียดงาน</div>
          <div className="font-semibold">{job.title}</div>
          <div className="text-sm text-gray-500">{job.company}</div>
          <div className="text-sm text-gray-500 mt-1">{job.salaryMin.toLocaleString()}–{job.salaryMax.toLocaleString()} บาท</div>
          <div className="mt-3 flex flex-wrap gap-1">
            {job.skills.map(s => <span key={s} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">{s}</span>)}
          </div>
        </div>
        <div className="p-4 border-t">
          <button onClick={search} className="w-full bg-blue-700 text-white text-sm font-semibold py-2.5 rounded-xl hover:bg-blue-800 transition">
            {loading ? "กำลังค้นหา..." : "🤖 แนะนำด้วย AI"}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto bg-gray-50 p-4">
        {!results && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="text-4xl mb-3">👥</div>
            <div className="text-sm">กดปุ่มให้ AI แนะนำผู้สมัคร</div>
          </div>
        )}
        {loading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="w-8 h-8 border-2 border-gray-200 border-t-blue-500 rounded-full animate-spin mb-3"></div>
            <div className="text-sm">AI กำลังวิเคราะห์...</div>
          </div>
        )}
        {results && (
          <div className="space-y-2">
            <div className="text-sm text-gray-500 mb-3">ผู้สมัครที่เหมาะสม ({results.length} คน)</div>
            {results.map((c, i) => (
              <div key={c.id} onClick={() => setSelected(c)}
                className={`bg-white rounded-xl p-4 cursor-pointer border-2 flex items-center gap-4 transition ${selected?.id === c.id ? "border-blue-500" : "border-transparent hover:border-blue-200"}`}>
                <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center text-sm font-bold text-blue-700 flex-shrink-0">
                  {i + 1}
                </div>
                <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-lg flex-shrink-0">👤</div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm">{c.name}</div>
                  <div className="text-xs text-gray-500">{c.title} · {c.exp} ปี</div>
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {c.matched.map(s => <SkillTag key={s} name={s} type="match" />)}
                    {c.missing.slice(0, 2).map(s => <SkillTag key={s} name={s} type="missing" />)}
                  </div>
                </div>
                <ScoreBadge score={c.score} />
              </div>
            ))}
          </div>
        )}
      </div>

      {selected && (
        <div className="w-72 bg-white border-l overflow-y-auto p-5">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center text-2xl mb-3">👤</div>
          <div className="font-bold text-base">{selected.name}</div>
          <div className="text-sm text-gray-500 mb-4">{selected.title} · {selected.exp} ปี</div>
          <div className="flex items-center justify-between bg-gray-50 rounded-xl p-3 mb-4">
            <span className="text-sm text-gray-500">Match Score</span>
            <ScoreBadge score={selected.score} />
          </div>
          <div className="flex gap-2 mb-4">
            <button className="flex-1 bg-blue-700 text-white text-sm font-semibold py-2 rounded-xl hover:bg-blue-800">เชิญสัมภาษณ์</button>
            <button className="flex-1 border border-gray-200 text-sm font-medium py-2 rounded-xl hover:bg-gray-50">บันทึก</button>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1.5">ทักษะ</div>
            <div className="flex flex-wrap gap-1">
              {selected.matched.map(s => <SkillTag key={s} name={s} type="match" />)}
              {selected.missing.map(s => <SkillTag key={s} name={s} type="missing" />)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── POOL VIEW ───────────────────────────────────────────────────
function PoolView() {
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [selected, setSelected] = useState(null);

  function analyze() {
    setLoading(true);
    setAnalyzed(false);
    setTimeout(() => { setLoading(false); setAnalyzed(true); }, 1200);
  }

  const sorted = [...POOL_CANDIDATES].sort((a, b) => b.score - a.score);

  return (
    <div className="flex h-[calc(100vh-112px)]">
      <div className="w-64 bg-white border-r flex flex-col">
        <div className="p-4 flex-1">
          <div className="text-xs text-gray-400 mb-2">ตำแหน่งงาน</div>
          <div className="font-semibold">{POOL_JOB.title}</div>
          <div className="text-sm text-gray-500">{POOL_JOB.company}</div>
          <div className="text-sm text-gray-500 mt-1">{POOL_JOB.salaryMin.toLocaleString()}–{POOL_JOB.salaryMax.toLocaleString()} บาท</div>
          <div className="mt-3 flex flex-wrap gap-1">
            {POOL_JOB.skills.map(s => <span key={s} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">{s}</span>)}
          </div>
          <div className="mt-4 border-t pt-4">
            <div className="text-xs text-gray-400 mb-2">ผู้สมัครใน Pool ({POOL_CANDIDATES.length} คน)</div>
            <div className="space-y-1.5">
              <div className="text-xs flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-purple-500"></span>AI แนะนำ: 1 คน</div>
              <div className="text-xs flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-blue-500"></span>สมัครเอง: 4 คน</div>
              <div className="text-xs flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500"></span>HR เพิ่ม: 1 คน</div>
            </div>
          </div>
        </div>
        <div className="p-4 border-t">
          <button onClick={analyze} className="w-full bg-purple-700 text-white text-sm font-semibold py-2.5 rounded-xl hover:bg-purple-800 transition">
            {loading ? "กำลังวิเคราะห์..." : "🔍 วิเคราะห์ทั้ง Pool"}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto bg-gray-50 p-4">
        {!analyzed && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="text-4xl mb-3">📊</div>
            <div className="text-sm">กดวิเคราะห์เพื่อดู % Match ทุกคน</div>
          </div>
        )}
        {loading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="w-8 h-8 border-2 border-gray-200 border-t-purple-500 rounded-full animate-spin mb-3"></div>
            <div className="text-sm">AI กำลังวิเคราะห์...</div>
          </div>
        )}
        {analyzed && (
          <div className="space-y-2">
            <div className="text-sm text-gray-500 mb-3">เรียงตาม % Match</div>
            {sorted.map((c, i) => (
              <div key={c.id} onClick={() => setSelected(c)}
                className={`bg-white rounded-xl p-4 cursor-pointer border-2 flex items-center gap-4 transition ${selected?.id === c.id ? "border-purple-500" : "border-transparent hover:border-purple-200"}`}>
                <div className="text-sm font-bold text-gray-400 w-5 text-center">{i + 1}</div>
                <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-lg flex-shrink-0">👤</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="font-semibold text-sm">{c.name}</span>
                    <SourceBadge source={c.source} />
                  </div>
                  <div className="text-xs text-gray-500">{c.title}</div>
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {c.matched.map(s => <SkillTag key={s} name={s} type="match" />)}
                    {c.missing.slice(0, 2).map(s => <SkillTag key={s} name={s} type="missing" />)}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <ScoreBadge score={c.score} />
                  <StatusDot status={c.status} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selected && (
        <div className="w-72 bg-white border-l overflow-y-auto p-5">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center text-2xl mb-3">👤</div>
          <div className="font-bold text-base">{selected.name}</div>
          <div className="text-sm text-gray-500 mb-1">{selected.title}</div>
          <div className="mb-4"><SourceBadge source={selected.source} /></div>
          <div className="flex items-center justify-between bg-gray-50 rounded-xl p-3 mb-4">
            <span className="text-sm text-gray-500">Match Score</span>
            <ScoreBadge score={selected.score} />
          </div>
          <div className="mb-4">
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div className="h-2 rounded-full bg-green-500 transition-all" style={{ width: `${selected.score}%` }}></div>
            </div>
          </div>
          <div className="flex gap-2 mb-4">
            <button className="flex-1 bg-purple-700 text-white text-sm font-semibold py-2 rounded-xl hover:bg-purple-800">เชิญสัมภาษณ์</button>
            <button className="flex-1 border text-sm py-2 rounded-xl hover:bg-gray-50">ข้าม</button>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1.5">ทักษะที่ตรง</div>
            <div className="flex flex-wrap gap-1 mb-3">
              {selected.matched.length ? selected.matched.map(s => <SkillTag key={s} name={s} type="match" />) : <span className="text-xs text-gray-400">ไม่มี</span>}
            </div>
            <div className="text-xs text-gray-400 mb-1.5">ทักษะที่ขาด</div>
            <div className="flex flex-wrap gap-1">
              {selected.missing.map(s => <SkillTag key={s} name={s} type="missing" />)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── APP ─────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("learner");

  const tabs = [
    { id: "learner", label: "👤 ผู้เรียนหางาน" },
    { id: "employer", label: "🏢 บริษัทหาผู้สมัคร" },
    { id: "pool", label: "🔍 วิเคราะห์ Pool" },
  ];

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Top Nav */}
      <div className="bg-[#1a3a2a] text-white px-6 py-3 flex items-center gap-4">
        <span className="text-lg font-bold text-green-400">L2E</span>
        <span className="text-xs text-green-300">AI Job Matching</span>
        <div className="flex gap-1 ml-auto">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition ${tab === t.id ? "bg-green-400 text-[#1a3a2a]" : "text-gray-300 hover:text-white"}`}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Sub header */}
      <div className="bg-white border-b px-6 py-2.5 text-sm text-gray-500 flex items-center gap-2">
        {tab === "learner" && <><span className="font-medium text-gray-700">หางาน</span><span>— AI แนะนำงานที่เหมาะกับโปรไฟล์ของคุณ</>}
        {tab === "employer" && <><span className="font-medium text-gray-700">ค้นหาผู้สมัคร</span><span>— AI แนะนำผู้สมัครที่ตรงกับตำแหน่ง</>}
        {tab === "pool" && <><span className="font-medium text-gray-700">วิเคราะห์ Pool</span><span>— AI วิเคราะห์ % Match ผู้สมัครทุกคนใน Pool</>}
      </div>

      {tab === "learner" && <LearnerView />}
      {tab === "employer" && <EmployerView />}
      {tab === "pool" && <PoolView />}
    </div>
  );
}
