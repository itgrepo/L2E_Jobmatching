from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import os
import traceback
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("⏳ L2E Server Initializing...")
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'models')
DATASET_PATH = os.path.join(BASE_DIR, 'DATA SET')

# In-memory profile override store (for onboarding saves)
profile_overrides = {}

# ==========================================
# 1. LOAD BERT MODEL
# ==========================================
bert_model = None
try:
    custom = os.path.join(MODEL_PATH, 'fine_tuned_bert')
    if os.path.exists(custom):
        bert_model = SentenceTransformer(custom)
        print("   ✓ Loaded fine-tuned BERT model")
    else:
        bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("   ✓ Loaded base multilingual BERT model")
except Exception as e:
    print(f"   ❌ BERT load error: {e}")

# ==========================================
# 2. LOAD ALL CSV TABLES
# ==========================================
print("   - Loading CSV tables...")

def load_csv(filename):
    path = os.path.join(DATASET_PATH, filename)
    if os.path.exists(path):
        return pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    print(f"  ⚠️ Missing: {filename}")
    return pd.DataFrame()

df_learner    = load_csv('TBL_LEARNER.csv')
df_skills     = load_csv('TBL_LEARNER_SKILL.csv')
df_certs      = load_csv('TBL_LEARNER_CERTIFICATEANDSKILL.csv')
df_education  = load_csv('TBL_LEARNER_EDUCATION.csv')
df_experience = load_csv('TBL_LEARNER_EXPERINCE.csv')
df_preference = load_csv('TBL_LEARNER_PREFERENCE.csv')
df_job_post   = load_csv('job_post.csv')
df_job_skill  = load_csv('job_post_skill.csv')
df_company    = load_csv('company.csv')
df_university = load_csv('TBM_UNIVERSITY.csv')
df_branch     = load_csv('TBM_BRANCH.csv')
df_edu_level  = load_csv('TBM_EDUCATIONBACKBROUND.csv')
df_position   = load_csv('TBM_POSITION.csv')
df_worktype   = load_csv('TBM_WORKTYPE.csv')
df_jobtype    = load_csv('TBM_JOBTYPE.csv')
df_postcode   = load_csv('TBM_POSTCODE.csv')
df_acad_skill = load_csv('Academy_skill_and_certificate.csv')
df_acad_course= load_csv('Academy_course.csv')

print(f"   ✓ Learners: {len(df_learner)}, Jobs: {len(df_job_post)}, Certs: {len(df_certs)}")

# ==========================================
# 3. BUILD LOOKUP TABLES
# ==========================================
uni_map      = dict(zip(df_university['UNIVERSITY_ID'].astype(str), df_university['UNIVERSITY_NAME'])) if not df_university.empty else {}
branch_map   = dict(zip(df_branch['BRANCHID'].astype(str), df_branch['BRANCH_NAME'])) if not df_branch.empty else {}
edu_lvl_map  = dict(zip(df_edu_level['EDUCATOINBG_ID'].astype(str), df_edu_level['EDUCATOINBG_NAME'])) if not df_edu_level.empty else {}
pos_map      = dict(zip(df_position['POSITION_ID'].astype(str), df_position['POSITION_NAME'])) if not df_position.empty else {}
wt_map       = dict(zip(df_worktype['WORKTYPE_ID'].astype(str), df_worktype['WORKTYPE_NAME'])) if not df_worktype.empty else {}
jt_map       = dict(zip(df_jobtype['JOBTYPE_ID'].astype(str), df_jobtype['JOBTYPE_NAME'])) if not df_jobtype.empty else {}
skill_id_map = dict(zip(df_acad_skill['skill_id'], df_acad_skill['skill_name'])) if not df_acad_skill.empty else {}

prov_map = {}  # SUBDISTRICT_ID -> PROVINCE_NAME
prov_id_map = {}  # PROVINCE_ID -> PROVINCE_NAME
if not df_postcode.empty:
    prov_map    = dict(zip(df_postcode['SUBDISTRICT_ID'].astype(str), df_postcode['PROVINCE_NAME']))
    prov_id_map = dict(zip(df_postcode['PROVINCE_ID'].astype(str), df_postcode['PROVINCE_NAME']))

company_map = {}
if not df_company.empty:
    for _, row in df_company.iterrows():
        cid = str(row.get('company_id', ''))
        company_map[cid] = {
            'name_th': str(row.get('company_name_th', '')),
            'name_en': str(row.get('company_name_en', '')),
            'logo': str(row.get('logo_image', '')),
            'website': str(row.get('company_website', '')),
            'specialty': str(row.get('specialty', '')),
            'size': str(row.get('organization_size', '')),
            'about': str(row.get('about_me', '')),
        }

# Academy skills list (for onboarding dropdowns)
academy_skills_list = []
cert_skills_list = []
if not df_acad_skill.empty:
    unique_skills = df_acad_skill[['skill_id','skill_name']].drop_duplicates()
    for _, r in unique_skills.iterrows():
        is_cert = bool(df_acad_skill[df_acad_skill['skill_id'] == r['skill_id']]['Is_certificate'].any())
        if is_cert:
            cert_skills_list.append({'id': r['skill_id'], 'name': r['skill_name']})
        else:
            academy_skills_list.append({'id': r['skill_id'], 'name': r['skill_name']})

# ==========================================
# 4. AGGREGATE LEARNER DATA
# ==========================================
print("   - Aggregating learner data...")

learner_skills = {}
if not df_skills.empty:
    for _, row in df_skills.iterrows():
        lid = str(row['LEARNER_ID'])
        if lid not in learner_skills: learner_skills[lid] = []
        learner_skills[lid].append({'name': str(row.get('SKILL_NAME', '')), 'level': str(row.get('LEVEL', ''))})

learner_certs = {}
if not df_certs.empty:
    for _, row in df_certs.iterrows():
        lid = str(row['LEARNER_ID'])
        if lid not in learner_certs: learner_certs[lid] = []
        learner_certs[lid].append({
            'course_name': str(row.get('COURCE_NAME', '')),
            'academy': str(row.get('ACADEMY_NAME', '')),
            'received_date': str(row.get('RECIVE_DATE', '')),
            'link': str(row.get('LINK_CERTIFICATE', ''))
        })

learner_edu = {}
if not df_education.empty:
    for _, row in df_education.iterrows():
        lid = str(row['LEARNER_ID'])
        if lid not in learner_edu: learner_edu[lid] = []
        learner_edu[lid].append({
            'university': uni_map.get(str(row.get('UNIVERSITY_ID', '')), f"มหาวิทยาลัย {row.get('UNIVERSITY_ID','')}"),
            'degree': edu_lvl_map.get(str(row.get('EDUCATOINBG_ID', '')), 'ปริญญาตรี'),
            'major': branch_map.get(str(row.get('BARNCH_ID', '')), f"สาขา {row.get('BARNCH_ID','')}"),
            'grad_year': int(row['GYEAR']) if str(row.get('GYEAR','')).isdigit() else row.get('GYEAR',''),
            'gpa': float(row['GRADE_POINT']) if pd.notnull(row.get('GRADE_POINT')) else 0.0
        })

learner_exp = {}
if not df_experience.empty:
    for _, row in df_experience.iterrows():
        lid = str(row['LEARNER_ID'])
        if lid not in learner_exp: learner_exp[lid] = []
        techtool = str(row.get('TECHTOOL',''))
        learner_exp[lid].append({
            'position': pos_map.get(str(row.get('POSITION_ID','')), f"Level {row.get('POSITION_ID','')}"),
            'company': str(row.get('COMPANY_NAME', '')),
            'start_date': str(row.get('START_DATE', '')),
            'end_date': str(row.get('END_DATE', '')),
            'description': str(row.get('DESCRIPTION', '')),
            'tech_tools': [t.strip() for t in techtool.split(',') if t.strip()] if techtool not in ['nan',''] else [],
            'province': str(row.get('PROVINCE', ''))
        })

learner_pref = {}
if not df_preference.empty:
    for _, row in df_preference.iterrows():
        lid = str(row['LEARNER_ID'])
        wt_id = str(int(row['WORKTYPE'])) if pd.notnull(row.get('WORKTYPE')) else '1'
        jt_id = str(int(row['JOBTYPE'])) if pd.notnull(row.get('JOBTYPE')) else '1'
        learner_pref[lid] = {
            'work_type': wt_map.get(wt_id, 'Onsite'),
            'job_type': jt_map.get(jt_id, 'Full-time'),
            'expected_salary': float(row.get('SALARY', 0)) if pd.notnull(row.get('SALARY')) else 0,
            'preferred_province': str(row.get('PROVINCE', ''))
        }

# ==========================================
# 5. AGGREGATE JOB DATA
# ==========================================
print("   - Aggregating job data...")

job_skills_map = {}
if not df_job_skill.empty:
    for _, row in df_job_skill.iterrows():
        jid = str(row['Job_post_id'])
        skill_id = str(row.get('skill_id', ''))
        skill_name = skill_id_map.get(skill_id, skill_id)
        is_cert = str(row.get('Is_certificate', 'False')).lower() in ['true','1']
        level = str(row.get('skill_level_id', ''))
        if jid not in job_skills_map: job_skills_map[jid] = {'skills': [], 'certs': []}
        if is_cert:
            job_skills_map[jid]['certs'].append({'name': skill_name, 'level': level})
        else:
            job_skills_map[jid]['skills'].append({'name': skill_name, 'level': level})

# ==========================================
# 6. COMPUTE BERT EMBEDDINGS
# ==========================================
print("   - Computing BERT embeddings...")

import datetime
MAX_JOBS = 500
MAX_USERS = 2000

# FIX 3: กรอง expired jobs ออก
_today = datetime.date.today()
if not df_job_post.empty:
    _open = df_job_post[df_job_post['job_post_status'] == 'open'].copy()
    _open['_end'] = pd.to_datetime(_open['end_date'], errors='coerce').dt.date
    # เอาเฉพาะที่ยังไม่หมดอายุ หรือ end_date เป็น null
    open_jobs = _open[_open['_end'].isna() | (_open['_end'] >= _today)].head(MAX_JOBS)
    if len(open_jobs) == 0:
        open_jobs = _open.head(MAX_JOBS)
    open_jobs = open_jobs.drop(columns=['_end'])
else:
    open_jobs = pd.DataFrame()

active_learners = df_learner[df_learner['status'] == 'active'].head(MAX_USERS) if not df_learner.empty else pd.DataFrame()
if len(active_learners) == 0: active_learners = df_learner.head(MAX_USERS)

job_matrix = None
user_matrix = None
job_records = []
user_records = []

if bert_model and not open_jobs.empty:
    job_texts = []
    for _, row in open_jobs.iterrows():
        jid = str(row['job_post_id'])
        j = job_skills_map.get(jid, {})
        skills_text = ' '.join([s['name'] for s in j.get('skills', [])])
        certs_text = ' '.join([c['name'] for c in j.get('certs', [])])
        title = str(row.get('position_name_en', ''))
        detail = str(row.get('job_detail', ''))[:300]
        job_texts.append(f"{title} {skills_text} {certs_text} {detail}")
        job_records.append(row.to_dict())
    job_matrix = bert_model.encode(job_texts, convert_to_tensor=True, show_progress_bar=False)
    print(f"   ✓ Job embeddings: {len(job_records)}")

# FIX 2: helper สร้าง user text ให้ consistent ทั้ง pre-compute และ inference
def build_user_text(title, about, skills_list, certs_list, exp_list):
    """สร้าง text สำหรับ BERT embedding ของ user — ใช้ทั้ง startup และ runtime"""
    exp_text = ' '.join([e.get('description', '')[:80] for e in exp_list[:2]])
    return f"{title} {str(about)[:200]} {' '.join(skills_list)} {' '.join(certs_list[:3])} {exp_text}"

if bert_model and not active_learners.empty:
    user_texts = []
    for _, row in active_learners.iterrows():
        lid = str(row['LEARNER_ID'])
        title = str(row.get('JOB_TITLE_EN', ''))
        about = str(row.get('ABOUT_ME', ''))
        skills_list_tmp = [s['name'] for s in learner_skills.get(lid, [])]
        certs_list_tmp = [c['course_name'] for c in learner_certs.get(lid, [])]
        exp_list_tmp = learner_exp.get(lid, [])
        user_texts.append(build_user_text(title, about, skills_list_tmp, certs_list_tmp, exp_list_tmp))
        user_records.append(row.to_dict())
    user_matrix = bert_model.encode(user_texts, convert_to_tensor=True, show_progress_bar=False)
    print(f"   ✓ User embeddings: {len(user_records)}")

print("✅ Server Ready! Port 5001")

# ==========================================
# 7. HELPER FUNCTIONS
# ==========================================

def scale_scores_minmax(scores_tensor, lo=40.0, hi=88.0):
    """FIX 5: Linear clamp แบบ honest — ไม่ stretch ทุก batch ให้ max=hi เสมอ.
    Cosine similarity [0,1] → [0,100] แล้ว clip ใน [lo, hi]"""
    scores = scores_tensor.cpu().numpy() if hasattr(scores_tensor, 'cpu') else np.array(scores_tensor)
    # Map cosine [0,1] → [0,100] แล้ว clip
    return np.clip(scores * 100.0, lo, hi)

def get_cert_match(user_cert_names, required_cert_names):
    if not required_cert_names or not user_cert_names:
        return []
    matched_reqs = []
    user_lower = [c.lower() for c in user_cert_names]
    for req in required_cert_names:
        req_lower = req.lower()
        found = any(req_lower in uc or uc in req_lower for uc in user_lower)
        if found:
            matched_reqs.append(req)
        elif bert_model:
            try:
                req_emb = bert_model.encode(req, convert_to_tensor=True)
                uc_embs = bert_model.encode(user_cert_names, convert_to_tensor=True)
                sims = util.cos_sim(req_emb, uc_embs)[0]
                if float(torch.max(sims)) >= 0.72:
                    matched_reqs.append(req)
            except: pass
    return matched_reqs

def get_skill_match_and_gap(user_skills, req_skills):
    if not req_skills: return [], [], 100, {}
    if not user_skills: return [], req_skills, 0, {req: "คุณยังไม่มีทักษะนี้หรือทักษะที่ใกล้เคียงเลย" for req in req_skills}
    user_lower = [s.lower() for s in user_skills]
    matched = []
    missing = []
    explanations = {}  # Store explanation for each requirement
    total_score = 0.0
    
    # Pre-encode user skills for BERT
    user_embs = None
    if bert_model:
        try:
            user_embs = bert_model.encode(user_skills, convert_to_tensor=True)
        except: pass

    for rs in req_skills:
        rs_lower = rs.lower()
        best_sim = 0.0
        best_match_user_skill = None
        
        # 1. Substring Match
        for i, u in enumerate(user_lower):
            if rs_lower in u or u in rs_lower:
                best_sim = 1.0
                best_match_user_skill = user_skills[i]
                break
        
        # 2. BERT Fuzzy match
        if best_sim < 1.0 and user_embs is not None:
            try:
                rs_emb = bert_model.encode(rs, convert_to_tensor=True)
                sims = util.cos_sim(rs_emb, user_embs)[0]
                
                # Boost known tech pairs slightly if BERT underestimates
                augmented_sims = sims.clone()
                for i, u in enumerate(user_lower):
                    sim_val = float(sims[i])
                    
                    if "backend" in rs_lower and any(tech in u for tech in ["python", "java", "node", "php", "go", "ruby"]):
                        sim_val = max(sim_val, 0.75)
                    elif "frontend" in rs_lower and any(tech in u for tech in ["react", "vue", "angular", "javascript", "html", "css"]):
                        sim_val = max(sim_val, 0.75)
                    elif ("database" in rs_lower or "sql" in rs_lower) and any(tech in u for tech in ["mysql", "postgresql", "oracle", "mongodb", "sql"]):
                        sim_val = max(sim_val, 0.85)
                    elif "data" in rs_lower and any(tech in u for tech in ["python", "r ", "sql", "tableau", "power bi", "pandas", "excel"]):
                        sim_val = max(sim_val, 0.65)
                        
                    # Explicit penalties to prevent cross-domain pollution
                    if "data engineering" in rs_lower and any(tech in u for tech in ["java", "javascript", "html", "css", "react"]):
                        sim_val = min(sim_val, 0.40)
                    if "sql" in rs_lower and any(tech in u for tech in ["java", "c#", "go", "ruby", "javascript", "html", "css", "react"]):
                        sim_val = min(sim_val, 0.40)
                        
                    augmented_sims[i] = sim_val

                max_idx = int(torch.argmax(augmented_sims))
                max_sim = float(augmented_sims[max_idx])

                if max_sim > best_sim:
                    best_sim = max_sim
                    best_match_user_skill = user_skills[max_idx]
            except: pass
            
        if best_sim >= 0.82:
            matched.append(rs)
            total_score += 1.0
        elif best_sim >= 0.70:
            missing.append(rs)
            total_score += best_sim
            explanations[rs] = f"คุณมีทักษะใกล้เคียงคือ '{best_match_user_skill}'"
        elif best_sim >= 0.45:
            missing.append(rs)
            # Give very minor partial credit for baseline tech crossover, but don't explain it as a similar skill
            total_score += best_sim * 0.5
            explanations[rs] = "คุณยังไม่มีทักษะนี้โดยตรง"
        else:
            missing.append(rs)
            explanations[rs] = "คุณยังไม่มีทักษะนี้หรือทักษะที่ใกล้เคียงเลย"

    skill_score = int(round((total_score / len(req_skills)) * 100))
    skill_score = min(100, max(0, skill_score))

    return list(dict.fromkeys(matched)), list(dict.fromkeys(missing)), skill_score, explanations

def calc_exp_years(exp_list):
    total = 0.0
    for e in exp_list:
        try:
            s = pd.to_datetime(e.get('start_date',''), errors='coerce')
            en = pd.to_datetime(e.get('end_date',''), errors='coerce')
            if pd.notnull(s) and pd.notnull(en):
                total += max(0, (en - s).days / 365)
        except: pass
    return round(total, 1)

COURSE_DB = {
    "python": "Python for Data Science",
    "sql": "Advanced SQL for Analytics",
    "machine learning": "Machine Learning A-Z",
    "tableau": "Tableau Desktop Specialist",
    "power bi": "Microsoft Power BI Masterclass",
    "oracle": "Oracle Database Administrator Certified",
    "aws": "AWS Certified Solutions Architect",
    "gcp": "Google Cloud Professional Data Engineer",
    "azure": "Microsoft Azure Fundamentals",
    "docker": "Docker & Kubernetes Essentials",
    "kubernetes": "Kubernetes for Developers",
    "react": "React - The Complete Guide",
    "frontend": "Frontend Development with React",
    "backend": "Backend Development with Node.js",
    "data engineering": "Data Engineering with Apache Spark",
    "cybersecurity": "Cybersecurity Fundamentals",
    "devops": "DevOps & CI/CD Pipelines",
    "agile": "Agile & Scrum Fundamentals",
    "generative ai": "Generative AI & LLM Ops",
    "network": "Network Engineering Fundamentals",
    "security": "Security Architecture Design",
}

def recommend_courses(missing_skills):
    courses = []
    for skill in missing_skills[:5]:
        sl = skill.lower()
        course = next((v for k, v in COURSE_DB.items() if k in sl), f"Introduction to {skill}")
        courses.append({"skill": skill, "course_name": course, "platform": "L2E Academy"})
    return courses

def get_logo(company_id):
    comp = company_map.get(str(company_id), {})
    name = comp.get('name_en') or comp.get('name_th') or company_id
    known = {
        'google': 'https://logo.clearbit.com/google.com',
        'scg': 'https://logo.clearbit.com/scg.co.th',
        'ptt': 'https://logo.clearbit.com/pttplc.com',
        'scb': 'https://logo.clearbit.com/scb.co.th',
        'kasikorn': 'https://logo.clearbit.com/kasikornbank.com',
        'kbank': 'https://logo.clearbit.com/kasikornbank.com',
        'line': 'https://logo.clearbit.com/line.me',
        'true': 'https://logo.clearbit.com/truecorp.co.th',
        'ais': 'https://logo.clearbit.com/ais.th',
        'dtac': 'https://logo.clearbit.com/dtac.co.th',
    }
    for k, v in known.items():
        if k in (name or '').lower():
            return v
    return f"https://ui-avatars.com/api/?name={str(name).replace(' ','+')}&background=0a66c2&color=fff&bold=true&size=128"

def build_candidate_profile(lid, learner_row=None):
    """Build a full candidate profile dict from all tables."""
    if learner_row is None:
        rows = df_learner[df_learner['LEARNER_ID'] == lid]
        if rows.empty: return None
        learner_row = rows.iloc[0]
    r = learner_row
    pref = learner_pref.get(lid, {})
    # FIX 4: ใช้ province จาก preference เป็นหลัก (ผู้ใช้เลือกเอง)
    # fallback ไป prov_map
    u_prov = pref.get('preferred_province', '')
    if not u_prov:
        sub_id = str(r.get('SUBDISTRICT_ID', ''))
        u_prov = prov_map.get(sub_id, '')
    skills_list = [s['name'] for s in learner_skills.get(lid, [])]
    certs_list = [c['course_name'] for c in learner_certs.get(lid, [])]
    exp_list = learner_exp.get(lid, [])
    edu_list = learner_edu.get(lid, [])
    return {
        'user_id': lid,
        'name': f"{r.get('FNAME_TH','')} {r.get('LANME_TH','')}".strip(),
        'name_en': f"{r.get('FNAME_ENG','')} {r.get('LNAME_ENG','')}".strip(),
        'email': str(r.get('EMAIL','')),
        'current_role': str(r.get('JOB_TITLE_EN', r.get('JOB_TITLE_TH',''))),
        'about_me': str(r.get('ABOUT_ME','')),
        'skills': skills_list,
        'skills_detail': learner_skills.get(lid, []),
        'certificates': certs_list,
        'certificates_detail': learner_certs.get(lid, []),
        'education': edu_list,
        'experience': exp_list,
        'exp_years_total': calc_exp_years(exp_list),
        'expected_salary': pref.get('expected_salary', 0),
        'work_style_pref': pref.get('work_type', 'Onsite'),
        'job_type_pref': pref.get('job_type', 'Full-time'),
        'location': u_prov,
        'status': str(r.get('status','active')),
        'language': str(r.get('LANGUAGE','1')),
        'profile_completeness': calc_completeness(lid, skills_list, pref)
    }

def calc_completeness(lid, skills_list, pref):
    score = 0
    checks = {
        'job_title': True,  # always from DB
        'about_me': True,   # always from DB
        'skills': len(skills_list) >= 2,
        'salary': pref.get('expected_salary', 0) > 0,
        'certs': len(learner_certs.get(lid, [])) > 0,
        'education': len(learner_edu.get(lid, [])) > 0,
        'experience': len(learner_exp.get(lid, [])) > 0,
    }
    score = sum(1 for v in checks.values() if v)
    return round((score / len(checks)) * 100)

def is_profile_complete(lid, skills_list, pref):
    return len(skills_list) >= 2 and pref.get('expected_salary', 0) > 0

def match_jobs_for_user(lid, user_emb=None, extra_certs=None):
    """Run matching for a user and return top job results."""
    if job_matrix is None or not job_records:
        return []
    pref = learner_pref.get(lid, {})
    user_prov = pref.get('preferred_province', '')
    if not user_prov and not df_learner.empty:
        rows = df_learner[df_learner['LEARNER_ID'] == lid]
        if not rows.empty:
            sub_id = str(rows.iloc[0].get('SUBDISTRICT_ID', ''))
            user_prov = prov_map.get(sub_id, '')

    skills_list = [s['name'] for s in learner_skills.get(lid, [])]
    user_cert_names = extra_certs or [c['course_name'] for c in learner_certs.get(lid, [])]
    exp_list = learner_exp.get(lid, [])
    expected_salary = pref.get('expected_salary', 0)

    if user_emb is None:
        rows = df_learner[df_learner['LEARNER_ID'] == lid]
        if not rows.empty:
            r = rows.iloc[0]
            title = str(r.get('JOB_TITLE_EN',''))
            about = str(r.get('ABOUT_ME',''))
            # FIX 2: ใช้ build_user_text() เดียวกันกับตอน pre-compute startup
            user_emb = bert_model.encode(
                build_user_text(title, about, skills_list, user_cert_names, exp_list),
                convert_to_tensor=True
            )

    cos_scores = util.cos_sim(user_emb, job_matrix)[0]
    top_k = min(50, len(job_records))
    top_results = torch.topk(cos_scores, k=top_k)
    scaled = scale_scores_minmax(top_results.values, 40.0, 88.0)

    # Pre-encode user's desired role for title-level matching
    user_role_text = str(pref.get('desired_role', '')) or str(pref.get('JOB_TITLE_EN', ''))
    if not user_role_text and not df_learner.empty:
        rows = df_learner[df_learner['LEARNER_ID'] == lid]
        if not rows.empty:
            user_role_text = str(rows.iloc[0].get('JOB_TITLE_EN', ''))
    user_role_emb = bert_model.encode(user_role_text, convert_to_tensor=True) if user_role_text else None

    results = []
    for i, idx in enumerate(top_results.indices):
        job = job_records[idx.item()]
        jid = str(job.get('job_post_id',''))
        cid = str(job.get('company_id',''))
        comp = company_map.get(cid, {})
        j_data = job_skills_map.get(jid, {})
        req_skills = [s['name'] for s in j_data.get('skills', [])]
        req_certs = [c['name'] for c in j_data.get('certs', [])]

        semantic = float(scaled[i])
        sal_min = float(job.get('income_min', 0) or 0)
        sal_max = float(job.get('income_max', 0) or 0)
        
        # Deviation calculation for budget
        budget_score = 100.0
        if expected_salary > 0:
            if sal_max > 0:
                if expected_salary <= sal_max:
                    budget_score = 100.0
                else:
                    budget_score = (sal_max / expected_salary) * 100.0
            elif sal_min > 0:
                if expected_salary <= sal_min:
                    budget_score = 100.0
                else:
                    budget_score = (sal_min / expected_salary) * 100.0
        budget_score = round(budget_score, 1)

        matched_skills, gap, skill_score, matched_explanations = get_skill_match_and_gap(skills_list, req_skills)

        # 1. Get unique and non-empty required certs
        req_certs = sorted(list(set([c['name'] for c in j_data.get('certs', []) if c.get('name')])))
        
        # 2. Match certs
        cert_matched = get_cert_match(user_cert_names, req_certs)
        cert_missing = [req for req in req_certs if req not in cert_matched]
        
        # 3. Final Certification Score
        if not req_certs:
            cert_score = 100
        else:
            cert_score = int((len(cert_matched) / len(req_certs)) * 100)

        # ---- Title-level role match ----
        job_title_text = str(job.get('position_name_en', job.get('position_name_th', ''))).lower()
        u_role_low = user_role_text.lower()
        
        # 1. Base Semantic Similarity
        title_match_score = 0.0
        raw_title_sim = 0.60
        if user_role_emb is not None and job_title_text:
            job_title_emb = bert_model.encode(job_title_text, convert_to_tensor=True)
            raw_title_sim = float(util.cos_sim(user_role_emb, job_title_emb)[0][0])
            
            # Relaxed mapping for fine-tuned range [0.75, 1.0]
            # Baseline: 0.75 -> 0%, 0.90 -> 70%, 1.0 -> 100%
            if raw_title_sim >= 0.95:
                title_match_score = 100.0 * ((raw_title_sim - 0.75) / 0.25)
            elif raw_title_sim >= 0.85:
                # Related bucket: Linear map 0.85-0.95 -> 60-90%
                title_match_score = 60.0 + (raw_title_sim - 0.85) / 0.10 * 30.0
            else:
                # Cross-domain: steep drop below 0.85
                title_match_score = max(5.0, 60.0 * ((max(0, raw_title_sim - 0.75)) / 0.10) ** 2)

        # 2. Tech Synonym Boost (Programmer, Developer, Software Engineer, Web Dev)
        tech_synonyms = ['programmer', 'developer', 'software engineer', 'software developer', 'web developer', 'full stack']
        if any(s in job_title_text for s in tech_synonyms) and any(s in u_role_low for s in tech_synonyms):
            # Boost score if they are both in the tech "developer" bucket but BERT missed the connection
            title_match_score = max(title_match_score, 85.0 if raw_title_sim > 0.80 else 70.0)
            raw_title_sim = max(raw_title_sim, 0.90) # Fake a higher sim to avoid hard-caps

        title_match_score = round(min(title_match_score, 100.0), 1)

        # Composite formula
        # weights: title(20%) + skill(20%) + budget(20%) + cert(20%) + semantic(20%)
        raw_final = (
            title_match_score * 0.20 +
            skill_score       * 0.20 +
            budget_score      * 0.20 +
            cert_score        * 0.20 +
            semantic          * 0.20
        )

        # 3-Tier Hard Cap
        if raw_title_sim < 0.80:
            # cross-domain: Civil Engineer vs Programmer
            raw_final = min(raw_final, 65.0)
        elif raw_title_sim < 0.88:
            # related: System Engineer vs Programmer
            raw_final = min(raw_final, 85.0)
        elif raw_title_sim < 0.92:
            # highly related
            raw_final = min(raw_final, 95.0)

        # Work style matching penalty
        work_type_penalty = 0.0
        job_work_type = wt_map.get(str(job.get('Work_location_type_id','1')), 'Onsite')
        user_work_type = pref.get('work_type', 'Onsite')
        if user_work_type.lower() != 'ไม่ระบุ' and job_work_type.lower() != user_work_type.lower():
            if 'onsite' in user_work_type.lower() and 'remote' in job_work_type.lower():
                work_type_penalty = 20.0
            elif 'remote' in user_work_type.lower() and 'onsite' in job_work_type.lower():
                work_type_penalty = 20.0
            else:
                work_type_penalty = 10.0
                
        # Location matching penalty
        location_fit = 100.0
        prov_id = str(job.get('province_id',''))
        job_location = prov_id_map.get(prov_id, prov_id)
        
        if user_prov and job_location:
            if 'remote' not in job_work_type.lower():
                u_prov_clean = user_prov.strip().lower()
                j_prov_clean = job_location.strip().lower()
                
                if u_prov_clean != 'ไม่ระบุ' and j_prov_clean != 'ไม่ระบุ' and u_prov_clean != j_prov_clean:
                    location_fit = 0.0
                    work_type_penalty += 25.0
        
        final = round(min(max(raw_final - work_type_penalty, 10.0), 99.0), 1)

        show_salary = str(job.get('is_display_income','True')).lower() == 'true'
        if show_salary and sal_min>0 and sal_max>0:
            salary_text = f"{int(sal_min):,} - {int(sal_max):,}"
        elif sal_max>0:
            salary_text = f"สูงสุด {int(sal_max):,}"
        else:
            salary_text = "Negotiable"

        prov_id = str(job.get('province_id',''))
        location = prov_id_map.get(prov_id, prov_id)
        wt_id = str(job.get('Work_location_type_id','1'))

        results.append({
            'job_id': jid,
            'title': str(job.get('position_name_en', job.get('position_name_th',''))),
            'company': comp.get('name_th') or comp.get('name_en') or cid,
            'company_id': cid,
            'company_about': comp.get('about',''),
            'logo_url': get_logo(cid),
            'location': location,
            'work_type': wt_map.get(wt_id, 'Onsite'),
            'job_type': str(job.get('job_type','IT')),
            'salary_text': salary_text,
            'salary_min': sal_min,
            'salary_max': sal_max,
            'description': str(job.get('job_detail','')),
            'summary': str(job.get('job_summary','')),
            'open_date': str(job.get('open_date','')),
            'end_date': str(job.get('end_date','')),
            'match_score': final,
            'required_skills': req_skills,
            'required_certs': req_certs,
            'matched_skills': matched_skills,
            'missing_skills': gap,
            'cert_matched': cert_matched,
            'cert_missing': cert_missing,
            'rec_courses': recommend_courses(gap + cert_missing),
            'score_breakdown': {
                'semantic': round(semantic, 1),
                'title': round(title_match_score, 1),
                'skill': skill_score,
                'budget_fit': round(budget_score, 1),
                'cert': cert_score,
                'location_fit': round(location_fit, 1)
            },
            'skill_explanations': matched_explanations
        })
    return sorted(results, key=lambda x: x['match_score'], reverse=True)

# ==========================================
# 8. API ENDPOINTS
# ==========================================

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        uid = str(data.get('id','')).strip()

        # Override from saved profile
        if uid in profile_overrides:
            ov = profile_overrides[uid]
            complete = len(ov.get('skills',[])) >= 2 and ov.get('expected_salary',0) > 0
            return jsonify({'role': 'candidate', 'profile_complete': complete, 'data': ov})

        # Learner
        if not df_learner.empty:
            row = df_learner[df_learner['LEARNER_ID'] == uid]
            if not row.empty:
                profile = build_candidate_profile(uid, row.iloc[0])
                complete = is_profile_complete(uid, profile['skills'], learner_pref.get(uid,{}))
                return jsonify({'role': 'candidate', 'profile_complete': complete, 'data': profile})

        # Employer (job post)
        if not df_job_post.empty:
            row = df_job_post[df_job_post['job_post_id'] == uid]
            if not row.empty:
                r = row.iloc[0]
                jid = str(r['job_post_id'])
                cid = str(r.get('company_id',''))
                comp = company_map.get(cid, {})
                j_data = job_skills_map.get(jid, {})
                prov_id = str(r.get('province_id',''))
                province = prov_id_map.get(prov_id, prov_id)
                wt_id = str(r.get('Work_location_type_id','1'))
                return jsonify({'role': 'employer', 'profile_complete': True, 'data': {
                    'job_post_id': jid,
                    'job_id': jid,
                    'title': str(r.get('position_name_en', r.get('position_name_th',''))),
                    'company_id': cid,
                    'company_name': comp.get('name_th', comp.get('name_en', cid)),
                    'company_name_en': comp.get('name_en',''),
                    'company_about': comp.get('about',''),
                    'logo_url': get_logo(cid),
                    'location': province,
                    'work_style': wt_map.get(wt_id,'Onsite'),
                    'job_type': str(r.get('job_type','IT')),
                    'job_sub_type': str(r.get('job_sub_type','')),
                    'salary_min': float(r.get('income_min',0) or 0),
                    'salary_max': float(r.get('income_max',0) or 0),
                    'description': str(r.get('job_detail','')),
                    'summary': str(r.get('job_summary','')),
                    'required_skills': [s['name'] for s in j_data.get('skills',[])],
                    'required_certs': [c['name'] for c in j_data.get('certs',[])],
                    'open_date': str(r.get('open_date','')),
                    'end_date': str(r.get('end_date','')),
                    'status': str(r.get('job_post_status','')),
                }})

        return jsonify({'error': 'ID not found. ใช้ LNR-202600001 หรือ JOB-202600001'}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/check_profile', methods=['GET'])
def check_profile():
    uid = request.args.get('id','').strip()
    if uid in profile_overrides:
        ov = profile_overrides[uid]
        complete = len(ov.get('skills',[])) >= 2 and ov.get('expected_salary',0) > 0
        return jsonify({'complete': complete, 'profile': ov})
    row = df_learner[df_learner['LEARNER_ID'] == uid] if not df_learner.empty else pd.DataFrame()
    if row.empty:
        return jsonify({'complete': False, 'profile': None})
    profile = build_candidate_profile(uid, row.iloc[0])
    complete = is_profile_complete(uid, profile['skills'], learner_pref.get(uid,{}))
    return jsonify({'complete': complete, 'profile': profile})


@app.route('/api/save_profile', methods=['POST'])
def save_profile():
    try:
        data = request.json
        uid = str(data.get('user_id','')).strip()
        if not uid:
            return jsonify({'error': 'user_id required'}), 400
        # Merge with existing DB data
        existing = {}
        row = df_learner[df_learner['LEARNER_ID'] == uid] if not df_learner.empty else pd.DataFrame()
        if not row.empty:
            existing = build_candidate_profile(uid, row.iloc[0]) or {}
        merged = {**existing, **data}
        profile_overrides[uid] = merged
        return jsonify({'success': True, 'profile': merged})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/match_jobs_instant', methods=['POST'])
def match_jobs_instant():
    """Match jobs using stored user data — no need to pass full profile again."""
    try:
        data = request.json
        uid = str(data.get('user_id',''))
        limit = int(data.get('limit', 20))
        # Get profile from override or DB
        if uid in profile_overrides:
            profile = profile_overrides[uid]
        else:
            row = df_learner[df_learner['LEARNER_ID'] == uid] if not df_learner.empty else pd.DataFrame()
            if row.empty:
                return jsonify({'jobs': [], 'error': 'User not found'})
            profile = build_candidate_profile(uid, row.iloc[0])

        if not bert_model or job_matrix is None:
            return jsonify({'jobs': [], 'error': 'BERT model not loaded'})

        # FIX 2: ใช้ build_user_text() เดียวกันกับ pre-compute
        skills = profile.get('skills', [])
        certs = profile.get('certificates', [])
        title = profile.get('current_role', '')
        about = profile.get('about_me', '')
        exp_list = profile.get('experience', [])
        text = build_user_text(title, about, skills, certs, exp_list)
        user_emb = bert_model.encode(text, convert_to_tensor=True)
        # Inject pref so title_match can use it
        if uid not in learner_pref:
            learner_pref[uid] = {}
        learner_pref[uid]['desired_role'] = title
        learner_pref[uid]['expected_salary'] = float(profile.get('expected_salary', 0) or 0)
        jobs = match_jobs_for_user(uid, user_emb=user_emb, extra_certs=certs)
        return jsonify({'jobs': jobs[:limit]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/match_jobs', methods=['POST'])
def match_jobs():
    """FIX 6: Match jobs from full form submission — ใช้ match_jobs_for_user() เดียวกันกับ match_jobs_instant."""
    try:
        data = request.json
        if not bert_model or job_matrix is None:
            return jsonify({'jobs': []})

        # รับข้อมูลจาก form แล้วสร้าง temp profile ใน learner_pref
        role = data.get('interested_role', data.get('current_role', ''))
        skills = data.get('skills', [])
        about_me = data.get('about_me', '')
        user_certs = data.get('certificates', [])
        expected_salary = float(data.get('expected_salary', 0) or 0)
        exp_list = [{'description': e} for e in data.get('work_experiences', []) if e]
        uid = str(data.get('user_id', '_form_user_'))

        # FIX 2: ใช้ build_user_text() เดียวกัน
        text = build_user_text(role, about_me, skills, user_certs, exp_list)
        user_emb = bert_model.encode(text, convert_to_tensor=True)

        # ใส่ temp pref เพื่อให้ match_jobs_for_user() คำนวณ salary/title ได้
        learner_pref[uid] = {
            'desired_role': role,
            'expected_salary': expected_salary,
            'work_type': data.get('work_type', 'Onsite'),
            'job_type': data.get('job_type', 'Full-time'),
        }
        learner_skills[uid] = [{'name': s, 'level': ''} for s in skills]
        learner_certs[uid] = [{'course_name': c} for c in user_certs]
        learner_exp[uid] = exp_list

        # FIX 6: ใช้ match_jobs_for_user() เดียวกัน — formula เหมือนกันแน่นอน
        jobs = match_jobs_for_user(uid, user_emb=user_emb, extra_certs=user_certs)
        limit = int(data.get('limit', 50))
        return jsonify({'jobs': jobs[:limit]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/job_detail/<job_id>', methods=['GET'])
def job_detail(job_id):
    try:
        row = df_job_post[df_job_post['job_post_id'] == job_id] if not df_job_post.empty else pd.DataFrame()
        if row.empty:
            return jsonify({'error': 'Job not found'}), 404
        r = row.iloc[0]
        jid = str(r['job_post_id'])
        cid = str(r.get('company_id',''))
        comp = company_map.get(cid, {})
        j_data = job_skills_map.get(jid, {})
        prov_id = str(r.get('province_id',''))
        wt_id = str(r.get('Work_location_type_id','1'))
        sal_min = float(r.get('income_min',0) or 0)
        sal_max = float(r.get('income_max',0) or 0)
        return jsonify({
            'job_id': jid, 'title': str(r.get('position_name_en',r.get('position_name_th',''))),
            'company': comp.get('name_th') or cid, 'company_name_en': comp.get('name_en',''),
            'company_about': comp.get('about',''), 'company_website': comp.get('website',''),
            'company_size': comp.get('size',''), 'company_specialty': comp.get('specialty',''),
            'logo_url': get_logo(cid), 'location': prov_id_map.get(prov_id, prov_id),
            'work_type': wt_map.get(wt_id,'Onsite'), 'job_type': str(r.get('job_type','')),
            'salary_min': sal_min, 'salary_max': sal_max,
            'salary_text': f"฿{int(sal_min):,} - {int(sal_max):,}" if sal_min>0 else "Negotiable",
            'description': str(r.get('job_detail','')), 'summary': str(r.get('job_summary','')),
            'working_time': str(r.get('Working_time','')),
            'open_date': str(r.get('open_date','')), 'end_date': str(r.get('end_date','')),
            'required_skills': [{'name': s['name'], 'level': s['level']} for s in j_data.get('skills',[])],
            'required_certs': [{'name': c['name'], 'level': c['level']} for c in j_data.get('certs',[])]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/skill_gap', methods=['POST'])
def skill_gap():
    try:
        data = request.json
        user_skills = data.get('user_skills', [])
        user_certs = data.get('user_certs', [])
        job_id = data.get('job_id','')
        j_data = job_skills_map.get(job_id, {})
        req_skills = [s['name'] for s in j_data.get('skills',[])]
        req_certs = [c['name'] for c in j_data.get('certs',[])]

        matched_skills, gap, skill_score, exps = get_skill_match_and_gap(user_skills, req_skills)
        cert_matched = get_cert_match(user_certs, req_certs)
        cert_missing = [c for c in req_certs if c not in cert_matched]

        return jsonify({
            'matched_skills': matched_skills,
            'missing_skills': gap,
            'cert_matched': cert_matched,
            'cert_missing': cert_missing,
            'skill_score': skill_score,
            'cert_score': round((len(cert_matched)/max(len(req_certs),1))*100) if req_certs else 100,
            'rec_courses': recommend_courses(gap + cert_missing),
            'skill_explanations': exps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/course_recommend', methods=['POST'])
def course_recommend():
    data = request.json
    missing = data.get('missing_skills', []) + data.get('missing_certs', [])
    return jsonify({'recommendations': recommend_courses(missing)})


@app.route('/api/match_candidates', methods=['POST'])
def match_candidates():
    try:
        if not user_records or user_matrix is None:
            return jsonify([])
        data = request.json
        jd_text = data.get('jd_text','')
        job_title = data.get('job_title', '')
        required_skills = data.get('required_skills', [])
        budget_min = float(data.get('budget_min', 40000) or 40000)
        budget_max = float(data.get('budget_max', 80000) or 80000)
        required_certs = data.get('required_certs', [])
        job_location_pref = data.get('job_location', 'ไม่ระบุ')
        job_work_style_pref = data.get('job_work_style', 'Onsite')
        
        if isinstance(required_certs, str):
            required_certs = [c.strip() for c in required_certs.split(',') if c.strip()]
            
        job_title_emb = bert_model.encode(job_title, convert_to_tensor=True) if job_title else None

        jd_emb = bert_model.encode(jd_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(jd_emb, user_matrix)[0]
        top_k = min(50, len(user_records))
        top_results = torch.topk(cos_scores, k=top_k)
        scaled = scale_scores_minmax(top_results.values, 50.0, 90.0)

        pool = []
        for i, idx in enumerate(top_results.indices):
            user = user_records[idx.item()]
            lid = str(user.get('LEARNER_ID',''))
            sub_id = str(user.get('SUBDISTRICT_ID',''))
            semantic = float(scaled[i])
            pref = learner_pref.get(lid, {})
            exp_sal = float(pref.get('expected_salary', 0))
            
            budget_score = 100.0
            if exp_sal > 0:
                if budget_max > 0:
                    if exp_sal <= budget_max:
                        budget_score = 100.0
                    else:
                        budget_score = (budget_max / exp_sal) * 100.0
            budget_score = round(budget_score, 1)

            user_cert_names = [c['course_name'] for c in learner_certs.get(lid,[])]
            cert_matched = get_cert_match(user_cert_names, required_certs)
            cert_score = round((len(cert_matched)/max(len(required_certs),1))*100) if required_certs else 100
            
            # Skill Match
            skills_list = [s['name'] for s in learner_skills.get(lid,[])]
            matched_skills, gap, skill_score, matched_explanations = get_skill_match_and_gap(skills_list, required_skills)
            
            # Title Match
            user_title_text = str(user.get('JOB_TITLE_EN', user.get('JOB_TITLE_TH',''))).lower()
            title_match_score = 0.0
            raw_title_sim = 0.60
            if job_title_emb is not None and user_title_text:
                user_role_emb = bert_model.encode(user_title_text, convert_to_tensor=True)
                raw_title_sim = float(util.cos_sim(job_title_emb, user_role_emb)[0][0])
                if raw_title_sim >= 0.95:
                    title_match_score = 100.0
                elif raw_title_sim >= 0.85:
                    title_match_score = 60.0 + (raw_title_sim - 0.85) / 0.10 * 30.0
                else:
                    title_match_score = max(5.0, 60.0 * ((max(0, raw_title_sim - 0.75)) / 0.10) ** 2)
            
            # Tech Synonym Boost
            tech_synonyms = ['programmer', 'developer', 'software engineer', 'software developer', 'web developer', 'full stack']
            if any(s in job_title.lower() for s in tech_synonyms) and any(s in user_title_text for s in tech_synonyms):
                title_match_score = max(title_match_score, 85.0 if raw_title_sim > 0.80 else 70.0)

            title_match_score = round(min(title_match_score, 100.0), 1)

            # Combined Score Formula Matching The Dashboard (20% sem, 25% title, 15% budget, 20% cert, 20% skill)
            raw_final = (title_match_score * 0.25 + skill_score * 0.20 + budget_score * 0.15 + cert_score * 0.20 + semantic * 0.20)
            
            # 3-Tier Hard Cap
            if raw_title_sim < 0.80:
                raw_final = min(raw_final, 65.0)
            elif raw_title_sim < 0.88:
                raw_final = min(raw_final, 85.0)
            elif raw_title_sim < 0.92:
                raw_final = min(raw_final, 95.0)
                
            # Work style matching penalty
            work_type_penalty = 0.0
            user_work_type = pref.get('work_type', 'Onsite')
            if user_work_type.lower() != 'ไม่ระบุ' and job_work_style_pref.lower() != 'ไม่ระบุ' and job_work_style_pref.lower() != user_work_type.lower():
                if 'onsite' in user_work_type.lower() and 'remote' in job_work_style_pref.lower():
                    work_type_penalty = 20.0
                elif 'remote' in user_work_type.lower() and 'onsite' in job_work_style_pref.lower():
                    work_type_penalty = 20.0
                else:
                    work_type_penalty = 10.0
                    
            # Location matching penalty
            location_fit = 100.0
            user_prov = pref.get('preferred_province') or prov_map.get(sub_id, '')
            if 'remote' not in job_work_style_pref.lower():
                if user_prov and job_location_pref and job_location_pref.lower() != 'ไม่ระบุ' and user_prov.lower() != 'ไม่ระบุ':
                    if user_prov.strip() != job_location_pref.strip():
                        location_fit = 75.0
            location_penalty = 100.0 - location_fit
                
            fitness = round(min(max(raw_final - work_type_penalty - location_penalty, 10.0), 99.0), 1)

            pool.append({
                'user_id': lid, 'name': f"{user.get('FNAME_TH','')} {user.get('LANME_TH','')}".strip(),
                'name_en': f"{user.get('FNAME_ENG','')} {user.get('LNAME_ENG','')}".strip(),
                'title': str(user.get('JOB_TITLE_EN', user.get('JOB_TITLE_TH',''))),
                'expected_salary': exp_sal, 'fitness': fitness, 'match_score': round(semantic, 1),
                'about_me': str(user.get('ABOUT_ME','')),
                'skills': skills_list, 'skills_detail': learner_skills.get(lid,[]),
                'certificates': user_cert_names, 'cert_matched': cert_matched,
                'cert_missing': [c for c in required_certs if c not in cert_matched],
                'education': learner_edu.get(lid,[]), 'experience': learner_exp.get(lid,[]),
                'exp_years_total': calc_exp_years(learner_exp.get(lid,[])),
                'work_style_pref': pref.get('work_type','Onsite'),
                'location': pref.get('preferred_province') or prov_map.get(sub_id, ''),
                'email': user.get('EMAIL') or user.get('EMAIL_CONTECT', ''),
                'phone': user.get('TELEPHONE_NO', ''),
                'score_breakdown': {'semantic': round(semantic,1), 'title': title_match_score, 'skill': skill_score, 'budget_fit': round(budget_score,1), 'cert': cert_score}
            })
        return jsonify(sorted(pool, key=lambda x: x['fitness'], reverse=True)[:50])
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/market_stats', methods=['GET'])
def market_stats():
    try:
        total_jobs = len(df_job_post)
        total_users = len(df_learner)
        active_users = int(df_learner[df_learner['status']=='active'].shape[0]) if not df_learner.empty else 0
        open_jobs_count = int(df_job_post[df_job_post['job_post_status']=='open'].shape[0]) if not df_job_post.empty else 0

        # Top job titles by posting count
        top_jobs = []
        if not df_job_post.empty:
            df_job_post['avg_sal'] = (pd.to_numeric(df_job_post['income_min'],errors='coerce').fillna(0) +
                                       pd.to_numeric(df_job_post['income_max'],errors='coerce').fillna(0)) / 2
            jg = df_job_post.groupby('position_name_en').agg(
                job_count=('job_post_id','count'),
                avg_salary=('avg_sal','mean')
            ).reset_index().sort_values('job_count', ascending=False).head(12)
            # Candidate interest count from learner
            role_counts = df_learner['JOB_TITLE_EN'].value_counts().to_dict() if not df_learner.empty else {}
            for _, row in jg.iterrows():
                title = row['position_name_en']
                cand_count = role_counts.get(title, 0)
                ratio = round(cand_count / max(row['job_count'],1), 1)
                top_jobs.append({
                    'title': title, 'job_count': int(row['job_count']),
                    'avg_salary': int(row['avg_salary']),
                    'candidate_count': int(cand_count), 'competition': ratio
                })

        # Top skills in demand
        top_skills = []
        if not df_job_skill.empty and not df_acad_skill.empty:
            skill_counts = df_job_skill[df_job_skill['Is_certificate'].astype(str).str.lower()!='true']['skill_id'].value_counts().head(10)
            for skill_id, count in skill_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                top_skills.append({'skill': name, 'count': int(count)})

        # Top certs in demand
        top_certs = []
        if not df_job_skill.empty:
            cert_counts = df_job_skill[df_job_skill['Is_certificate'].astype(str).str.lower()=='true']['skill_id'].value_counts().head(8)
            for skill_id, count in cert_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                top_certs.append({'cert': name, 'count': int(count)})

        # Top provinces
        top_provinces = []
        if not df_job_post.empty:
            prov_counts = df_job_post['province_id'].value_counts().head(8)
            for prov_id, count in prov_counts.items():
                name = prov_id_map.get(str(prov_id), str(prov_id))
                top_provinces.append({'province': name, 'count': int(count)})

        # Salary by category
        salary_by_cat = []
        if not df_job_post.empty:
            catg = df_job_post.groupby('job_type').agg(avg_min=('income_min','mean'), avg_max=('income_max','mean')).reset_index()
            for _, row in catg.iterrows():
                salary_by_cat.append({'category': row['job_type'], 'avg_min': int(row['avg_min'] or 0), 'avg_max': int(row['avg_max'] or 0)})

        return jsonify({
            'total_jobs': total_jobs, 'total_users': total_users,
            'active_users': active_users, 'open_jobs': open_jobs_count,
            'top_jobs': top_jobs, 'top_skills': top_skills,
            'top_certs': top_certs, 'top_provinces': top_provinces,
            'salary_by_category': salary_by_cat
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample_ids', methods=['GET'])
def sample_ids():
    sample_learners = df_learner['LEARNER_ID'].head(10).tolist() if not df_learner.empty else []
    open_j = df_job_post[df_job_post['job_post_status']=='open'] if not df_job_post.empty else pd.DataFrame()
    sample_jobs = open_j['job_post_id'].head(5).tolist() if not open_j.empty else []
    return jsonify({'sample_learner_ids': sample_learners, 'sample_job_ids': sample_jobs})


@app.route('/api/academy_skills', methods=['GET'])
def academy_skills():
    return jsonify({'skills': academy_skills_list, 'certs': cert_skills_list})
@app.route('/api/job_trends', methods=['GET'])
def get_job_trends():
    try:
        # 1. Top Positions
        top_positions = []
        if not df_job_post.empty:
            pos_counts = df_job_post['position_name_en'].value_counts().head(10)
            for title, count in pos_counts.items():
                top_positions.append({'title': str(title), 'count': int(count)})
                
        # 2. Top Skills
        top_skills = []
        if not df_job_skill.empty and not df_acad_skill.empty:
            skill_counts = df_job_skill[df_job_skill['Is_certificate'].astype(str).str.lower()!='true']['skill_id'].value_counts().head(10)
            for skill_id, count in skill_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                top_skills.append({'skill': name, 'count': int(count)})
                
        # 3. Monthly Trends
        monthly_trends = []
        if not df_job_post.empty and 'open_date' in df_job_post.columns:
            df_dates = pd.to_datetime(df_job_post['open_date'], errors='coerce').dropna()
            monthly_counts = df_dates.dt.to_period('M').value_counts().sort_index().tail(6)
            for period, count in monthly_counts.items():
                monthly_trends.append({
                    'month': period.strftime('%b %Y'),
                    'count': int(count)
                })

        return jsonify({
            'top_positions': top_positions,
            'top_skills': top_skills,
            'monthly_trends': monthly_trends
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/course_performance', methods=['GET'])
def get_course_performance():
    try:
        # Mocking dynamic data based on available datasets
        # In a real scenario, this would involve complex joins between learners, courses, and job outcomes
        
        # 1. Monthly Enrollment Trend (Last 6 months)
        months = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.']
        enrollment_data = [3800, 4200, 4100, 4800, 5200, 5600] # Simulated growth
        
        # 2. Top 5 Impact Courses
        top_impact = []
        if not df_acad_course.empty:
            sample_courses = df_acad_course.head(5)
            impact_vals = [48, 42, 39, 35, 28] # Percent increase in match score
            for i, (_, row) in enumerate(sample_courses.iterrows()):
                top_impact.append({
                    'name': str(row.get('Course_name', 'Course')),
                    'impact': impact_vals[i] if i < len(impact_vals) else 20
                })
        
        # 3. Skill Alignment (Market Demand vs Course Output)
        # Aligning with actual job skills from df_job_skill
        skills = []
        market_demand = []
        course_supply = []
        
        if not df_job_skill.empty:
            skill_counts = df_job_skill[df_job_skill['Is_certificate'].astype(str).str.lower()!='true']['skill_id'].value_counts().head(6)
            for skill_id, count in skill_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                skills.append(name)
                # Normalize count to 0-100 for market demand scale
                # Assuming max count represents ~95% demand for visualization
                max_c = skill_counts.iloc[0] if not skill_counts.empty else 1
                market_val = min(95, int((count / max_c) * 95))
                market_demand.append(market_val)
                # Simulated course supply (randomized around market demand for demo)
                import random
                course_supply.append(max(40, market_val - random.randint(5, 20)))
        
        if not skills: # Fallback if data missing
            skills = ['Programming', 'Cloud', 'Data Science', 'English', 'Design', 'Management']
            market_demand = [85, 78, 92, 70, 65, 80]
            course_supply = [72, 60, 88, 75, 58, 70]
        
        # 4. Category Distribution
        categories = ['Technology', 'Business', 'Language', 'Creative', 'Others']
        cat_counts = [45, 25, 15, 10, 5]

        return jsonify({
            'kpis': {
                'total_enrollments': 45280,
                'completion_rate': 68.5,
                'employment_impact': 82.3
            },
            'monthly_trend': {
                'labels': months,
                'data': enrollment_data
            },
            'top_impact_courses': top_impact,
            'skill_alignment': {
                'labels': skills,
                'market': market_demand,
                'course': course_supply
            },
            'categories': {
                'labels': categories,
                'data': cat_counts
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)