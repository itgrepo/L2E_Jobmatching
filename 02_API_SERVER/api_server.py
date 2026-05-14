from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import os
import re
import traceback
import pickle
import hashlib
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("L2E Server Initializing...")
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / 'models'
TRAINED_MODEL_PATH = REPO_ROOT / '03_MODEL_TRAINING' / 'models'
CACHE_ROOT = BASE_DIR / '.cache_embeddings'
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_METADATA = {}
MODEL_SOURCE = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Weight presets for job matching
WEIGHT_PRESETS = {
    'balanced':     {'title': 0.20, 'skill': 0.20, 'budget': 0.20, 'cert': 0.20, 'semantic': 0.20},
    'salary_focus': {'title': 0.15, 'skill': 0.15, 'budget': 0.50, 'cert': 0.10, 'semantic': 0.10},
    'skill_focus':  {'title': 0.15, 'skill': 0.50, 'budget': 0.10, 'cert': 0.15, 'semantic': 0.10},
    'cert_focus':   {'title': 0.15, 'skill': 0.15, 'budget': 0.10, 'cert': 0.50, 'semantic': 0.10},
    'title_focus':  {'title': 0.50, 'skill': 0.20, 'budget': 0.10, 'cert': 0.10, 'semantic': 0.10},
}
MODEL_SIGNATURE = hashlib.md5(MODEL_SOURCE.encode('utf-8')).hexdigest()[:12]


def resolve_model_dir():
    env_path = os.environ.get('L2E_MODEL_DIR', '').strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        MODEL_PATH / 'fine_tuned_bert',
        TRAINED_MODEL_PATH / 'fine_tuned_bert',
    ])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


# In-memory profile override store (for onboarding saves)
profile_overrides = {}

# ==========================================
# 1. LOAD BERT MODEL
# ==========================================
bert_model = None
try:
    custom = resolve_model_dir()
    if custom is not None:
        bert_model = SentenceTransformer(str(custom))
        MODEL_SOURCE = str(custom)
        meta_candidates = [
            custom.parent / 'model_info_v6.json',
            custom.parent / 'model_info.json',
        ]
        for meta_path in meta_candidates:
            if meta_path.exists():
                try:
                    MODEL_METADATA = json.loads(meta_path.read_text(encoding='utf-8'))
                    break
                except Exception:
                    MODEL_METADATA = {}
        MODEL_SIGNATURE = hashlib.md5(MODEL_SOURCE.encode('utf-8')).hexdigest()[:12]
        print(f"   Loaded fine-tuned BERT model from {custom}")
    else:
        bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("   Loaded base multilingual BERT model")
except Exception as e:
    print(f"   BERT load error: {e}")

CACHE_DIR = CACHE_ROOT / MODEL_SIGNATURE
CACHE_DIR.mkdir(parents=True, exist_ok=True)
if MODEL_METADATA:
    print(f"   Model metadata: version={MODEL_METADATA.get('version', 'unknown')}")
print(f"   Cache namespace: {CACHE_DIR}")

# ==========================================
# 2. LOAD ALL TABLES FROM CSV
# ==========================================
print("   Loading tables from CSV...")

from db_config import (
    fetch_learner, fetch_skills, fetch_certs, fetch_education,
    fetch_experience, fetch_preference, fetch_job_post, fetch_job_skill,
    fetch_company, fetch_university, fetch_branch, fetch_edu_level,
    fetch_position, fetch_worktype, fetch_jobtype, fetch_postcode,
    fetch_acad_skill, fetch_acad_course
)

df_learner    = fetch_learner()
df_skills     = fetch_skills()
df_certs      = fetch_certs()
df_education  = fetch_education()
df_experience = fetch_experience()
df_preference = fetch_preference()
df_job_post   = fetch_job_post()
df_job_skill  = fetch_job_skill()
df_company    = fetch_company()
df_university = fetch_university()
df_branch     = fetch_branch()
df_edu_level  = fetch_edu_level()
df_position   = fetch_position()
df_worktype   = fetch_worktype()
df_jobtype    = fetch_jobtype()
df_postcode   = fetch_postcode()
df_acad_skill = fetch_acad_skill()
df_acad_course= fetch_acad_course()

print(f"   Learners: {len(df_learner)}, Jobs: {len(df_job_post)}, Certs: {len(df_certs)}")

# ==========================================
# 3. BUILD LOOKUP TABLES
# ==========================================
uni_map      = dict(zip(df_university['UNIVERSITY_ID'].astype(str), df_university['UNIVERSITY_NAME'])) if not df_university.empty else {}
branch_map   = dict(zip(df_branch['BRANCHID'].astype(str), df_branch['BRANCH_NAME'])) if not df_branch.empty else {}
edu_lvl_map  = dict(zip(df_edu_level['EDUCATOINBG_ID'].astype(str), df_edu_level['EDUCATOINBG_NAME'])) if not df_edu_level.empty else {}
pos_map      = dict(zip(df_position['POSITION_ID'].astype(str), df_position['POSITION_NAME'])) if not df_position.empty else {}
wt_map       = dict(zip(df_worktype['WORKTYPE_ID'].astype(str), df_worktype['WORKTYPE_NAME'])) if not df_worktype.empty else {}
jt_map       = dict(zip(df_jobtype['JOBTYPE_ID'].astype(str), df_jobtype['JOBTYPE_NAME'])) if not df_jobtype.empty else {}
skill_id_map = dict(zip(df_acad_skill['SKILL_ID'], df_acad_skill['SKILL_NAME'])) if not df_acad_skill.empty else {}

prov_map = {}    # SUBDISTRICT_ID -> PROVINCE_NAME
prov_id_map = {}  # PROVINCE_ID -> PROVINCE_NAME
if not df_postcode.empty:
    prov_map    = dict(zip(df_postcode['SUBDISTRICT_ID'].astype(str), df_postcode['PROVINCE_NAME']))
    prov_id_map = dict(zip(df_postcode['PROVINCE_ID'].astype(str), df_postcode['PROVINCE_NAME']))

company_map = {}
if not df_company.empty:
    for _, row in df_company.iterrows():
        cid = str(row.get('COMPANY_ID', ''))
        company_map[cid] = {
            'name_th': str(row.get('COMPANY_NAME_TH', '')),
            'name_en': str(row.get('COMPANY_NAME_EN', '')),
            'logo': str(row.get('LOGO_IMAGE', '')),
            'website': str(row.get('COMPANY_WEBSITE', '')),
            'specialty': str(row.get('SPECIALTY', '')),
            'size': str(row.get('ORGANIZATION_SIZE', '')),
            'about': str(row.get('ABOUT_ME', '')),
        }

# Academy skills list (for onboarding dropdowns)
academy_skills_list = []
cert_skills_list = []
if not df_acad_skill.empty:
    unique_skills = df_acad_skill[['SKILL_ID','SKILL_NAME']].drop_duplicates()
    for _, r in unique_skills.iterrows():
        is_cert = bool(df_acad_skill[df_acad_skill['SKILL_ID'] == r['SKILL_ID']]['IS_CERTIFICATE'].any())
        if is_cert:
            cert_skills_list.append({'id': r['SKILL_ID'], 'name': r['SKILL_NAME']})
        else:
            academy_skills_list.append({'id': r['SKILL_ID'], 'name': r['SKILL_NAME']})

# ==========================================
# 4. AGGREGATE LEARNER DATA
# ==========================================
print("   Aggregating learner data...")

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
        wt_raw = row.get('WORKTYPE')
        wt_id = str(wt_raw[0]) if isinstance(wt_raw, list) and wt_raw else str(int(wt_raw)) if pd.notnull(wt_raw) else '1'
        jt_raw = row.get('JOBTYPE')
        jt_id = str(jt_raw[0]) if isinstance(jt_raw, list) and jt_raw else str(int(jt_raw)) if pd.notnull(jt_raw) else '1'
        learner_pref[lid] = {
            'work_type': wt_map.get(wt_id, 'Onsite'),
            'job_type': jt_map.get(jt_id, 'Full-time'),
            'expected_salary': float(row.get('SALARY', 0)) if pd.notnull(row.get('SALARY')) else 0,
            'preferred_province': str(row.get('PROVINCE', ''))
        }

# ==========================================
# 5. AGGREGATE JOB DATA
# ==========================================
print("   Aggregating job data...")

job_skills_map = {}
if not df_job_skill.empty:
    for _, row in df_job_skill.iterrows():
        jid = str(row['JOB_POST_ID'])
        skill_id = str(row.get('SKILL_ID', ''))
        skill_name = skill_id_map.get(skill_id, skill_id)
        is_cert = str(row.get('IS_CERTIFICATE', 'False')).lower() in ['true','1']
        level = str(row.get('SKILL_LEVEL_ID', ''))
        if jid not in job_skills_map: job_skills_map[jid] = {'skills': [], 'certs': []}
        if is_cert:
            job_skills_map[jid]['certs'].append({'name': skill_name, 'level': level})
        else:
            job_skills_map[jid]['skills'].append({'name': skill_name, 'level': level})

# ==========================================
# 6. COMPUTE BERT EMBEDDINGS
# ==========================================
print("   Computing BERT embeddings...")

import datetime
MAX_JOBS = int(os.environ.get('L2E_MAX_JOBS', '3000') or 3000)
MAX_USERS = int(os.environ.get('L2E_MAX_USERS', '10000') or 10000)
print(f"   Precompute limits: jobs={MAX_JOBS}, users={MAX_USERS}")


def clean_text(value):
    if value is None or pd.isna(value):
        return ''
    text = str(value).strip()
    if text.lower() == 'nan':
        return ''
    return ' '.join(text.split())


def join_unique(values, sep=', '):
    seen = set()
    output = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return sep.join(output)


def build_user_text(title, about, skills_list, certs_list, exp_list, pref=None, language=''):
    sections = []
    role_text = clean_text(title)
    about_text = clean_text(about)
    skills_text = join_unique(skills_list)
    certs_text = join_unique(certs_list)

    experience_parts = []
    for exp in (exp_list or [])[:3]:
        position = clean_text(exp.get('position', ''))
        company = clean_text(exp.get('company', ''))
        description = clean_text(exp.get('description', ''))
        tech_tools = join_unique(exp.get('tech_tools', []))
        chunk = ' | '.join([part for part in [position, company, description] if part])
        if tech_tools:
            chunk = f"{chunk} | Tools: {tech_tools}" if chunk else f"Tools: {tech_tools}"
        if chunk:
            experience_parts.append(chunk)

    pref_parts = []
    pref = pref or {}
    desired_role = clean_text(pref.get('desired_role', ''))
    work_type = clean_text(pref.get('work_type', ''))
    job_type = clean_text(pref.get('job_type', ''))
    preferred_province = clean_text(pref.get('preferred_province', ''))
    expected_salary = pref.get('expected_salary', 0)
    if desired_role:
        pref_parts.append(f"Desired role: {desired_role}")
    if work_type:
        pref_parts.append(f"Work mode: {work_type}")
    if job_type:
        pref_parts.append(f"Job type: {job_type}")
    if preferred_province:
        pref_parts.append(f"Preferred province: {preferred_province}")
    if expected_salary:
        pref_parts.append(f"Expected salary: {int(float(expected_salary))}")

    if role_text:
        sections.append(f"Current role: {role_text}")
    if skills_text:
        sections.append(f"Skills: {skills_text}")
    if certs_text:
        sections.append(f"Certificates: {certs_text}")
    if experience_parts:
        sections.append(f"Experience: {' || '.join(experience_parts)}")
    if pref_parts:
        sections.append(f"Preferences: {'; '.join(pref_parts)}")
    language_text = clean_text(language)
    if language_text:
        sections.append(f"Language: {language_text}")
    if about_text:
        sections.append(f"About: {about_text}")
    return '\n'.join(sections).strip()


def build_job_text(row, job_data, company_data=None):
    company_data = company_data or {}
    title = clean_text(row.get('POSITION_NAME_EN', row.get('POSITION_NAME_TH', row.get('JOB_TITLE', ''))))
    summary = clean_text(row.get('JOB_SUMMARY', ''))
    detail = clean_text(row.get('JOB_DETAIL', ''))
    job_type = clean_text(row.get('JOB_TYPE', ''))
    job_sub_type = clean_text(row.get('JOB_SUB_TYPE', ''))
    province = clean_text(prov_id_map.get(str(row.get('PROVINCE_ID', '')), row.get('PROVINCE_ID', '')))
    required_skills = join_unique([s.get('name', '') for s in job_data.get('skills', [])])
    required_certs = join_unique([c.get('name', '') for c in job_data.get('certs', [])])
    company_specialty = clean_text(company_data.get('specialty', ''))
    income_min = row.get('INCOME_MIN', 0)
    income_max = row.get('INCOME_MAX', 0)

    sections = []
    if title:
        sections.append(f"Job title: {title}")
    if required_skills:
        sections.append(f"Required skills: {required_skills}")
    if required_certs:
        sections.append(f"Required certificates: {required_certs}")
    if summary:
        sections.append(f"Summary: {summary}")
    if detail:
        sections.append(f"Details: {detail}")
    if job_type:
        sections.append(f"Job family: {job_type}")
    if job_sub_type:
        sections.append(f"Job sub-type: {job_sub_type}")
    if province:
        sections.append(f"Province: {province}")
    if company_specialty:
        sections.append(f"Company specialty: {company_specialty}")
    if income_min or income_max:
        sections.append(f"Salary range: {int(float(income_min or 0))} - {int(float(income_max or 0))}")
    return '\n'.join(sections).strip()


def limit_rows(df, max_rows):
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.head(max_rows)


_today = datetime.date.today()
if not df_job_post.empty:
    _open = df_job_post.copy()
    _open['_end'] = pd.to_datetime(_open['END_DATE'], errors='coerce').dt.date
    # Keep only jobs that are not yet expired, or have no end date
    open_jobs = limit_rows(_open[_open['_end'].isna() | (_open['_end'] >= _today)], MAX_JOBS)
    if len(open_jobs) == 0:
        open_jobs = limit_rows(_open, MAX_JOBS)
    open_jobs = open_jobs.drop(columns=['_end'])
else:
    open_jobs = pd.DataFrame()

active_learners = limit_rows(df_learner[df_learner['STATUS'] == 'active'], MAX_USERS) if not df_learner.empty else pd.DataFrame()
if len(active_learners) == 0:
    active_learners = limit_rows(df_learner, MAX_USERS)

job_matrix = None
user_matrix = None
job_records = []
user_records = []

if bert_model and not open_jobs.empty:
    job_texts = []
    for _, row in open_jobs.iterrows():
        jid = str(row['JOB_POST_ID'])
        j = job_skills_map.get(jid, {})
        cid = str(row.get('COMPANY_ID', ''))
        job_texts.append(build_job_text(row, j, company_map.get(cid, {})))
        job_records.append(row.to_dict())

    job_texts_hash = hashlib.md5((MODEL_SIGNATURE + '||' + ''.join(job_texts)).encode()).hexdigest()
    job_cache_path = CACHE_DIR / f'job_embeddings_{job_texts_hash}.pkl'
    if job_cache_path.exists():
        with open(job_cache_path, 'rb') as f:
            job_matrix = pickle.load(f)
        print(f"   Job embeddings (cached): {len(job_records)}")
    else:
        job_matrix = bert_model.encode(job_texts, convert_to_tensor=True, show_progress_bar=False)
        with open(job_cache_path, 'wb') as f:
            pickle.dump(job_matrix, f)
        print(f"   Job embeddings: {len(job_records)}")

if bert_model and not active_learners.empty:
    user_texts = []
    for _, row in active_learners.iterrows():
        lid = str(row['LEARNER_ID'])
        title = str(row.get('JOB_TITLE_EN', ''))
        about = str(row.get('ABOUT_ME', ''))
        skills_list_tmp = [s['name'] for s in learner_skills.get(lid, [])]
        certs_list_tmp = [c['course_name'] for c in learner_certs.get(lid, [])]
        exp_list_tmp = learner_exp.get(lid, [])
        pref_tmp = learner_pref.get(lid, {})
        language = str(row.get('LANGUAGE', ''))
        user_texts.append(build_user_text(title, about, skills_list_tmp, certs_list_tmp, exp_list_tmp, pref_tmp, language))
        user_records.append(row.to_dict())

    user_texts_hash = hashlib.md5((MODEL_SIGNATURE + '||' + ''.join(user_texts)).encode()).hexdigest()
    user_cache_path = CACHE_DIR / f'user_embeddings_{user_texts_hash}.pkl'
    if user_cache_path.exists():
        with open(user_cache_path, 'rb') as f:
            user_matrix = pickle.load(f)
        print(f"   User embeddings (cached): {len(user_records)}")
    else:
        user_matrix = bert_model.encode(user_texts, convert_to_tensor=True, show_progress_bar=False)
        with open(user_cache_path, 'wb') as f:
            pickle.dump(user_matrix, f)
        print(f"   User embeddings: {len(user_records)}")

print("Server Ready! Port 5001")

# ==========================================
# 7. HELPER FUNCTIONS
# ==========================================

def scale_scores_minmax(scores_tensor, lo=40.0, hi=88.0):
    """Linear clamp — cosine similarity [0,1] -> [0,100] then clip to [lo, hi]."""
    scores = scores_tensor.cpu().numpy() if hasattr(scores_tensor, 'cpu') else np.array(scores_tensor)
    return np.clip(scores * 100.0, lo, hi)


def get_cert_match(user_cert_names, required_cert_names):
    if not required_cert_names or not user_cert_names:
        return []
    matched_reqs = []
    user_lower = [c.lower() for c in user_cert_names]

    # Pre-encode once for all BERT comparisons
    needs_bert = []
    string_matched = set()
    for req in required_cert_names:
        req_lower = req.lower()
        if any(req_lower in uc or uc in req_lower for uc in user_lower):
            matched_reqs.append(req)
            string_matched.add(req)
        else:
            needs_bert.append(req)

    if needs_bert and bert_model:
        try:
            uc_embs = bert_model.encode(user_cert_names, convert_to_tensor=True)
            req_embs = bert_model.encode(needs_bert, convert_to_tensor=True)
            sims = util.cos_sim(req_embs, uc_embs)
            for i, req in enumerate(needs_bert):
                if float(torch.max(sims[i])) >= 0.72:
                    matched_reqs.append(req)
        except:
            pass
    return matched_reqs


def get_skill_match_and_gap(user_skills, req_skills, precomputed_req_embs=None, precomputed_user_embs=None):
    if not req_skills: return [], [], 100, {}
    if not user_skills: return [], req_skills, 0, {req: "คุณยังไม่มีทักษะนี้หรือทักษะที่ใกล้เคียงเลย" for req in req_skills}
    user_lower = [s.lower() for s in user_skills]
    matched = []
    missing = []
    explanations = {}
    total_score = 0.0

    user_embs = precomputed_user_embs
    req_embs = precomputed_req_embs
    if bert_model:
        try:
            if user_embs is None:
                user_embs = bert_model.encode(user_skills, convert_to_tensor=True)
            if req_embs is None:
                req_embs = bert_model.encode(req_skills, convert_to_tensor=True)
        except: pass

    for ri, rs in enumerate(req_skills):
        rs_lower = rs.lower()
        best_sim = 0.0
        best_match_user_skill = None

        # 1. Substring match
        for i, u in enumerate(user_lower):
            if rs_lower in u or u in rs_lower:
                best_sim = 1.0
                best_match_user_skill = user_skills[i]
                break

        # 2. BERT fuzzy match with domain-aware boosts and penalties
        if best_sim < 1.0 and user_embs is not None and req_embs is not None:
            try:
                sims = util.cos_sim(req_embs[ri], user_embs)[0]

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
        'status': str(r.get('STATUS','active')),
        'language': str(r.get('LANGUAGE','1')),
        'profile_completeness': calc_completeness(lid, skills_list, pref)
    }


def calc_completeness(lid, skills_list, pref):
    checks = {
        'job_title': True,
        'about_me': True,
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


def match_jobs_for_user(lid, user_emb=None, extra_certs=None, preset='balanced'):
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
            language = str(r.get('LANGUAGE', ''))
            user_emb = bert_model.encode(
                build_user_text(title, about, skills_list, user_cert_names, exp_list, pref, language),
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
        jid = str(job.get('JOB_POST_ID',''))
        cid = str(job.get('COMPANY_ID',''))
        comp = company_map.get(cid, {})
        j_data = job_skills_map.get(jid, {})
        req_skills = [s['name'] for s in j_data.get('skills', [])]
        req_certs = [c['name'] for c in j_data.get('certs', [])]

        semantic = float(scaled[i])
        sal_min = float(job.get('INCOME_MIN', 0) or 0)
        sal_max = float(job.get('INCOME_MAX', 0) or 0)

        # Deviation calculation for budget fit
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

        req_certs = sorted(list(set([c['name'] for c in j_data.get('certs', []) if c.get('name')])))
        cert_matched = get_cert_match(user_cert_names, req_certs)
        cert_missing = [req for req in req_certs if req not in cert_matched]

        if not req_certs:
            cert_score = 100
        else:
            cert_score = int((len(cert_matched) / len(req_certs)) * 100)

        # Title-level role match
        job_title_text = str(job.get('POSITION_NAME_EN', job.get('POSITION_NAME_TH', ''))).lower()
        u_role_low = user_role_text.lower()

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
                # Related bucket: linear map 0.85-0.95 -> 60-90%
                title_match_score = 60.0 + (raw_title_sim - 0.85) / 0.10 * 30.0
            else:
                # Cross-domain: steep drop below 0.85
                title_match_score = max(5.0, 60.0 * ((max(0, raw_title_sim - 0.75)) / 0.10) ** 2)

        # Tech synonym boost: treat programmer/developer/software engineer variants as equivalent
        tech_synonyms = ['programmer', 'developer', 'software engineer', 'software developer', 'web developer', 'full stack']
        if any(s in job_title_text for s in tech_synonyms) and any(s in u_role_low for s in tech_synonyms):
            title_match_score = max(title_match_score, 85.0 if raw_title_sim > 0.80 else 70.0)
            raw_title_sim = max(raw_title_sim, 0.90)

        title_match_score = round(min(title_match_score, 100.0), 1)

        # Composite formula with preset weights
        w = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS['balanced'])
        raw_final = (
            title_match_score * w['title']   +
            skill_score       * w['skill']   +
            budget_score      * w['budget']  +
            cert_score        * w['cert']    +
            semantic          * w['semantic']
        )

        # 3-tier hard cap based on title similarity
        if raw_title_sim < 0.80:
            raw_final = min(raw_final, 65.0)
        elif raw_title_sim < 0.88:
            raw_final = min(raw_final, 85.0)
        elif raw_title_sim < 0.92:
            raw_final = min(raw_final, 95.0)

        # Work style mismatch penalty
        work_type_penalty = 0.0
        job_work_type = wt_map.get(str(job.get('WORK_LOCATION_TYPE_ID','1')), 'Onsite')
        user_work_type = pref.get('work_type', 'Onsite')
        if user_work_type.lower() != 'ไม่ระบุ' and job_work_type.lower() != user_work_type.lower():
            if 'onsite' in user_work_type.lower() and 'remote' in job_work_type.lower():
                work_type_penalty = 20.0
            elif 'remote' in user_work_type.lower() and 'onsite' in job_work_type.lower():
                work_type_penalty = 20.0
            else:
                work_type_penalty = 10.0

        # Location mismatch penalty
        location_fit = 100.0
        prov_id = str(job.get('PROVINCE_ID',''))
        job_location = prov_id_map.get(prov_id, prov_id)

        if user_prov and job_location:
            if 'remote' not in job_work_type.lower():
                u_prov_clean = user_prov.strip().lower()
                j_prov_clean = job_location.strip().lower()

                if u_prov_clean != 'ไม่ระบุ' and j_prov_clean != 'ไม่ระบุ' and u_prov_clean != j_prov_clean:
                    location_fit = 0.0
                    work_type_penalty += 25.0

        final = round(min(max(raw_final - work_type_penalty, 10.0), 99.0), 1)

        show_salary = str(job.get('IS_DISPLAY_INCOME','True')).lower() == 'true'
        if show_salary and sal_min>0 and sal_max>0:
            salary_text = f"{int(sal_min):,} - {int(sal_max):,}"
        elif sal_max>0:
            salary_text = f"สูงสุด {int(sal_max):,}"
        else:
            salary_text = "Negotiable"

        prov_id = str(job.get('PROVINCE_ID',''))
        location = prov_id_map.get(prov_id, prov_id)
        wt_id = str(job.get('WORK_LOCATION_TYPE_ID','1'))

        # Use JOB_TITLE (actual posting title) but strip leading 'รับสมัคร ' prefix
        raw_jt = str(job.get('JOB_TITLE', ''))
        display_title = re.sub(r'^รับสมัคร\s*', '', raw_jt).strip() if raw_jt else str(job.get('POSITION_NAME_EN', job.get('POSITION_NAME_TH','')))

        results.append({
            'job_id': jid,
            'title': display_title,
            'company': comp.get('name_th') or comp.get('name_en') or cid,
            'company_id': cid,
            'company_about': comp.get('about',''),
            'logo_url': get_logo(cid),
            'location': location,
            'work_type': wt_map.get(wt_id, 'Onsite'),
            'job_type': str(job.get('JOB_TYPE','IT')),
            'salary_text': salary_text,
            'salary_min': sal_min,
            'salary_max': sal_max,
            'description': str(job.get('JOB_DETAIL','')),
            'summary': str(job.get('JOB_SUMMARY','')),
            'open_date': str(job.get('OPEN_DATE','')),
            'end_date': str(job.get('END_DATE','')),
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

        if uid in profile_overrides:
            ov = profile_overrides[uid]
            complete = len(ov.get('skills',[])) >= 2 and ov.get('expected_salary',0) > 0
            return jsonify({'role': 'candidate', 'profile_complete': complete, 'data': ov})

        if not df_learner.empty:
            row = df_learner[df_learner['LEARNER_ID'] == uid]
            if not row.empty:
                profile = build_candidate_profile(uid, row.iloc[0])
                complete = is_profile_complete(uid, profile['skills'], learner_pref.get(uid,{}))
                return jsonify({'role': 'candidate', 'profile_complete': complete, 'data': profile})

        if not df_job_post.empty:
            row = df_job_post[df_job_post['JOB_POST_ID'] == uid]
            if not row.empty:
                r = row.iloc[0]
                jid = str(r['JOB_POST_ID'])
                cid = str(r.get('COMPANY_ID',''))
                comp = company_map.get(cid, {})
                j_data = job_skills_map.get(jid, {})
                prov_id = str(r.get('PROVINCE_ID',''))
                province = prov_id_map.get(prov_id, prov_id)
                wt_id = str(r.get('WORK_LOCATION_TYPE_ID','1'))
                return jsonify({'role': 'employer', 'profile_complete': True, 'data': {
                    'job_post_id': jid,
                    'job_id': jid,
                    'title': str(r.get('POSITION_NAME_EN', r.get('POSITION_NAME_TH',''))),
                    'company_id': cid,
                    'company_name': comp.get('name_th', comp.get('name_en', cid)),
                    'company_name_en': comp.get('name_en',''),
                    'company_about': comp.get('about',''),
                    'logo_url': get_logo(cid),
                    'location': province,
                    'work_style': wt_map.get(wt_id,'Onsite'),
                    'job_type': str(r.get('JOB_TYPE','IT')),
                    'job_sub_type': str(r.get('JOB_SUB_TYPE','')),
                    'salary_min': float(r.get('INCOME_MIN',0) or 0),
                    'salary_max': float(r.get('INCOME_MAX',0) or 0),
                    'description': str(r.get('JOB_DETAIL','')),
                    'summary': str(r.get('JOB_SUMMARY','')),
                    'required_skills': [s['name'] for s in j_data.get('skills',[])],
                    'required_certs': [c['name'] for c in j_data.get('certs',[])],
                    'open_date': str(r.get('OPEN_DATE','')),
                    'end_date': str(r.get('END_DATE','')),
                    'status': str(r.get('JOB_POST_STATUS','')),
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

        if uid in profile_overrides:
            profile = profile_overrides[uid]
        else:
            row = df_learner[df_learner['LEARNER_ID'] == uid] if not df_learner.empty else pd.DataFrame()
            if row.empty:
                return jsonify({'jobs': [], 'error': 'User not found'})
            profile = build_candidate_profile(uid, row.iloc[0])

        if not bert_model or job_matrix is None:
            return jsonify({'jobs': [], 'error': 'BERT model not loaded'})

        skills = profile.get('skills', [])
        certs = profile.get('certificates', [])
        title = profile.get('current_role', '')
        about = profile.get('about_me', '')
        exp_list = profile.get('experience', [])
        pref = {
            'desired_role': title,
            'expected_salary': float(profile.get('expected_salary', 0) or 0),
            'work_type': profile.get('work_style_pref', 'Onsite'),
            'job_type': profile.get('job_type_pref', 'Full-time'),
            'preferred_province': profile.get('location', ''),
        }
        text = build_user_text(title, about, skills, certs, exp_list, pref, profile.get('language', ''))
        user_emb = bert_model.encode(text, convert_to_tensor=True)

        # Inject pref so title_match can use it inside match_jobs_for_user
        if uid not in learner_pref:
            learner_pref[uid] = {}
        learner_pref[uid]['desired_role'] = title
        learner_pref[uid]['expected_salary'] = float(profile.get('expected_salary', 0) or 0)
        learner_pref[uid]['work_type'] = profile.get('work_style_pref', 'Onsite')
        learner_pref[uid]['job_type'] = profile.get('job_type_pref', 'Full-time')
        learner_pref[uid]['preferred_province'] = profile.get('location', '')
        preset = data.get('preset', 'balanced')
        jobs = match_jobs_for_user(uid, user_emb=user_emb, extra_certs=certs, preset=preset)
        return jsonify({'jobs': jobs[:limit], 'preset_used': preset})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/match_jobs', methods=['POST'])
def match_jobs():
    """Match jobs from full form submission — delegates to match_jobs_for_user."""
    try:
        data = request.json
        if not bert_model or job_matrix is None:
            return jsonify({'jobs': []})

        role = data.get('interested_role', data.get('current_role', ''))
        skills = data.get('skills', [])
        about_me = data.get('about_me', '')
        user_certs = data.get('certificates', [])
        expected_salary = float(data.get('expected_salary', 0) or 0)
        exp_list = [{'description': e} for e in data.get('work_experiences', []) if e]
        uid = str(data.get('user_id', '_form_user_'))

        learner_pref[uid] = {
            'desired_role': role,
            'expected_salary': expected_salary,
            'work_type': data.get('work_type', 'Onsite'),
            'job_type': data.get('job_type', 'Full-time'),
            'preferred_province': data.get('location', data.get('preferred_province', '')),
        }
        text = build_user_text(role, about_me, skills, user_certs, exp_list, learner_pref[uid], data.get('language', ''))
        learner_skills[uid] = [{'name': s, 'level': ''} for s in skills]
        learner_certs[uid] = [{'course_name': c} for c in user_certs]
        learner_exp[uid] = exp_list
        user_emb = bert_model.encode(text, convert_to_tensor=True)

        jobs = match_jobs_for_user(uid, user_emb=user_emb, extra_certs=user_certs)
        limit = int(data.get('limit', 50))
        return jsonify({'jobs': jobs[:limit]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/job_detail/<job_id>', methods=['GET'])
def job_detail(job_id):
    try:
        row = df_job_post[df_job_post['JOB_POST_ID'] == job_id] if not df_job_post.empty else pd.DataFrame()
        if row.empty:
            return jsonify({'error': 'Job not found'}), 404
        r = row.iloc[0]
        jid = str(r['JOB_POST_ID'])
        cid = str(r.get('COMPANY_ID',''))
        comp = company_map.get(cid, {})
        j_data = job_skills_map.get(jid, {})
        prov_id = str(r.get('PROVINCE_ID',''))
        wt_id = str(r.get('WORK_LOCATION_TYPE_ID','1'))
        sal_min = float(r.get('INCOME_MIN',0) or 0)
        sal_max = float(r.get('INCOME_MAX',0) or 0)
        return jsonify({
            'job_id': jid, 'title': str(r.get('POSITION_NAME_EN',r.get('POSITION_NAME_TH',''))),
            'company': comp.get('name_th') or cid, 'company_name_en': comp.get('name_en',''),
            'company_about': comp.get('about',''), 'company_website': comp.get('website',''),
            'company_size': comp.get('size',''), 'company_specialty': comp.get('specialty',''),
            'logo_url': get_logo(cid), 'location': prov_id_map.get(prov_id, prov_id),
            'work_type': wt_map.get(wt_id,'Onsite'), 'job_type': str(r.get('JOB_TYPE','')),
            'salary_min': sal_min, 'salary_max': sal_max,
            'salary_text': f"฿{int(sal_min):,} - {int(sal_max):,}" if sal_min>0 else "Negotiable",
            'description': str(r.get('JOB_DETAIL','')), 'summary': str(r.get('JOB_SUMMARY','')),
            'working_time': str(r.get('WORKING_TIME','')),
            'open_date': str(r.get('OPEN_DATE','')), 'end_date': str(r.get('END_DATE','')),
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


@app.route('/api/company_jobs/<company_id>', methods=['GET'])
def company_jobs(company_id):
    """Return all open jobs belonging to a company."""
    try:
        if df_job_post.empty:
            return jsonify([])
        rows = df_job_post[df_job_post['COMPANY_ID'] == company_id]
        jobs = []
        for _, r in rows.iterrows():
            jid = str(r['JOB_POST_ID'])
            j_data = job_skills_map.get(jid, {})
            prov_id = str(r.get('PROVINCE_ID', ''))
            wt_id = str(r.get('Work_location_type_id', r.get('WORK_LOCATION_TYPE_ID', '1')))
            raw_jt2 = str(r.get('JOB_TITLE', ''))
            disp_title2 = re.sub(r'^รับสมัคร\s*', '', raw_jt2).strip() if raw_jt2 else str(r.get('POSITION_NAME_EN', r.get('POSITION_NAME_TH', '')))
            jobs.append({
                'job_post_id': jid,
                'title': disp_title2,
                'job_type': str(r.get('JOB_TYPE', '')),
                'job_sub_type': str(r.get('JOB_SUB_TYPE', '')),
                'salary_min': float(r.get('INCOME_MIN', 0) or 0),
                'salary_max': float(r.get('INCOME_MAX', 0) or 0),
                'location': prov_id_map.get(prov_id, prov_id),
                'work_style': wt_map.get(wt_id, 'Onsite'),
                'description': str(r.get('JOB_DETAIL', '')),
                'summary': str(r.get('JOB_SUMMARY', '')),
                'status': str(r.get('JOB_POST_STATUS', '')),
                'required_skills': [s['name'] for s in j_data.get('skills', [])],
                'required_certs': [c['name'] for c in j_data.get('certs', [])],
            })
        return jsonify(sorted(jobs, key=lambda x: x['title']))
    except Exception as e:
        traceback.print_exc()
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
        skill_level = data.get('skill_level', '')
        budget_min = float(data.get('budget_min', 40000) or 40000)
        budget_max = float(data.get('budget_max', 80000) or 80000)
        required_certs = data.get('required_certs', [])
        job_location_pref = data.get('job_location', 'ไม่ระบุ')
        job_work_style_pref = data.get('job_work_style', 'Onsite')
        preset = data.get('preset', 'balanced')
        w = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS['balanced'])

        # Auto-lookup from job_id if provided
        job_id = data.get('job_id', '')
        if job_id and not df_job_post.empty:
            jrow = df_job_post[df_job_post['JOB_POST_ID'] == job_id]
            if not jrow.empty:
                jr = jrow.iloc[0]
                if not job_title:
                    job_title = str(jr.get('POSITION_NAME_EN', jr.get('POSITION_NAME_TH', '')))
                if not jd_text:
                    jd_text = str(jr.get('JOB_DETAIL', jr.get('JOB_SUMMARY', '')))
                if not required_skills:
                    jskills = job_skills_map.get(job_id, {})
                    required_skills = [s['name'] for s in jskills.get('skills', [])]
                if not required_certs:
                    jskills = job_skills_map.get(job_id, {})
                    required_certs = [c['name'] for c in jskills.get('certs', [])]
                if budget_min == 40000 and jr.get('INCOME_MIN'):
                    try: budget_min = float(jr['INCOME_MIN'])
                    except: pass
                if budget_max == 80000 and jr.get('INCOME_MAX'):
                    try: budget_max = float(jr['INCOME_MAX'])
                    except: pass
                if job_location_pref == 'ไม่ระบุ' and jr.get('PROVINCE_ID'):
                    job_location_pref = str(jr['PROVINCE_ID'])
                wlt = str(jr.get('WORK_LOCATION_TYPE_ID', '')).strip()
                if wlt == '1': job_work_style_pref = 'Onsite'
                elif wlt == '2': job_work_style_pref = 'Remote'
                elif wlt == '3': job_work_style_pref = 'Hybrid'

        if isinstance(required_certs, str):
            required_certs = [c.strip() for c in required_certs.split(',') if c.strip()]

        pseudo_job_row = {
            'POSITION_NAME_EN': job_title,
            'JOB_SUMMARY': jd_text[:300],
            'JOB_DETAIL': jd_text,
            'JOB_TYPE': data.get('job_type', ''),
            'JOB_SUB_TYPE': data.get('job_sub_type', ''),
            'PROVINCE_ID': '',
            'INCOME_MIN': budget_min,
            'INCOME_MAX': budget_max,
        }
        pseudo_job_data = {
            'skills': [{'name': s, 'level': skill_level} for s in required_skills],
            'certs': [{'name': c, 'level': ''} for c in required_certs],
        }
        query_text = build_job_text(pseudo_job_row, pseudo_job_data, {})

        job_title_emb = bert_model.encode(clean_text(job_title), convert_to_tensor=True) if job_title else None
        jd_emb = bert_model.encode(query_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(jd_emb, user_matrix)[0]

        # Title rescue: ensure same-title candidates always appear in top pool
        # (BERT semantic search may miss them if job description uses different words)
        title_rescue_indices = set()
        if job_title:
            j_title_lower = job_title.lower().strip()
            DATA_ROLE_GROUPS_RESCUE = [
                ['data engineer', 'etl engineer', 'data platform engineer', 'dataops', 'data pipeline'],
                ['data scientist', 'research scientist', 'machine learning scientist'],
                ['ml engineer', 'machine learning engineer', 'ai engineer', 'mlops engineer'],
                ['data analyst', 'bi analyst', 'business analyst', 'analytics engineer'],
                ['programmer', 'developer', 'software engineer', 'software developer', 'web developer', 'full stack'],
                ['devops', 'cloud engineer', 'system engineer', 'network engineer'],
            ]
            job_rescue_grp = next((g for g in DATA_ROLE_GROUPS_RESCUE if any(t in j_title_lower for t in g)), None)
            for ui, urec in enumerate(user_records):
                u_title = str(urec.get('JOB_TITLE_EN', urec.get('JOB_TITLE_TH', ''))).lower()
                if job_rescue_grp and any(t in u_title for t in job_rescue_grp):
                    title_rescue_indices.add(ui)
                elif j_title_lower and j_title_lower in u_title:
                    title_rescue_indices.add(ui)

        # Build candidate pool: top-50 semantic + title rescue (up to 100 total)
        top_k = min(50, len(user_records))
        top_results = torch.topk(cos_scores, k=top_k)
        semantic_indices = set(int(x) for x in top_results.indices)
        combined_indices = list(semantic_indices | title_rescue_indices)

        # Assign semantic scores
        sem_score_map = {}
        scaled_sem = scale_scores_minmax(top_results.values, 50.0, 90.0)
        for i, idx in enumerate(top_results.indices):
            sem_score_map[int(idx)] = float(scaled_sem[i])
        # Title rescue candidates that weren't in semantic top → give midpoint score
        for ui in title_rescue_indices:
            if ui not in sem_score_map:
                sem_score_map[ui] = float(util.cos_sim(jd_emb, user_matrix[ui:ui+1])[0][0]) * 40 + 50
                sem_score_map[ui] = max(50.0, min(90.0, sem_score_map[ui]))

        # Pre-encode required_skills once for all candidates
        precomputed_req_skill_embs = None
        if bert_model and required_skills:
            try:
                precomputed_req_skill_embs = bert_model.encode(required_skills, convert_to_tensor=True)
            except: pass

        # Pre-encode all candidate titles in one batch
        cand_title_texts = [str(user_records[idx].get('JOB_TITLE_EN', user_records[idx].get('JOB_TITLE_TH', ''))).lower() for idx in combined_indices]
        cand_title_embs = None
        if bert_model and job_title_emb is not None:
            try:
                cand_title_embs = bert_model.encode(cand_title_texts, convert_to_tensor=True)
            except: pass

        # Collect all candidate skill lists, then batch encode in one call
        cand_skill_lists = []
        for idx in combined_indices:
            lid = str(user_records[idx].get('LEARNER_ID', ''))
            cand_skill_lists.append([s['name'] for s in learner_skills.get(lid, [])])

        all_skills_flat = [s for sl in cand_skill_lists for s in sl]
        skill_span_ends = []
        pos = 0
        for sl in cand_skill_lists:
            pos += len(sl)
            skill_span_ends.append(pos)

        all_skill_embs = None
        if bert_model and all_skills_flat:
            try:
                all_skill_embs = bert_model.encode(all_skills_flat, convert_to_tensor=True)
            except: pass

        pool = []
        for ci, idx in enumerate(combined_indices):
            i = idx  # alias for compatibility
            user = user_records[idx]
            lid = str(user.get('LEARNER_ID',''))
            sub_id = str(user.get('SUBDISTRICT_ID',''))
            semantic = sem_score_map.get(idx, 55.0)
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

            skills_list_raw = learner_skills.get(lid, [])
            skills_list = cand_skill_lists[ci]

            # Slice pre-encoded user skill embeddings for this candidate
            span_end = skill_span_ends[ci]
            span_start = skill_span_ends[ci - 1] if ci > 0 else 0
            user_skill_embs_slice = all_skill_embs[span_start:span_end] if all_skill_embs is not None and span_end > span_start else None

            skill_level_penalty = 0.0
            if skill_level and skill_level != 'ไม่ระบุระดับ':
                lvl_num = {'Beginner (ขั้นต้น)': 1, 'Intermediate (กลาง)': 2, 'Advanced (สูง)': 3, 'Expert (เชี่ยวชาญ)': 4}.get(skill_level, 0)
                user_avg_lvl = 1.5
                if skills_list_raw:
                    lvls = []
                    for s in skills_list_raw:
                        ltext = str(s.get('level', '')).lower()
                        if 'expert' in ltext or '4' in ltext: lvls.append(4)
                        elif 'high' in ltext or 'advanced' in ltext or '3' in ltext: lvls.append(3)
                        elif 'mid' in ltext or 'intermediate' in ltext or '2' in ltext: lvls.append(2)
                        else: lvls.append(1)
                    if lvls: user_avg_lvl = sum(lvls)/len(lvls)

                if lvl_num > user_avg_lvl:
                    skill_level_penalty = (lvl_num - user_avg_lvl) * 10.0

            matched_skills, gap, skill_score, matched_explanations = get_skill_match_and_gap(skills_list, required_skills, precomputed_req_skill_embs, user_skill_embs_slice)
            skill_score = max(0, skill_score - skill_level_penalty)

            user_title_text = cand_title_texts[ci]
            title_match_score = 0.0
            raw_title_sim = 0.60
            if job_title_emb is not None and user_title_text:
                if cand_title_embs is not None:
                    raw_title_sim = float(util.cos_sim(job_title_emb, cand_title_embs[ci:ci+1])[0][0])
                else:
                    raw_title_sim = float(util.cos_sim(job_title_emb, bert_model.encode(user_title_text, convert_to_tensor=True))[0][0])
                if raw_title_sim >= 0.95:
                    title_match_score = 100.0
                elif raw_title_sim >= 0.85:
                    title_match_score = 60.0 + (raw_title_sim - 0.85) / 0.10 * 30.0
                else:
                    title_match_score = max(5.0, 60.0 * ((max(0, raw_title_sim - 0.75)) / 0.10) ** 2)

            # Tech synonym boost
            tech_synonyms = ['programmer', 'developer', 'software engineer', 'software developer', 'web developer', 'full stack']
            if any(s in job_title.lower() for s in tech_synonyms) and any(s in user_title_text for s in tech_synonyms):
                title_match_score = max(title_match_score, 85.0 if raw_title_sim > 0.80 else 70.0)

            # Data role disambiguation — prevent cross-role high similarity within data domain
            DATA_ROLE_GROUPS = [
                ['data engineer', 'etl engineer', 'data platform engineer', 'dataops', 'data pipeline'],
                ['data scientist', 'research scientist', 'machine learning scientist'],
                ['ml engineer', 'machine learning engineer', 'ai engineer', 'mlops engineer'],
                ['data analyst', 'bi analyst', 'business analyst', 'analytics engineer'],
            ]
            j_lower = job_title.lower()
            job_grp = next((g for g in DATA_ROLE_GROUPS if any(t in j_lower for t in g)), None)
            usr_grp = next((g for g in DATA_ROLE_GROUPS if any(t in user_title_text for t in g)), None)
            if job_grp and usr_grp:
                if job_grp == usr_grp:
                    # Same data role family → boost
                    title_match_score = max(title_match_score, 88.0)
                    raw_title_sim = max(raw_title_sim, 0.92)
                else:
                    # Different data role family → penalize hard
                    raw_title_sim = min(raw_title_sim, 0.65)   # fix: ใช้ 0.65 ให้ < 0.72 fire
                    title_match_score = min(title_match_score, 10.0)

            title_match_score = round(min(title_match_score, 100.0), 1)

            raw_final = (
                title_match_score * w['title'] +
                skill_score       * w['skill'] +
                budget_score      * w['budget'] +
                cert_score        * w['cert'] +
                semantic          * w['semantic']
            )

            # Hard cap by title similarity tier
            if raw_title_sim < 0.72:
                raw_final = min(raw_final, 45.0)   # cross-role data → max 45%
            elif raw_title_sim < 0.80:
                raw_final = min(raw_final, 65.0)
            elif raw_title_sim < 0.88:
                raw_final = min(raw_final, 85.0)
            elif raw_title_sim < 0.92:
                raw_final = min(raw_final, 95.0)

            # Extra guard: title_match ต่ำมาก → cap final ที่ 40%
            if title_match_score < 12.0:
                raw_final = min(raw_final, 40.0)

            user_work_type = str(pref.get('work_type', 'ไม่ระบุ'))
            work_type_penalty = 0.0
            if user_work_type.lower() != 'ไม่ระบุ' and job_work_style_pref.lower() != 'ไม่ระบุ' and job_work_style_pref.lower() != user_work_type.lower():
                if 'onsite' in user_work_type.lower() and 'remote' in job_work_style_pref.lower():
                    work_type_penalty = 20.0
                elif 'remote' in user_work_type.lower() and 'onsite' in job_work_style_pref.lower():
                    work_type_penalty = 20.0
                else:
                    work_type_penalty = 10.0

            # Location mismatch penalty
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
                'expected_salary': exp_sal,
                'fitness': fitness,
                'match_score': fitness,
                'semantic_score': round(semantic, 1),
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


@app.route('/api/parse_resume', methods=['POST'])
def parse_resume():
    """Extract structured profile data from uploaded PDF or DOCX resume."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = file.filename.lower()

        text = ''
        if filename.endswith('.pdf'):
            import fitz
            pdf = fitz.open(stream=file.read(), filetype='pdf')
            for page in pdf:
                text += page.get_text()
            pdf.close()
        elif filename.endswith('.docx'):
            import docx as docx_lib
            import io
            doc = docx_lib.Document(io.BytesIO(file.read()))
            text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            return jsonify({'error': 'Unsupported file type. Use PDF or DOCX'}), 400

        if not text.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400

        result = _parse_resume_text(text)
        result['raw_text_length'] = len(text)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _parse_resume_text(text):
    """Extract structured fields from resume text using regex + keyword matching."""
    import re

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    known_skills = set()
    if not df_acad_skill.empty and 'SKILL_NAME' in df_acad_skill.columns:
        known_skills = set(df_acad_skill['SKILL_NAME'].dropna().str.lower().tolist())
    known_skills.update([
        'python', 'sql', 'java', 'javascript', 'typescript', 'react', 'node.js',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas',
        'numpy', 'scikit-learn', 'docker', 'kubernetes', 'aws', 'gcp', 'azure',
        'git', 'linux', 'postgresql', 'mysql', 'mongodb', 'redis', 'tableau',
        'power bi', 'excel', 'c++', 'c#', 'golang', 'rust', 'kotlin', 'swift',
        'flutter', 'django', 'fastapi', 'flask', 'spring boot', 'devops', 'ci/cd',
        'nlp', 'computer vision', 'data analysis', 'statistics', 'r',
    ])

    text_lower = text.lower()

    found_skills = []
    for skill in sorted(known_skills):
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill.title() if len(skill) > 3 else skill.upper())

    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text_lower)
    email = email_match.group(0) if email_match else ''

    phone_match = re.search(r'(\+66|0)[0-9][\d\s\-]{7,11}', text)
    phone = re.sub(r'[\s\-]', '', phone_match.group(0)) if phone_match else ''

    salary = 0
    sal_match = re.search(r'(?:salary|เงินเดือน|ต้องการ)[^\d]*(\d{2,3}[,\d]*)', text_lower)
    if sal_match:
        salary = int(re.sub(r'\D', '', sal_match.group(1)))
    else:
        # Fall back to standalone number in the salary range 15,000-200,000
        for m in re.finditer(r'\b(\d{2,3},?\d{3})\b', text):
            val = int(re.sub(r'\D', '', m.group(1)))
            if 15000 <= val <= 200000:
                salary = val
                break

    edu_keywords = ['bachelor', 'master', 'phd', 'ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก',
                    'วท.บ', 'วท.ม', 'บธ.บ', 'วศ.บ', 'ศศ.บ']
    education = []
    for line in lines:
        if any(kw in line.lower() for kw in edu_keywords):
            education.append(line)

    exp_keywords = ['จำกัด', 'company', 'co.,ltd', 'co., ltd', 'บริษัท', 'corporation', 'inc.']
    experience = []
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in exp_keywords):
            experience.append(line)

    role_keywords = ['engineer', 'developer', 'analyst', 'scientist', 'manager',
                     'designer', 'consultant', 'architect', 'officer', 'director',
                     'นักพัฒนา', 'วิศวกร', 'นักวิเคราะห์']
    current_role = ''
    for line in lines[:10]:
        if any(kw in line.lower() for kw in role_keywords):
            current_role = line
            break

    return {
        'email':        email,
        'phone':        phone,
        'current_role': current_role,
        'skills':       found_skills[:20],
        'education':    education[:3],
        'experience':   experience[:5],
        'expected_salary': salary,
    }


@app.route('/api/market_stats', methods=['GET'])
def market_stats():
    try:
        total_jobs = len(df_job_post)
        total_users = len(df_learner)
        active_users = int(df_learner[df_learner['STATUS']=='active'].shape[0]) if not df_learner.empty else 0
        open_jobs_count = int(df_job_post[df_job_post['JOB_POST_STATUS']=='open'].shape[0]) if not df_job_post.empty else 0

        top_jobs = []
        if not df_job_post.empty:
            df_job_post['avg_sal'] = (pd.to_numeric(df_job_post['INCOME_MIN'],errors='coerce').fillna(0) +
                                       pd.to_numeric(df_job_post['INCOME_MAX'],errors='coerce').fillna(0)) / 2
            jg = df_job_post.groupby('POSITION_NAME_EN').agg(
                job_count=('JOB_POST_ID','count'),
                avg_salary=('avg_sal','mean')
            ).reset_index().sort_values('job_count', ascending=False).head(12)
            role_counts = df_learner['JOB_TITLE_EN'].value_counts().to_dict() if not df_learner.empty else {}
            for _, row in jg.iterrows():
                title = row['POSITION_NAME_EN']
                cand_count = role_counts.get(title, 0)
                ratio = round(cand_count / max(row['job_count'],1), 1)
                top_jobs.append({
                    'title': title, 'job_count': int(row['job_count']),
                    'avg_salary': int(row['avg_salary']),
                    'candidate_count': int(cand_count), 'competition': ratio
                })

        top_skills = []
        if not df_job_skill.empty and not df_acad_skill.empty:
            skill_counts = df_job_skill[df_job_skill['IS_CERTIFICATE'].astype(str).str.lower()!='true']['SKILL_ID'].value_counts().head(10)
            for skill_id, count in skill_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                top_skills.append({'skill': name, 'count': int(count)})

        top_certs = []
        if not df_job_skill.empty:
            cert_counts = df_job_skill[df_job_skill['IS_CERTIFICATE'].astype(str).str.lower()=='true']['SKILL_ID'].value_counts().head(8)
            for skill_id, count in cert_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                top_certs.append({'cert': name, 'count': int(count)})

        top_provinces = []
        if not df_job_post.empty:
            prov_counts = df_job_post['PROVINCE_ID'].value_counts().head(8)
            for prov_id, count in prov_counts.items():
                name = prov_id_map.get(str(prov_id), str(prov_id))
                top_provinces.append({'province': name, 'count': int(count)})

        salary_by_cat = []
        if not df_job_post.empty:
            catg = df_job_post.groupby('JOB_TYPE').agg(avg_min=('INCOME_MIN','mean'), avg_max=('INCOME_MAX','mean')).reset_index()
            for _, row in catg.iterrows():
                salary_by_cat.append({'category': row['JOB_TYPE'], 'avg_min': int(row['avg_min'] or 0), 'avg_max': int(row['avg_max'] or 0)})

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
    open_j = df_job_post[df_job_post['JOB_POST_STATUS']=='open'] if not df_job_post.empty else pd.DataFrame()
    sample_jobs = open_j['JOB_POST_ID'].head(5).tolist() if not open_j.empty else []
    return jsonify({'sample_learner_ids': sample_learners, 'sample_job_ids': sample_jobs})


@app.route('/api/academy_skills', methods=['GET'])
def academy_skills():
    return jsonify({'skills': academy_skills_list, 'certs': cert_skills_list})


@app.route('/api/job_trends', methods=['GET'])
def get_job_trends():
    try:
        top_positions = []
        if not df_job_post.empty:
            pos_counts = df_job_post['POSITION_NAME_EN'].value_counts().head(10)
            for title, count in pos_counts.items():
                top_positions.append({'title': str(title), 'count': int(count)})

        top_skills = []
        if not df_job_skill.empty and not df_acad_skill.empty:
            skill_counts = df_job_skill[df_job_skill['IS_CERTIFICATE'].astype(str).str.lower()!='true']['SKILL_ID'].value_counts().head(10)
            for skill_id, count in skill_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                top_skills.append({'skill': name, 'count': int(count)})

        monthly_trends = []
        if not df_job_post.empty and 'OPEN_DATE' in df_job_post.columns:
            df_dates = pd.to_datetime(df_job_post['OPEN_DATE'], errors='coerce').dropna()
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
        months = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.']
        enrollment_data = [3800, 4200, 4100, 4800, 5200, 5600]

        top_impact = []
        if not df_acad_course.empty:
            sample_courses = df_acad_course.head(5)
            impact_vals = [48, 42, 39, 35, 28]
            for i, (_, row) in enumerate(sample_courses.iterrows()):
                top_impact.append({
                    'name': str(row.get('Course_name', 'Course')),
                    'impact': impact_vals[i] if i < len(impact_vals) else 20
                })

        skills = []
        market_demand = []
        course_supply = []

        if not df_job_skill.empty:
            skill_counts = df_job_skill[df_job_skill['IS_CERTIFICATE'].astype(str).str.lower()!='true']['SKILL_ID'].value_counts().head(6)
            for skill_id, count in skill_counts.items():
                name = skill_id_map.get(str(skill_id), str(skill_id))
                skills.append(name)
                max_c = skill_counts.iloc[0] if not skill_counts.empty else 1
                market_val = min(95, int((count / max_c) * 95))
                market_demand.append(market_val)
                import random
                course_supply.append(max(40, market_val - random.randint(5, 20)))

        if not skills:
            skills = ['Programming', 'Cloud', 'Data Science', 'English', 'Design', 'Management']
            market_demand = [85, 78, 92, 70, 65, 80]
            course_supply = [72, 60, 88, 75, 58, 70]

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
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
