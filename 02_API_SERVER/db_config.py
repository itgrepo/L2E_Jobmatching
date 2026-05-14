import pandas as pd
from pathlib import Path

_here = Path(__file__).resolve().parent
DATA_DIR = _here.parent / 'DATA_SET_V2'

print(f"[db_config] DATA_DIR: {DATA_DIR}")


def _read(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"[db_config] WARNING: {path} not found, returning empty DataFrame")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def fetch_learner() -> pd.DataFrame:
    return _read('TBL_LEARNER.csv')


def fetch_skills() -> pd.DataFrame:
    return _read('TBL_LEARNER_SKILL.csv')


def fetch_certs() -> pd.DataFrame:
    return _read('TBL_LEARNER_CERTIFICATEANDSKILL.csv')


def fetch_education() -> pd.DataFrame:
    return _read('TBL_LEARNER_EDUCATION.csv')


def fetch_experience() -> pd.DataFrame:
    return _read('TBL_LEARNER_EXPERINCE.csv')


def fetch_preference() -> pd.DataFrame:
    return _read('TBL_LEARNER_PREFERENCE.csv')


def fetch_job_post() -> pd.DataFrame:
    df = _read('job_post.csv')
    return df


def fetch_job_skill() -> pd.DataFrame:
    return _read('job_post_skill.csv')


def fetch_company() -> pd.DataFrame:
    return _read('company.csv')


def fetch_university() -> pd.DataFrame:
    df = _read('TBM_UNIVERSITY.csv')
    if df.empty:
        return df
    return df[['UNIVERSITY_ID', 'UNIVERSITY_NAME']].drop_duplicates()


def fetch_branch() -> pd.DataFrame:
    df = _read('TBM_BRANCH.csv')
    if df.empty:
        return df
    return df[['BRANCHID', 'BRANCH_NAME']].drop_duplicates()


def fetch_edu_level() -> pd.DataFrame:
    df = _read('TBM_EDUCATIONBACKBROUND.csv')
    if df.empty:
        return df
    return df[['EDUCATOINBG_ID', 'EDUCATOINBG_NAME']].drop_duplicates()


def fetch_position() -> pd.DataFrame:
    return _read('TBM_POSITION.csv')


def fetch_worktype() -> pd.DataFrame:
    return _read('TBM_WORKTYPE.csv')


def fetch_jobtype() -> pd.DataFrame:
    return _read('TBM_JOBTYPE.csv')


def fetch_postcode() -> pd.DataFrame:
    return _read('TBM_POSTCODE.csv')


def fetch_acad_skill() -> pd.DataFrame:
    df = _read('Academy_skill_and_certificate.csv')
    if df.empty:
        return df
    if 'IS_CERTIFICATE' not in df.columns:
        df['IS_CERTIFICATE'] = False
    else:
        df['IS_CERTIFICATE'] = df['IS_CERTIFICATE'].map(lambda x: str(x).strip().lower() == 'true')
    return df


def fetch_acad_course() -> pd.DataFrame:
    df = _read('Academy_course.csv')
    if df.empty:
        return df
    # Rename COURSE_NAME back to the mixed-case key api_server.py expects
    if 'COURSE_NAME' in df.columns:
        df = df.rename(columns={'COURSE_NAME': 'Course_name'})
    return df


if __name__ == "__main__":
    print("ทดสอบ CSV Loading...")
    df = fetch_learner()
    print(f"TBL_LEARNER: {len(df)} rows, columns: {df.columns.tolist()}")
    df = fetch_job_post()
    print(f"job_post: {len(df)} rows, columns: {df.columns.tolist()}")
    df = fetch_skills()
    print(f"TBL_LEARNER_SKILL: {len(df)} rows")
    df = fetch_acad_skill()
    print(f"Academy_skill_and_certificate: {len(df)} rows")
    print("เสร็จสิ้น")
