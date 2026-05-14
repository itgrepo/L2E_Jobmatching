"""
=======================================================
 DEMO: AI Matching Service (แบบที่ 2 - AI คำนวณอย่างเดียว)
=======================================================
 สิ่งที่ Demo นี้แสดง:
   - SRS ส่ง job_detail + candidates มาให้ครบ
   - AI คำนวณ Matching Score
   - Return candidate list พร้อม score กลับ
   - AI ไม่ยุ่งกับ DB เลย

 วิธีรัน:
   python demo_ai_service.py

 วิธีทดสอบ (เปิด Terminal อีกอันแล้วรัน):
   python demo_test_call.py
=======================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import traceback

app = Flask(__name__)
CORS(app)

# โหลด Model ครั้งเดียวตอน start
print("⏳ กำลังโหลด AI Model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ โหลด Model สำเร็จ!")


# ==========================================
# ENDPOINT: /api/fill_pool
# ==========================================
# INPUT จาก SRS:
# {
#   "pool_id": "P001",
#   "job_detail": {
#       "title": "Data Engineer",
#       "description": "...",
#       "required_skills": ["Python", "SQL", "Spark"]
#   },
#   "candidates": [
#       {
#           "candidate_id": "C001",
#           "name": "สมชาย ใจดี",
#           "skills": ["Python", "SQL", "Machine Learning"],
#           "about": "มีประสบการณ์ด้าน Data Engineering 3 ปี"
#       },
#       ...
#   ]
# }
#
# OUTPUT กลับ SRS:
# {
#   "pool_id": "P001",
#   "matched_candidates": [
#       { "candidate_id": "C001", "name": "...", "score": 87.5 },
#       ...
#   ],
#   "total": 3
# }
# ==========================================

@app.route('/api/fill_pool', methods=['POST'])
def fill_pool():
    try:
        data = request.json

        pool_id     = data.get('pool_id', '')
        job_detail  = data.get('job_detail', {})
        candidates  = data.get('candidates', [])

        if not job_detail or not candidates:
            return jsonify({'error': 'กรุณาส่ง job_detail และ candidates มาด้วยครับ'}), 400

        # สร้าง Job Text จากข้อมูลที่ SRS ส่งมา
        job_title       = job_detail.get('title', '')
        job_description = job_detail.get('description', '')
        required_skills = job_detail.get('required_skills', [])
        job_text = f"{job_title} {job_description} {' '.join(required_skills)}"

        # Encode Job ด้วย BERT
        job_embedding = model.encode(job_text, convert_to_tensor=True)

        results = []
        for candidate in candidates:
            candidate_id = candidate.get('candidate_id', '')
            name         = candidate.get('name', '')
            skills       = candidate.get('skills', [])
            about        = candidate.get('about', '')

            # สร้าง Candidate Text
            candidate_text = f"{' '.join(skills)} {about}"

            # Encode Candidate ด้วย BERT
            candidate_embedding = model.encode(candidate_text, convert_to_tensor=True)

            # คำนวณ Cosine Similarity Score
            score = float(util.cos_sim(job_embedding, candidate_embedding)[0][0])
            score_percent = round(score * 100, 1)

            results.append({
                'candidate_id': candidate_id,
                'name':         name,
                'score':        score_percent,
                'skills':       skills,
            })

        # เรียงจาก score สูงสุด
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        # =========================================
        # AI แค่ return กลับ ไม่ได้ insert DB เอง!
        # SRS เอาผลนี้ไป insert ลง pool pool เอง
        # =========================================
        return jsonify({
            'pool_id':            pool_id,
            'matched_candidates': results,
            'total':              len(results),
            'status':             'success'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'fail'}), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': '🤖 AI Matching Service กำลังทำงานอยู่ครับ!',
        'endpoints': {
            'POST /api/fill_pool': 'ส่ง job_detail + candidates มา → รับ matched list กลับ'
        }
    })


if __name__ == '__main__':
    print("\n🚀 AI Demo Service เริ่มทำงานที่ http://localhost:5050")
    print("📌 ทดสอบได้โดยรัน: python demo_test_call.py\n")
    app.run(debug=True, port=5050)
