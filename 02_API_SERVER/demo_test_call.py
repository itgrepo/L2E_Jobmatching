"""
=======================================================
 DEMO: จำลองการที่ SRS เรียก AI Service
=======================================================
 รันไฟล์นี้หลังจากเปิด demo_ai_service.py แล้ว
=======================================================
"""

import requests
import json

AI_SERVICE_URL = "http://localhost:5050/api/fill_pool"

# =========================================
# จำลอง SRS ส่งข้อมูลมาให้ AI
# =========================================
payload = {
    "pool_id": "POOL-2025-001",

    # SRS ส่ง job detail มาให้ครบ
    "job_detail": {
        "title": "Data Engineer",
        "description": "ต้องการผู้เชี่ยวชาญด้าน Data Pipeline และ ETL ที่มีประสบการณ์ด้าน Big Data",
        "required_skills": ["Python", "SQL", "Apache Spark", "Airflow", "AWS"]
    },

    # SRS ส่ง candidate list มาให้ครบ
    "candidates": [
        {
            "candidate_id": "C001",
            "name": "สมชาย ใจดี",
            "skills": ["Python", "SQL", "Apache Spark", "Hadoop", "AWS"],
            "about": "มีประสบการณ์ด้าน Data Engineering 3 ปี เชี่ยวชาญ ETL และ Data Pipeline"
        },
        {
            "candidate_id": "C002",
            "name": "วิไล มีสุข",
            "skills": ["Python", "Machine Learning", "TensorFlow", "Keras"],
            "about": "Data Scientist ที่เชี่ยวชาญด้าน ML Model และ Deep Learning"
        },
        {
            "candidate_id": "C003",
            "name": "ประสิทธิ์ เก่งมาก",
            "skills": ["SQL", "Airflow", "dbt", "Snowflake", "Python"],
            "about": "Analytics Engineer มีประสบการณ์ด้าน Data Warehouse 4 ปี"
        },
        {
            "candidate_id": "C004",
            "name": "มานี รักเรียน",
            "skills": ["Excel", "PowerPoint", "Word"],
            "about": "ทำงานด้านธุรการมา 5 ปี ไม่มีพื้นฐาน IT"
        },
        {
            "candidate_id": "C005",
            "name": "กิตติ ดีงาม",
            "skills": ["Python", "SQL", "Kafka", "Spark", "Airflow", "GCP"],
            "about": "Senior Data Engineer ประสบการณ์ 5 ปี ด้าน Streaming Data Pipeline"
        },
    ]
}

# =========================================
# เรียก AI Service
# =========================================
print("=" * 55)
print("📤 SRS กำลังส่งข้อมูลไปให้ AI Service...")
print(f"   Pool ID     : {payload['pool_id']}")
print(f"   Job Title   : {payload['job_detail']['title']}")
print(f"   Candidates  : {len(payload['candidates'])} คน")
print("=" * 55)

try:
    response = requests.post(AI_SERVICE_URL, json=payload)
    result = response.json()

    print("\n✅ AI คำนวณเสร็จแล้ว! ผลลัพธ์ที่ได้:")
    print("=" * 55)
    print(f"   Pool ID : {result['pool_id']}")
    print(f"   พบ Candidate ที่เหมาะสม: {result['total']} คน")
    print()
    print("   อันดับ  | ชื่อ              | Score")
    print("   " + "-" * 45)

    for i, c in enumerate(result['matched_candidates'], 1):
        bar = "█" * int(c['score'] / 10)
        print(f"   #{i:<5} | {c['name']:<18} | {c['score']:>5}%  {bar}")

    print()
    print("=" * 55)
    print("💡 AI แค่ Return list นี้กลับมา")
    print("   SRS เอา list นี้ไป INSERT ลง Database เอง")
    print("   AI ไม่ได้แตะ Database เลย! ✅")
    print("=" * 55)

    print("\n📦 Raw JSON ที่ AI ส่งกลับมา:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("   กรุณาตรวจสอบว่าเปิด demo_ai_service.py แล้วหรือยังครับ")
