# ============================================================
# PostgreSQL 数据导入脚本
# 功能：从 medical.json 导入医学数据到已建好的表
# 前置条件：已通过 Alembic 建表（alembic upgrade head）
# 用法: cd tiangong-agent && python scripts/init_postgres.py
# ============================================================

import json
import sys
import logging
from pathlib import Path

# 把 tiangong-agent/ 加入 Python 路径，让 src.* 可以导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# medical.json 路径：项目根目录 data/raw/medical.json
MEDICAL_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "medical.json"

def load_medical_data(filepath: Path) -> list[dict]:
    """
    加载 medical.json
    文件格式：每行一个 JSON 对象（JSONL 格式）
    """
    if not filepath.exists():
        logger.error(f"数据文件不存在: {filepath}")
        logger.error("请先执行: cp QASystemOnMedicalKG/data/medical.json data/raw/medical.json")
        sys.exit(1)

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    logger.info(f"加载了 {len(data)} 条疾病记录")
    return data

def import_data():
    import psycopg2
    from tqdm import tqdm

    settings = get_settings()

    logger.info("连接 PostgreSQL...")
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        dbname=settings.DB_NAME,
    )
    conn.autocommit = False
    cur = conn.cursor()

    records = load_medical_data(MEDICAL_JSON_PATH)

    # ----------------------------------------------------------
    # 第 1 步：导入科室（departments）
    # medical.json 字段：department 或 cure_department（列表）
    # model 字段：name（唯一）
    # ----------------------------------------------------------
    logger.info("导入科室数据...")
    dept_set = set()
    for record in records:
        for dept in record.get("cure_department", record.get("department", [])):
            dept = dept.strip()
            if dept:
                dept_set.add(dept)

    dept_name_to_id: dict[str, int] = {}
    for dept_name in tqdm(dept_set, desc="科室"):
        cur.execute(
            "INSERT INTO departments (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
            (dept_name,),
        )
        row = cur.fetchone()
        if row:
            dept_name_to_id[dept_name] = row[0]
        else:
            cur.execute("SELECT id FROM departments WHERE name = %s", (dept_name,))
            dept_name_to_id[dept_name] = cur.fetchone()[0]

    conn.commit()
    logger.info(f"科室导入完成: {len(dept_name_to_id)} 个")

    # ----------------------------------------------------------
    # 第 2 步：导入症状（symptoms）
    # medical.json 字段：symptom（列表）
    # model 字段：name（唯一）
    # ----------------------------------------------------------
    logger.info("导入症状数据...")
    symptom_set = set()
    for record in records:
        for s in record.get("symptom", []):
            s = s.strip()
            if s:
                symptom_set.add(s)

    symptom_name_to_id: dict[str, int] = {}
    for symptom_name in tqdm(symptom_set, desc="症状"):
        cur.execute(
            "INSERT INTO symptoms (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
            (symptom_name,),
        )
        row = cur.fetchone()
        if row:
            symptom_name_to_id[symptom_name] = row[0]
        else:
            cur.execute("SELECT id FROM symptoms WHERE name = %s", (symptom_name,))
            symptom_name_to_id[symptom_name] = cur.fetchone()[0]

    conn.commit()
    logger.info(f"症状导入完成: {len(symptom_name_to_id)} 种")

    # ----------------------------------------------------------
    # 第 3 步：导入药品（drugs）
    # medical.json 字段：common_drug / recommand_drug（列表）
    # model 字段：name（非唯一，但实际去重插入）
    # 注意：medical.json 只有药品名，其余字段（alias/category 等）留空
    # ----------------------------------------------------------
    logger.info("导入药品数据...")
    drug_set = set()
    for record in records:
        for drug_name in record.get("common_drug", []) + record.get("recommand_drug", []):
            drug_name = drug_name.strip()
            if drug_name:
                drug_set.add(drug_name)

    drug_name_to_id: dict[str, int] = {}
    for drug_name in tqdm(drug_set, desc="药品"):
        # drugs.name 没有 UNIQUE 约束，用 SELECT 先查再插
        cur.execute("SELECT id FROM drugs WHERE name = %s", (drug_name,))
        row = cur.fetchone()
        if row:
            drug_name_to_id[drug_name] = row[0]
        else:
            cur.execute(
                "INSERT INTO drugs (name) VALUES (%s) RETURNING id",
                (drug_name,),
            )
            drug_name_to_id[drug_name] = cur.fetchone()[0]

    conn.commit()
    logger.info(f"药品导入完成: {len(drug_name_to_id)} 种")

    # ----------------------------------------------------------
    # 第 4 步：导入疾病 + 关联表
    # diseases：name / department_id(FK) / description / cause /
    #           prevent / cure_way / cure_lasttime / cured_prob /
    #           easy_get / cost_money
    # disease_symptoms：disease_id + symptom_id（FK，不是字符串）
    # disease_drugs：disease_id + drug_id + relation_type
    # ----------------------------------------------------------
    logger.info("导入疾病及关联数据...")
    disease_count = 0
    ds_count = 0   # disease_symptoms 行数
    dd_count = 0   # disease_drugs 行数

    for record in tqdm(records, desc="疾病"):
        name = record.get("name", "").strip()
        if not name:
            continue

        # 取第一个科室作为 department_id（一病对应一科室）
        depts = record.get("cure_department", record.get("department", []))
        dept_id = dept_name_to_id.get(depts[0].strip()) if depts else None

        # cure_way 是列表，存为逗号分隔字符串
        cure_way = "，".join(record.get("cure_way", [])) or None

        cur.execute(
            """
            INSERT INTO diseases
                (name, department_id, description, cause, prevent,
                 cure_way, cure_lasttime, cured_prob, easy_get, cost_money)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING
            RETURNING id
            """,
            (
                name,
                dept_id,
                record.get("desc") or None,
                record.get("cause") or None,
                record.get("prevent") or None,
                cure_way,
                record.get("cure_lasttime") or None,
                record.get("cured_prob") or None,
                record.get("easy_get") or None,
                record.get("cost_money") or None,
            ),
        )
        row = cur.fetchone()
        if not row:
            continue  # 已存在，跳过关联数据（避免重复）
        disease_id = row[0]
        disease_count += 1

        # 症状关联：disease_symptoms(disease_id, symptom_id)
        for s in record.get("symptom", []):
            s = s.strip()
            symptom_id = symptom_name_to_id.get(s)
            if symptom_id:
                cur.execute(
                    "INSERT INTO disease_symptoms (disease_id, symptom_id) VALUES (%s, %s)",
                    (disease_id, symptom_id),
                )
                ds_count += 1

        # 常用药关联
        for drug_name in record.get("common_drug", []):
            drug_name = drug_name.strip()
            drug_id = drug_name_to_id.get(drug_name)
            if drug_id:
                cur.execute(
                    "INSERT INTO disease_drugs (disease_id, drug_id, relation_type) VALUES (%s, %s, %s)",
                    (disease_id, drug_id, "common"),
                )
                dd_count += 1

        # 推荐药关联
        for drug_name in record.get("recommand_drug", []):
            drug_name = drug_name.strip()
            drug_id = drug_name_to_id.get(drug_name)
            if drug_id:
                cur.execute(
                    "INSERT INTO disease_drugs (disease_id, drug_id, relation_type) VALUES (%s, %s, %s)",
                    (disease_id, drug_id, "recommend"),
                )
                dd_count += 1

    conn.commit()

    # ----------------------------------------------------------
    # 输出统计
    # ----------------------------------------------------------
    logger.info("")
    logger.info("=" * 45)
    logger.info("PostgreSQL 数据导入完成")
    logger.info("=" * 45)
    logger.info(f"  科室:         {len(dept_name_to_id):>6} 个")
    logger.info(f"  症状:         {len(symptom_name_to_id):>6} 种")
    logger.info(f"  药品:         {len(drug_name_to_id):>6} 种")
    logger.info(f"  疾病:         {disease_count:>6} 条")
    logger.info(f"  疾病-症状关联: {ds_count:>6} 条")
    logger.info(f"  疾病-药品关联: {dd_count:>6} 条")

    cur.close()
    conn.close()

if __name__ == "__main__":
    import_data()