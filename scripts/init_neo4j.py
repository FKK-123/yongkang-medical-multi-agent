# ============================================================
# Neo4j 知识图谱初始化脚本
# 功能：从 medical.json 构建医学知识图谱
# 节点：7 类（Disease, Symptom, Drug, Department, Check, Food, Producer）
# 关系：8 类（medical.json 可提供的部分）
#   已实现：HAS_SYMPTOM / BELONGS_TO / COMMON_DRUG / RECOMMEND_DRUG /
#           NEED_CHECK / DO_EAT / NO_EAT / ACOMPANY_WITH
#   待补充：DRUG_INTERACTION / SAME_CLASS / PRODUCED_BY
#           （medical.json 无对应字段，需接入 NMPA 数据后补充）
# 前置条件：Neo4j 已启动，APOC 插件已加载
# 用法: cd yongkang-agent && python scripts/init_neo4j.py
# ============================================================

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDICAL_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "medical.json"

def load_medical_data(filepath: Path) -> list[dict]:
    """加载 medical.json（JSONL 格式，每行一个 JSON 对象）"""
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

def init_graph():
    from neo4j import GraphDatabase
    from tqdm import tqdm

    settings = get_settings()

    # --- 1. 连接 Neo4j ---
    logger.info("连接 Neo4j...")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Neo4j 连接成功")

    # --- 2. 清空旧数据（开发环境用，生产环境慎用）---
    logger.info("清空旧图谱数据...")
    with driver.session() as session:
        while True:
            result = session.run(
                "MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS deleted"
            )
            if result.single()["deleted"] == 0:
                break
    logger.info("旧数据已清空")

    # --- 3. 创建唯一性约束（自动建索引）---
    logger.info("创建约束和索引...")
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease)    REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Symptom)    REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Drug)       REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Department) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Check)      REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Food)       REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Producer)   REQUIRE n.name IS UNIQUE",
    ]
    with driver.session() as session:
        for cypher in constraints:
            session.run(cypher)
    logger.info("约束创建完成")

    # --- 4. 加载数据 ---
    records = load_medical_data(MEDICAL_JSON_PATH)

    # --- 5. 第一轮：收集所有实体（去重）---
    logger.info("收集实体...")
    entities: dict[str, set] = {
        "Disease":    set(),
        "Symptom":    set(),
        "Drug":       set(),
        "Department": set(),
        "Check":      set(),
        "Food":       set(),
        # Producer：medical.json 无生产商字段，留空，接入 NMPA 数据后补充
        "Producer":   set(),
    }

    for record in records:
        name = record.get("name", "").strip()
        if not name:
            continue

        entities["Disease"].add(name)

        for s in record.get("symptom", []):
            if s.strip():
                entities["Symptom"].add(s.strip())

        for d in record.get("common_drug", []) + record.get("recommand_drug", []):
            if d.strip():
                entities["Drug"].add(d.strip())

        for dep in record.get("cure_department", record.get("department", [])):
            if dep.strip():
                entities["Department"].add(dep.strip())

        for c in record.get("check", []):
            if c.strip():
                entities["Check"].add(c.strip())

        # DO_EAT 宜吃 + NO_EAT 忌吃，都是 Food 节点
        for f in record.get("recommand_eat", []) + record.get("not_eat", []):
            if f.strip():
                entities["Food"].add(f.strip())

    for label, names in entities.items():
        logger.info(f"  {label}: {len(names)} 个")

    # --- 6. 第二轮：批量创建节点 ---
    logger.info("创建节点...")

    def batch_create_nodes(session, label: str, names: set, batch_size: int = 500):
        """UNWIND + MERGE 批量写入，幂等"""
        name_list = list(names)
        for i in range(0, len(name_list), batch_size):
            batch = name_list[i : i + batch_size]
            session.run(
                f"UNWIND $names AS name MERGE (n:{label} {{name: name}})",
                names=batch,
            )

    with driver.session() as session:
        for label, names in entities.items():
            if names:
                batch_create_nodes(session, label, names)
            logger.info(f"  {label} 节点写入完成")

    # --- 7. 第三轮：收集关系对 ---
    # 关系名称与 phase1_data_import.md 保持一致
    relation_pairs: dict[str, list[tuple[str, str]]] = {
        "HAS_SYMPTOM":   [],   # 疾病 → 症状
        "BELONGS_TO":    [],   # 疾病 → 科室
        "COMMON_DRUG":   [],   # 疾病 → 常用药
        "RECOMMEND_DRUG":[],   # 疾病 → 推荐药
        "NEED_CHECK":    [],   # 疾病 → 检查项目
        "DO_EAT":        [],   # 疾病 → 宜吃食物
        "NO_EAT":        [],   # 疾病 → 忌吃食物
        "ACOMPANY_WITH": [],   # 疾病 → 并发症（疾病之间）
        # DRUG_INTERACTION / SAME_CLASS / PRODUCED_BY：medical.json 无此字段，暂不实现
    }

    for record in records:
        name = record.get("name", "").strip()
        if not name:
            continue

        for s in record.get("symptom", []):
            if s.strip():
                relation_pairs["HAS_SYMPTOM"].append((name, s.strip()))

        for dep in record.get("cure_department", record.get("department", [])):
            if dep.strip():
                relation_pairs["BELONGS_TO"].append((name, dep.strip()))

        for d in record.get("common_drug", []):
            if d.strip():
                relation_pairs["COMMON_DRUG"].append((name, d.strip()))

        for d in record.get("recommand_drug", []):
            if d.strip():
                relation_pairs["RECOMMEND_DRUG"].append((name, d.strip()))

        for c in record.get("check", []):
            if c.strip():
                relation_pairs["NEED_CHECK"].append((name, c.strip()))

        for f in record.get("recommand_eat", []):
            if f.strip():
                relation_pairs["DO_EAT"].append((name, f.strip()))

        for f in record.get("not_eat", []):
            if f.strip():
                relation_pairs["NO_EAT"].append((name, f.strip()))

        for comp in record.get("acompany", []):
            if comp.strip():
                relation_pairs["ACOMPANY_WITH"].append((name, comp.strip()))

    # --- 8. 批量写入关系 ---
    logger.info("创建关系...")

    def batch_create_relations(
        session,
        from_label: str,
        rel_type: str,
        to_label: str,
        pairs: list[tuple[str, str]],
        batch_size: int = 500,
    ):
        for i in range(0, len(pairs), batch_size):
            batch = [{"from_name": p[0], "to_name": p[1]} for p in pairs[i : i + batch_size]]
            session.run(
                f"""UNWIND $pairs AS pair
                    MATCH (a:{from_label} {{name: pair.from_name}})
                    MATCH (b:{to_label}   {{name: pair.to_name}})
                    MERGE (a)-[:{rel_type}]->(b)""",
                pairs=batch,
            )

    rel_config = [
        ("Disease", "HAS_SYMPTOM",    "Symptom"),
        ("Disease", "BELONGS_TO",     "Department"),
        ("Disease", "COMMON_DRUG",    "Drug"),
        ("Disease", "RECOMMEND_DRUG", "Drug"),
        ("Disease", "NEED_CHECK",     "Check"),
        ("Disease", "DO_EAT",         "Food"),
        ("Disease", "NO_EAT",         "Food"),
        ("Disease", "ACOMPANY_WITH",  "Disease"),
    ]

    with driver.session() as session:
        for from_label, rel_type, to_label in rel_config:
            pairs = relation_pairs[rel_type]
            if pairs:
                batch_create_relations(session, from_label, rel_type, to_label, pairs)
            logger.info(f"  {rel_type}: {len(pairs)} 条")

    # --- 9. 为 Disease 节点补充属性 ---
    logger.info("为疾病节点写入属性...")
    with driver.session() as session:
        for record in tqdm(records, desc="疾病属性"):
            name = record.get("name", "").strip()
            if not name:
                continue

            props = {
                "description":      record.get("desc", "") or "",
                "cause":            record.get("cause", "") or "",
                "prevent":          record.get("prevent", "") or "",
                "cure_way":         "，".join(record.get("cure_way", [])),
                "cured_prob":       record.get("cured_prob", "") or "",
                "cure_lasttime":    record.get("cure_lasttime", "") or "",
                "easy_get":         record.get("easy_get", "") or "",
                "cost_money":       record.get("cost_money", "") or "",
            }

            # 只 SET 非空字段，避免覆盖已有数据
            set_clauses = []
            params: dict = {"name": name}
            for key, value in props.items():
                if value:
                    set_clauses.append(f"d.{key} = ${key}")
                    params[key] = value

            if set_clauses:
                session.run(
                    f"MATCH (d:Disease {{name: $name}}) SET {', '.join(set_clauses)}",
                    **params,
                )

    # --- 10. 输出统计 ---
    total_nodes = sum(len(v) for v in entities.values())
    total_rels  = sum(len(v) for v in relation_pairs.values())

    logger.info("")
    logger.info("=" * 45)
    logger.info("Neo4j 知识图谱构建完成")
    logger.info("=" * 45)
    logger.info(f"  节点总数: {total_nodes}")
    logger.info(f"  关系总数: {total_rels}")
    logger.info("")
    logger.info("验证方式（Neo4j Browser: http://localhost:7474）:")
    logger.info("  MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC")
    logger.info("  MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC")
    logger.info('  MATCH (d:Disease {name:"糖尿病"})-[r]->(n) RETURN d,r,n LIMIT 30')

    driver.close()

if __name__ == "__main__":
    init_graph()
