import sqlite3
from datetime import date
import csv
import psycopg
from sentence_transformers import SentenceTransformer
import json

# -------------------------------
# Part (1): SQLite DB + Initial Data
# -------------------------------

DB_PATH = "bariatric_meds.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_name TEXT UNIQUE,
    generic_name TEXT,
    drug_class TEXT,
    indication TEXT,
    status_after_sleeve TEXT,
    reason TEXT,
    dose_adjustment_notes TEXT,
    administration_form TEXT,
    interactions TEXT,
    evidence_level TEXT,
    source_links TEXT,
    last_reviewed DATE,
    notes TEXT
);
"""

EXAMPLE_ENTRIES = [
    {
        "drug_name": "Ibuprofen",
        "generic_name": "ibuprofen",
        "drug_class": "NSAID",
        "indication": "مسكن/مضاد التهاب",
        "status_after_sleeve": "avoid",
        "reason": "يزيد خطر التقرحات بعد جراحات السمنة",
        "dose_adjustment_notes": "تجنب نهائيًا إن أمكن؛ استخدم باراسيتامول كبديل تحت إشراف طبي",
        "administration_form": "tablet",
        "interactions": "",
        "evidence_level": "guideline/review",
        "source_links": "https://www.sps.nhs.uk/articles/considerations-for-using-medicines-following-bariatric-surgery/",
        "last_reviewed": str(date.today()),
        "notes": "ممنوع في معظم الحالات بعد التكميم."
    },
    {
        "drug_name": "Omeprazole",
        "generic_name": "omeprazole",
        "drug_class": "PPI",
        "indication": "حماية من القرحة/علاج الارتجاع",
        "status_after_sleeve": "conditional",
        "reason": "يُستخدم كوقاية بعد الجراحة لفترة معينة",
        "dose_adjustment_notes": "عادة يُوصف لشهور بعد العملية",
        "administration_form": "capsule",
        "interactions": "",
        "evidence_level": "guideline",
        "source_links": "https://pubmed.ncbi.nlm.nih.gov/",
        "last_reviewed": str(date.today()),
        "notes": "المدة تختلف حسب تعليمات الجرّاح."
    },
    {
        "drug_name": "Paracetamol",
        "generic_name": "acetaminophen",
        "drug_class": "Analgesic",
        "indication": "مسكن ألم",
        "status_after_sleeve": "allowed",
        "reason": "آمن كمسكن بعد التكميم",
        "dose_adjustment_notes": "انتبه للجرعة القصوى اليومية",
        "administration_form": "tablet/liquid",
        "interactions": "",
        "evidence_level": "guideline",
        "source_links": "https://www.ssmhealth.com/",
        "last_reviewed": str(date.today()),
        "notes": "البديل المفضل للمسكنات بعد التكميم."
    }
]

def create_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()
    conn.close()

def insert_entry(entry, path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cols = ",".join(entry.keys())
    placeholders = ",".join(["?"] * len(entry))
    sql = f"INSERT OR IGNORE INTO meds ({cols}) VALUES ({placeholders})"
    cur.execute(sql, tuple(entry.values()))
    conn.commit()
    conn.close()

def seed_examples():
    for e in EXAMPLE_ENTRIES:
        insert_entry(e)

def export_csv(csv_path="bariatric_meds_export.csv", path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM meds")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    return csv_path

# -------------------------------
# Part (2): PostgreSQL Connection + Setup
# -------------------------------

def get_connection():
    return psycopg.connect("postgresql://postgres:12345@localhost:5432/postgres")

def create_postgres_table():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS meds (
            id SERIAL PRIMARY KEY,
            drug_name TEXT UNIQUE,
            generic_name TEXT,
            drug_class TEXT,
            indication TEXT,
            status_after_sleeve TEXT,
            reason TEXT,
            dose_adjustment_notes TEXT,
            administration_form TEXT,
            interactions TEXT,
            evidence_level TEXT,
            source_links TEXT,
            last_reviewed DATE,
            notes TEXT,
            embedding JSONB
        );
        """)
    conn.commit()
    conn.close()

def seed_postgres_examples():
    conn = get_connection()
    with conn.cursor() as cur:
        for e in EXAMPLE_ENTRIES:
            cols = ",".join(e.keys())
            placeholders = ",".join(["%s"] * len(e))
            sql = f"""
                INSERT INTO meds ({cols}) VALUES ({placeholders})
                ON CONFLICT (drug_name) DO NOTHING
            """
            cur.execute(sql, tuple(e.values()))
    conn.commit()
    conn.close()

def test_postgres_connection():
    try:
        conn = get_connection()
        print("✅ Connected to PostgreSQL successfully!")
        conn.close()
    except Exception as e:
        print("❌ Error connecting to PostgreSQL:", e)

# -------------------------------
# Part (3): Embeddings Update
# -------------------------------

model = SentenceTransformer("BAAI/bge-base-en")

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True).tolist()

def update_embeddings():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id, drug_name, generic_name, indication FROM meds WHERE embedding IS NULL;")
        rows = cur.fetchall()
        for row in rows:
            drug_id, drug_name, generic_name, indication = row
            text = f"{drug_name} ({generic_name}) - {indication}"
            emb = get_embeddings([text])[0]
            cur.execute("UPDATE meds SET embedding = %s WHERE id = %s;", (json.dumps(emb), drug_id))
    conn.commit()
    conn.close()
    print("✅ Embeddings updated successfully!")

# -------------------------------
# Part (4): Insert & Search (for chatbot)
# -------------------------------

def insert_drug(entry: dict):
    """Insert drug into PostgreSQL with embedding."""
    conn = get_connection()
    with conn.cursor() as cur:
        cols = ",".join(entry.keys())
        placeholders = ",".join(["%s"] * len(entry))
        sql = f"""
            INSERT INTO meds ({cols}) VALUES ({placeholders})
            ON CONFLICT (drug_name) DO NOTHING
            RETURNING id;
        """
        cur.execute(sql, tuple(entry.values()))
        result = cur.fetchone()
        if result:
            drug_id = result[0]
            text = f"{entry['drug_name']} ({entry['generic_name']}) - {entry['indication']}"
            emb = get_embeddings([text])[0]
            cur.execute("UPDATE meds SET embedding = %s WHERE id = %s;", (json.dumps(emb), drug_id))
            print(f"✅ Drug '{entry['drug_name']}' inserted with embedding.")
        else:
            print(f"⚠️ Drug '{entry['drug_name']}' already exists. Skipped.")
    conn.commit()
    conn.close()

def search_drug(query: str):
    """Search drug by name in PostgreSQL and return details."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT drug_name, generic_name, drug_class, indication,
                   status_after_sleeve, reason, dose_adjustment_notes,
                   administration_form, interactions, evidence_level,
                   source_links, notes
            FROM meds
            WHERE LOWER(drug_name) = LOWER(%s)
               OR LOWER(generic_name) = LOWER(%s)
            LIMIT 1;
        """, (query, query))
        row = cur.fetchone()
    conn.close()

    if row:
        return {
            "drug_name": row[0],
            "generic_name": row[1],
            "drug_class": row[2],
            "indication": row[3],
            "status_after_sleeve": row[4],
            "reason": row[5],
            "dose_adjustment_notes": row[6],
            "administration_form": row[7],
            "interactions": row[8],
            "evidence_level": row[9],
            "source_links": row[10],
            "notes": row[11],
        }
    return None

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # SQLite
    create_db()
    seed_examples()
    export_csv()

    # PostgreSQL
    create_postgres_table()
    seed_postgres_examples()
    test_postgres_connection()
    update_embeddings()

    # Test search
    print(search_drug("Omeprazole"))





