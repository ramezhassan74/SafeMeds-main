# filename: bariatric_meds_setup.py
import sqlite3
from datetime import date
import csv
import psycopg
from sentence_transformers import SentenceTransformer

# -------------------------------
# جزء (1): إنشاء قاعدة بيانات SQLite + إدخال بيانات أولية
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
        "drug_name":"Ibuprofen",
        "generic_name":"ibuprofen",
        "drug_class":"NSAID",
        "indication":"مسكن/مضاد التهاب",
        "status_after_sleeve":"avoid",
        "reason":"يزيد خطر التقرحات بعد جراحات السمنة",
        "dose_adjustment_notes":"تجنب نهائيًا إن أمكن؛ استخدم باراسيتامول كبديل تحت إشراف طبي",
        "administration_form":"tablet",
        "interactions":"",
        "evidence_level":"guideline/review",
        "source_links":"https://www.sps.nhs.uk/articles/considerations-for-using-medicines-following-bariatric-surgery/",
        "last_reviewed": str(date.today()),
        "notes":"ممنوع في معظم الحالات بعد التكميم."
    },
    {
        "drug_name":"Omeprazole",
        "generic_name":"omeprazole",
        "drug_class":"PPI",
        "indication":"حماية من القرحة/علاج الارتجاع",
        "status_after_sleeve":"conditional",
        "reason":"يُستخدم كوقاية بعد الجراحة لفترة معينة",
        "dose_adjustment_notes":"عادة يُوصف لشهور بعد العملية",
        "administration_form":"capsule",
        "interactions":"",
        "evidence_level":"guideline",
        "source_links":"https://pubmed.ncbi.nlm.nih.gov/",
        "last_reviewed": str(date.today()),
        "notes":"المدة تختلف حسب تعليمات الجرّاح."
    },
    {
        "drug_name":"Paracetamol",
        "generic_name":"acetaminophen",
        "drug_class":"Analgesic",
        "indication":"مسكن ألم",
        "status_after_sleeve":"allowed",
        "reason":"آمن كمسكن بعد التكميم",
        "dose_adjustment_notes":"انتبه للجرعة القصوى اليومية",
        "administration_form":"tablet/liquid",
        "interactions":"",
        "evidence_level":"guideline",
        "source_links":"https://www.ssmhealth.com/",
        "last_reviewed": str(date.today()),
        "notes":"البديل المفضل للمسكنات بعد التكميم."
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
    placeholders = ",".join(["?"]*len(entry))
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
# جزء (2): PostgreSQL Connection + Table + Seed + Search Function
# -------------------------------
def get_connection():
    return psycopg.connect("postgresql://postgres:12345@localhost:5432/postgres")

def create_postgres_table():
    conn = get_connection()
    cur = conn.cursor()
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
        embedding VECTOR(384) -- عشان نخزن الـ embeddings (bge-small-en = 384)
    );
    """)
    conn.commit()
    conn.close()

def seed_postgres_examples():
    conn = get_connection()
    cur = conn.cursor()
    for e in EXAMPLE_ENTRIES:
        cols = ",".join(e.keys())
        placeholders = ",".join(["%s"]*len(e))
        sql = f"INSERT INTO meds ({cols}) VALUES ({placeholders}) ON CONFLICT (drug_name) DO NOTHING"
        cur.execute(sql, tuple(e.values()))
    conn.commit()
    conn.close()

def search_drug(drug_name: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT drug_name, generic_name, drug_class, indication,
                       status_after_sleeve, reason, dose_adjustment_notes,
                       administration_form, notes
                FROM meds
                WHERE LOWER(drug_name) LIKE LOWER(%s)
                   OR LOWER(generic_name) LIKE LOWER(%s)
                LIMIT 1;
            """, (f"%{drug_name}%", f"%{drug_name}%"))
            row = cur.fetchone()
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
                    "notes": row[8],
                }
            return None
    finally:
        conn.close()

def test_postgres_connection():
    try:
        conn = get_connection()
        print("✅ Connected to PostgreSQL successfully!")
        conn.close()
    except Exception as e:
        print("❌ Error connecting to PostgreSQL:", e)

# -------------------------------
# جزء (3): Embeddings Update
# -------------------------------
model = SentenceTransformer("BAAI/bge-small-en")

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
            cur.execute("UPDATE meds SET embedding = %s WHERE id = %s;", (emb, drug_id))
        conn.commit()
    conn.close()
    print("✅ Embeddings updated successfully!")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # SQLite setup
    create_db()
    seed_examples()
    csv_file = export_csv()
    print(f"📂 SQLite DB created at: {DB_PATH}")
    print(f"📂 CSV export created at: {csv_file}")

    # PostgreSQL setup
    create_postgres_table()
    seed_postgres_examples()
    test_postgres_connection()
    print("🔍 Test search:", search_drug("paracetamol"))

    # Update embeddings
    update_embeddings()


def add_embeddings_column():
    conn = get_connection()
    cur = conn.cursor()
    
    # ✅ فعل extension pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name='meds' AND column_name='embedding'
            ) THEN
                ALTER TABLE meds ADD COLUMN embedding VECTOR(384);
            END IF;
        END$$;
    """)
    conn.commit()
    conn.close()
