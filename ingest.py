# filename: ingest_meds.py
import psycopg
from sentence_transformers import SentenceTransformer

# إعدادات قاعدة البيانات
DB_URL = "postgresql://postgres:12345@localhost:5432/postgres"

# تحميل الموديل من HuggingFace
model = SentenceTransformer("BAAI/bge-small-en")

# -------------------------------
# DB utils
# -------------------------------
def get_connection():
    return psycopg.connect(DB_URL)

def add_embeddings_column():
    """
    يضيف عمود embeddings للجدول meds لو مش موجود
    """
    conn = get_connection()
    cur = conn.cursor()
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

# -------------------------------
# Ingest from DB
# -------------------------------
def ingest_meds():
    all_texts = []
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, drug_name, indication, notes FROM meds;")
    rows = cur.fetchall()

    for row in rows:
        med_id, drug_name, indication, notes = row
        text = " ".join([drug_name or "", indication or "", notes or ""]).strip()
        if not text:
            continue
        all_texts.append((med_id, text))

    print(f"📋 Loaded {len(all_texts)} meds from DB")
    conn.close()
    return all_texts

def update_embeddings():
    conn = get_connection()
    cur = conn.cursor()

    meds = ingest_meds()
    for med_id, text in meds:
        emb = model.encode([text])[0]  # numpy array
        cur.execute("UPDATE meds SET embedding = %s WHERE id = %s;", (emb.tolist(), med_id))

    conn.commit()
    conn.close()
    print(f"✅ Updated embeddings for {len(meds)} meds")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    add_embeddings_column()
    update_embeddings()


    