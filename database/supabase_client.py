"""
Supabase Integration for Telecom Churn Pipeline
Replaces CSV files with PostgreSQL via Supabase
"""
import os
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

# Try supabase-py, fallback to psycopg2
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    print("[WARNING] supabase-py not installed. Using psycopg2 fallback.")
    print("Install: pip install supabase")


class SupabaseDB:
    """
    Unified database interface for the churn pipeline.
    Uses Supabase (PostgreSQL) instead of CSV files.
    """

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.password = os.getenv("SUPABASE_PASSWORD")
        self.host = os.getenv("SUPABASE_HOST")
        self.db = os.getenv("SUPABASE_DB", "postgres")
        self.user = os.getenv("SUPABASE_USER", "postgres")
        self.port = int(os.getenv("SUPABASE_PORT", "5432"))

        self.client: Optional[Client] = None
        self._connect()

    def _connect(self):
        """Initialize Supabase client or psycopg2 fallback."""
        if HAS_SUPABASE and self.url and self.key:
            self.client = create_client(self.url, self.key)
            print("[INFO] Connected to Supabase via REST API")
        else:
            # Fallback: direct PostgreSQL connection
            import psycopg2
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.db,
                user=self.user,
                password=self.password,
                port=self.port
            )
            print("[INFO] Connected to Supabase via psycopg2")

    # ============================================
    # SCRAPED COMMENTS (replaces JSON/CSV)
    # ============================================

    def insert_scraped_comments(self, comments: List[Dict]) -> bool:
        """
        Insert scraped Facebook comments from scrape_facebook.py output.

        Args:
            comments: List of dicts with keys: customer_id, post_url, username, text, scraped_at
        """
        if not comments:
            return False

        if self.client:
            # Supabase REST API
            data = [
                {
                    "customer_id": c["customer_id"],
                    "post_url": c["post_url"],
                    "username": c["username"],
                    "text": c["text"],
                    "scraped_at": c.get("scraped_at", datetime.utcnow().isoformat())
                }
                for c in comments
            ]
            response = self.client.table("scraped_comments").insert(data).execute()
            print(f"[INFO] Inserted {len(data)} scraped comments")
            return True
        else:
            # psycopg2 batch insert
            cur = self.conn.cursor()
            cur.executemany(
                """
                INSERT INTO scraped_comments (customer_id, post_url, username, text, scraped_at)
                VALUES (%(customer_id)s, %(post_url)s, %(username)s, %(text)s, %(scraped_at)s)
                """,
                comments
            )
            self.conn.commit()
            print(f"[INFO] Inserted {len(comments)} scraped comments")
            return True

    def get_scraped_comments(self, limit: int = 1000) -> pd.DataFrame:
        """Retrieve scraped comments for LLM processing."""
        if self.client:
            response = self.client.table("scraped_comments").select("*").limit(limit).execute()
            return pd.DataFrame(response.data)
        else:
            return pd.read_sql(
                f"SELECT * FROM scraped_comments LIMIT {limit}",
                self.conn
            )

    # ============================================
    # PROCESSED FEEDBACK (replaces processed_feedback.csv)
    # ============================================

    def insert_processed_feedback(self, feedback_items: List[Dict]) -> bool:
        """
        Insert LLM-analyzed feedback from feedback_analyse.py.

        Args:
            feedback_items: List of dicts with keys: customer_id, feedback_category, 
                           sentiment, complaint_intensity, raw_text
        """
        if not feedback_items:
            return False

        if self.client:
            data = [
                {
                    "customer_id": f["customer_id"],
                    "feedback_category": f["feedback_category"],
                    "sentiment": f["sentiment"],
                    "complaint_intensity": f["complaint_intensity"],
                    "raw_text": f["raw_text"]
                }
                for f in feedback_items
            ]
            response = self.client.table("processed_feedback").insert(data).execute()
            print(f"[INFO] Inserted {len(data)} processed feedback records")
            return True
        else:
            cur = self.conn.cursor()
            cur.executemany(
                """
                INSERT INTO processed_feedback 
                (customer_id, feedback_category, sentiment, complaint_intensity, raw_text)
                VALUES (%(customer_id)s, %(feedback_category)s, %(sentiment)s, 
                        %(complaint_intensity)s, %(raw_text)s)
                """,
                feedback_items
            )
            self.conn.commit()
            print(f"[INFO] Inserted {len(feedback_items)} processed feedback records")
            return True

    def get_processed_feedback(self) -> pd.DataFrame:
        """Retrieve all processed feedback for /reputation API."""
        if self.client:
            response = self.client.table("processed_feedback").select("*").execute()
            return pd.DataFrame(response.data)
        else:
            return pd.read_sql("SELECT * FROM processed_feedback", self.conn)

    # ============================================
    # SUBSCRIBERS (replaces train/test CSVs)
    # ============================================

    def insert_subscribers(self, df: pd.DataFrame) -> bool:
        """Bulk insert subscriber data from CDR/billing CSV."""
        records = df.to_dict("records")

        if self.client:
            # Supabase has 1000 row limit per insert, batch if needed
            BATCH_SIZE = 500
            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i:i+BATCH_SIZE]
                self.client.table("subscribers").insert(batch).execute()
            print(f"[INFO] Inserted {len(records)} subscribers")
            return True
        else:
            # Use COPY for fast bulk insert
            from io import StringIO
            buffer = StringIO()
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            cur = self.conn.cursor()
            cur.copy_from(buffer, "subscribers", sep=",", columns=df.columns.tolist())
            self.conn.commit()
            print(f"[INFO] Inserted {len(records)} subscribers")
            return True

    def get_subscribers(self, churn_only: bool = False) -> pd.DataFrame:
        """Retrieve subscribers for model training."""
        query = "SELECT * FROM subscribers"
        if churn_only:
            query += " WHERE churn = 1"

        if self.client:
            response = self.client.table("subscribers").select("*").execute()
            df = pd.DataFrame(response.data)
            if churn_only:
                df = df[df["churn"] == 1]
            return df
        else:
            return pd.read_sql(query, self.conn)

    # ============================================
    # PREDICTIONS (new — replaces in-memory results)
    # ============================================

    def insert_prediction(self, customer_id: str, probability: float, 
                          risk_level: str, action_required: bool) -> bool:
        """Log a model prediction."""
        data = {
            "customer_id": customer_id,
            "churn_probability": probability,
            "churn_risk_level": risk_level,
            "action_required": action_required
        }

        if self.client:
            self.client.table("predictions").insert(data).execute()
        else:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions (customer_id, churn_probability, churn_risk_level, action_required)
                VALUES (%s, %s, %s, %s)
                """,
                (customer_id, probability, risk_level, action_required)
            )
            self.conn.commit()
        return True

    def get_high_risk_customers(self) -> pd.DataFrame:
        """Get all customers with High risk for retention campaigns."""
        query = """
        SELECT s.*, p.churn_probability, p.predicted_at
        FROM subscribers s
        JOIN predictions p ON s.customer_id = p.customer_id
        WHERE p.churn_risk_level = 'High'
        ORDER BY p.churn_probability DESC
        """
        if self.client:
            # Use RPC for complex joins
            response = self.client.rpc("get_high_risk_customers").execute()
            return pd.DataFrame(response.data)
        else:
            return pd.read_sql(query, self.conn)

    # ============================================
    # REPUTATION (replaces processed_feedback.csv for /reputation API)
    # ============================================

    def get_reputation_summary(self) -> Dict:
        """
        Get brand reputation KPIs for /reputation endpoint.
        Uses the PostgreSQL function for efficiency.
        """
        if self.client:
            response = self.client.rpc("get_reputation_summary").execute()
            return response.data
        else:
            cur = self.conn.cursor()
            cur.execute("SELECT get_reputation_summary()")
            result = cur.fetchone()[0]
            return result

    def get_reputation_snapshot(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get historical reputation snapshot."""
        query = "SELECT * FROM reputation_snapshots"
        if date:
            query += f" WHERE snapshot_date = '{date}'"
        query += " ORDER BY snapshot_date DESC"

        if self.client:
            response = self.client.table("reputation_snapshots").select("*").execute()
            df = pd.DataFrame(response.data)
            if date:
                df = df[df["snapshot_date"] == date]
            return df
        else:
            return pd.read_sql(query, self.conn)

    # ============================================
    # UTILITY
    # ============================================

    def close(self):
        """Close database connection."""
        if not self.client and hasattr(self, "conn"):
            self.conn.close()
            print("[INFO] Database connection closed")


# ============================================
# MIGRATION: CSV → Supabase
# ============================================

def migrate_csv_to_supabase():
    """
    One-time migration script.
    Run this after setting up Supabase tables to move existing CSV data.
    """
    db = SupabaseDB()

    # 1. Migrate subscribers
    import glob
    for csv_file in glob.glob("data/*subscriber*.csv") + glob.glob("data/train*.csv") + glob.glob("data/test*.csv"):
        if os.path.exists(csv_file):
            print(f"Migrating {csv_file}...")
            df = pd.read_csv(csv_file)
            db.insert_subscribers(df)

    # 2. Migrate processed feedback
    feedback_file = "data/processed_feedback.csv"
    if os.path.exists(feedback_file):
        print(f"Migrating {feedback_file}...")
        df = pd.read_csv(feedback_file)
        records = df.to_dict("records")
        db.insert_processed_feedback(records)

    # 3. Migrate scraped comments (JSON)
    import glob
    for json_file in glob.glob("data/comments_*.json"):
        print(f"Migrating {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            comments = json.load(f)
        db.insert_scraped_comments(comments)

    db.close()
    print("✅ Migration complete!")


if __name__ == "__main__":
    # Test connection
    db = SupabaseDB()

    # Test: get reputation
    rep = db.get_reputation_summary()
    print("\nReputation Summary:")
    print(rep)

    db.close()