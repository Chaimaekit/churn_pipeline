"""
Supabase Integration for Telecom Churn Pipeline
Replaces CSV files with PostgreSQL via Supabase
"""
import os
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv
import json
import glob

load_dotenv()

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    print("[WARNING] supabase-py not installed. Using psycopg2 fallback.")
    print("Install: pip install supabase")


class SupabaseDB:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        self.db_uri = os.getenv("SUPABASE_CONNECTION_STRING") 
        
        self.client: Optional[Client] = None
        self._connect()

    def _connect(self):
        if HAS_SUPABASE and self.url and self.key:
            self.client = create_client(self.url, self.key)
            print("[INFO] Connected to Supabase via REST API")
        else:
            import psycopg2
            self.conn = psycopg2.connect(self.db_uri)
            print("[INFO] Connected to Supabase via psycopg2 URI")


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
    One-time migration script following Option A.
    Safely runs multiple times without creating duplicates by using upsert mechanics.
    """
    db = SupabaseDB()

    # 1. Exact columns allowed in the public.subscribers database table
    allowed_db_columns = [
        "customer_id", "state", "account_length", "area_code", "international_plan",
        "voice_mail_plan", "number_vmail_messages", "total_day_minutes", "total_day_calls",
        "total_day_charge", "total_eve_minutes", "total_eve_calls", "total_eve_charge",
        "total_night_minutes", "total_night_calls", "total_night_charge", "total_intl_minutes",
        "total_intl_calls", "total_intl_charge", "customer_service_calls", "churn",
        "feedback_text", "feedback_category", "sentiment", "complaint_intensity"
    ]

    # 2. Migrate only the test subscriber dataset safely
    import glob
    for csv_file in glob.glob("data/test*.csv"):
        if os.path.exists(csv_file):
            print(f"Processing subscriber records from: {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # Programmatic column filtering
            columns_to_keep = [col for col in allowed_db_columns if col in df.columns]
            df_cleaned = df[columns_to_keep].copy()
            
            if "churn" in df_cleaned.columns:
                df_cleaned["churn"] = df_cleaned["churn"].astype(int)
            
            # --- SAFE UPSERT CONVERSION ---
            # Instead of standard insert(), we perform an upsert on the unique key 'customer_id'
            records = df_cleaned.to_dict(orient="records")
            try:
                # This instructs Supabase to resolve clashes on customer_id safely
                db.client.table("subscribers").upsert(records, on_conflict="customer_id").execute()
                print(f"[INFO] Successfully synced/updated {len(records)} subscribers.")
            except Exception as e:
                print(f"[WARNING] Native upsert failed, attempting fallback batch processing: {e}")
                # Fallback to standard insert method if your setup doesn't support basic upsert strings
                db.insert_subscribers(df_cleaned)

    # 3. Migrate processed feedback comments
    feedback_file = "data/processed_comments.csv"
    if os.path.exists(feedback_file):
        print(f"Migrating processed sentiment records from: {feedback_file}...")
        df = pd.read_csv(feedback_file)
        records = df.to_dict("records")
        
        # Using upsert over insert to prevent duplication of identical parsed feedback texts
        try:
            db.client.table("processed_feedback").upsert(records).execute()
            print(f"[INFO] Successfully migrated feedback text matrix rows.")
        except Exception as e:
            print(f"[INFO] Attempting standard insertion for feedback: {e}")
            db.insert_processed_feedback(records)

    db.close()
    print("✅ Safe Dynamic Migration execution sequence completed!")


if __name__ == "__main__":
    # 1. Test connection
    db = SupabaseDB()
    
    # 2. Run the migration to populate the cloud tables
    print("\n[MIGRATION] Starting data migration to Supabase...")
    migrate_csv_to_supabase()
    
    # 3. Now test the reputation calculation after data is inserted
    print("\n[TEST] Fetching updated Reputation Summary...")
    try:
        rep = db.get_reputation_summary()
        print("Reputation Summary Result:")
        print(rep)
    except Exception as e:
        print(f"Error checking summary: {e}")

    db.close()