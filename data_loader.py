import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_real_telecom_data(file_path: str) -> pd.DataFrame:
    """
    Loads a local subscriber dataset (.csv, .xlsx, or .xls) from disk.
    """
    if not os.path.exists(file_path):
        print(f"\n[CRITICAL ERROR] File not found at target path: '{file_path}'")
        print("Please check the file name and try again.")
        sys.exit(1)

    print(f"Reading data from: {file_path}")

    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        print("\n[CRITICAL ERROR] Unsupported file format! Please use a .csv or .xlsx file.",
              file=sys.stderr)
        sys.exit(1)



def load_from_postgres(query: str) -> pd.DataFrame:
    """
    Runs a SQL query against the configured PostgreSQL database.
    Connection is created on demand so importing this module never crashes.
    """
    try:
        import psycopg2
    except ImportError:
        print("[ERROR] psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
    )
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


_ANGER_KEYWORDS = [
    "غالي", "غالية", "مشا", "حيدوا", "رجعو", "رجع", "واش كاين", "امتا",
    "كونيكسيون", "انترنت", "مخدامش", "مشكل", "ضعيف", "بطي",
    "تفوووو", "تفو", "عيقتو", "الله يعطيكم الإفلاس", "حرام", "سرقة",
]

_PROMO_KEYWORDS = ["نجمة", "درهم", "جيغا", "Go", "عرض", "روشارج", "شارجي"]


def _count_keywords(text: str, keywords: list) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for kw in keywords if kw in text)


def load_facebook_enrichment(json_path: str) -> pd.DataFrame:
    """
    Reads the Orange Maroc scraped Facebook JSON and returns a single-row
    summary DataFrame with telecom-context features ready to merge into
    the churn dataset.

    Expected JSON structure (n8n pinned output):
        [ { "posts": [ { "id", "text", "reactionCount", "reactionCounts",
                         "commentCount", "publishTime", "topComments": [...] } ] } ]

    Returns a DataFrame with one row and these columns:
        fb_total_posts, fb_avg_reactions, fb_anger_ratio,
        fb_avg_comment_count, fb_pricing_post_ratio,
        fb_complaint_comment_ratio, fb_sentiment_score
    """
    if not os.path.exists(json_path):
        print(f"[WARNING] Facebook enrichment file not found at '{json_path}'. "
              "Skipping enrichment.")
        return pd.DataFrame()

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list) and "posts" in raw[0]:
        posts = raw[0]["posts"]
    elif isinstance(raw, list):
        posts = raw
    else:
        posts = raw.get("posts", [])

    records = []
    for p in posts:
        reaction_counts = p.get("reactionCounts") or p.get("reaction_counts") or {}
        total_reactions = p.get("reactionCount", 0) or 0
        anger           = reaction_counts.get("anger", 0) or 0

        top_comments = p.get("topComments", [])
        comment_text = " ".join(
            c.get("text", "") for c in top_comments if isinstance(c, dict)
        )

        records.append({
            "post_id":           p.get("id", ""),
            "total_reactions":   total_reactions,
            "anger_reactions":   anger,
            "comment_count":     p.get("commentCount", 0) or 0,
            "post_text":         p.get("text", "") or "",
            "comment_text":      comment_text,
        })

    df = pd.DataFrame(records)

    if df.empty:
        print("[WARNING] No posts found in Facebook JSON.")
        return pd.DataFrame()

    df["anger_ratio"] = (
        df["anger_reactions"] / df["total_reactions"].replace(0, 1)
    ).round(4)

    df["is_pricing_post"] = df["post_text"].apply(
        lambda t: int(_count_keywords(t, _PROMO_KEYWORDS) > 0)
    )

    df["has_complaint_comment"] = df["comment_text"].apply(
        lambda t: int(_count_keywords(t, _ANGER_KEYWORDS) > 0)
    )

    total_posts   = len(df)
    anger_ratio   = df["anger_ratio"].mean().round(4)
    total_pos = sum(
        (p.get("reactionCounts") or p.get("reaction_counts") or {}).get("like", 0) +
        (p.get("reactionCounts") or p.get("reaction_counts") or {}).get("love", 0)
        for p in posts
    )
    total_neg = sum(
        (p.get("reactionCounts") or p.get("reaction_counts") or {}).get("anger", 0)
        for p in posts
    )
    total_all = total_pos + total_neg or 1
    sentiment_score = round((total_pos - total_neg) / total_all, 4)

    summary = pd.DataFrame([{
        "fb_total_posts":            total_posts,
        "fb_avg_reactions":          round(df["total_reactions"].mean(), 2),
        "fb_anger_ratio":            anger_ratio,
        "fb_avg_comment_count":      round(df["comment_count"].mean(), 2),
        "fb_pricing_post_ratio":     round(df["is_pricing_post"].mean(), 4),
        "fb_complaint_comment_ratio":round(df["has_complaint_comment"].mean(), 4),
        "fb_sentiment_score":        sentiment_score,
    }])

    print(f"[INFO] Facebook enrichment loaded — {total_posts} posts, "
          f"anger_ratio={anger_ratio}, sentiment={sentiment_score}")

    return summary


def merge_facebook_enrichment(
    subscriber_df: pd.DataFrame,
    json_path: str = "data/orange_fb_posts.json",
) -> pd.DataFrame:
    """
    Convenience wrapper: loads FB enrichment and broadcasts it as new columns
    onto every row of the subscriber DataFrame.

    Usage in main.py / processing.py:
        df = merge_facebook_enrichment(df)
    """
    enrichment = load_facebook_enrichment(json_path)
    if enrichment.empty:
        return subscriber_df

    for col in enrichment.columns:
        subscriber_df[col] = enrichment[col].iloc[0]

    print(f"[INFO] Added {len(enrichment.columns)} Facebook context columns to dataset.")
    return subscriber_df