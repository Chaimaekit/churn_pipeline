import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tabulate import tabulate
from supabase import create_client

load_dotenv()
os.makedirs("data", exist_ok=True)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)


def analyze_feedback_with_schema(feedback_file, output_file="data/processed_comments.csv"):
    """
    Reads scraped Facebook comments (JSON) and converts to structured CSV
    for the /reputation API consumption.

    Expected input JSON format:
    [
      {
        "customer_id": "e819425d08fb",
        "post_url": "https://facebook.com/reel/...",
        "username": "ام افنان",
        "text": "واش كاين لعرض 6*20",
        "scraped_at": "2026-06-20T13:07:23.251884"
      }
    ]

    Output CSV columns (matching /reputation API expectations):
    - customer_id, feedback_category, sentiment, complaint_intensity, raw_text
    """
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if not os.path.exists(feedback_file):
        print(f"Error: Scraped file '{feedback_file}' not found.")
        return

    with open(feedback_file, "r", encoding="utf-8") as f:
        comments = json.load(f)

    comments_df = pd.DataFrame(comments)
    print(f"--> Loaded {len(comments_df)} comments from {feedback_file}")

    analysis_results = []

    print(f"--> Sending {len(comments_df)} comments to Groq LLM for Darija analysis...")

    for idx, row in comments_df.iterrows():
        text_payload = str(row.get("text", "")).strip()
        if not text_payload:
            continue

        current_id = str(row.get("customer_id", f"unknown_{idx}"))

        try:
            response = groq.chat.completions.create(
                model="openai/gpt-oss-120b",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert telecom classification engine specializing in Moroccan Darija "
                            "(both Arabic script and Arabizi/Latin script).\n"
                            "Analyze the text and return ONLY a JSON object with a single root array named 'results'.\n"
                            "Each result must have: customer_id, feedback_category, sentiment, complaint_intensity.\n\n"
                            "Classification Rules:\n"
                            "1. pricing: price complaints, billing, high costs (e.g., 'غالي', 'بزاف', 'ثمن', 'فلوس')\n"
                            "2. service_quality: neutral questions, general remarks, satisfaction (e.g., 'واش غادي يلقاوهم', 'زوين')\n"
                            "3. customer_support: bad support, unhelpful service (e.g., 'سيرفيس عيان', 'ماجاوبونيش')\n"
                            "4. churn_intent: explicit switching/canceling (e.g., 'غادي نلغي', 'نبدل', 'انوي', 'اتصالات')\n\n"
                            "sentiment must be exactly one of: positive, neutral, negative\n"
                            "complaint_intensity must be an integer 1-5\n\n"
                            "Output format:\n"
                            '{\n'
                            '  "results": [\n'
                            '    {"customer_id": "ID", "feedback_category": "pricing", "sentiment": "negative", "complaint_intensity": 3}\n'
                            '  ]\n'
                            '}'
                        )
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze this comment:\nID: {current_id}\nText: \"{text_payload}\""
                    }
                ],
                temperature=0.0,
                max_tokens=500
            )

            analysis = json.loads(response.choices[0].message.content)
            for item in analysis.get("results", []):
                item["customer_id"] = current_id
                item["raw_text"] = text_payload
                # Ensure all required columns exist
                item.setdefault("feedback_category", "service_quality")
                item.setdefault("sentiment", "neutral")
                item.setdefault("complaint_intensity", 1)
                analysis_results.append(item)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response for ID {current_id}: {e}")
            # Fallback: add with defaults
            analysis_results.append({
                "customer_id": current_id,
                "feedback_category": "service_quality",
                "sentiment": "neutral",
                "complaint_intensity": 1,
                "raw_text": text_payload
            })
        except Exception as e:
            print(f"Warning: Error processing ID {current_id}: {e}")
            continue

        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(comments_df)} comments...")

    if analysis_results:
        result_df = pd.DataFrame(analysis_results)

        # Ensure column order matches /reputation API expectations
        columns = ["customer_id", "feedback_category", "sentiment", "complaint_intensity", "raw_text"]
        result_df = result_df[columns]

        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nAnalysis complete! Saved {len(result_df)} records to '{output_file}'")
        print(f"\nPreview:")
        print(tabulate(result_df.head(10), headers="keys", tablefmt="psql", showindex=False))

        # Summary stats
        print(f"\nSummary:")
        print(f"  Total comments: {len(result_df)}")
        print(f"  Sentiment distribution:")
        for sentiment, count in result_df["sentiment"].value_counts().items():
            print(f"    {sentiment}: {count}")
        print(f"  Category distribution:")
        for cat, count in result_df["feedback_category"].value_counts().items():
            print(f"    {cat}: {count}")
        print(f"  Avg complaint intensity: {result_df['complaint_intensity'].mean():.2f}")
    else:
        print("No valid analysis results were generated.")


if __name__ == "__main__":
    import glob

    json_files = glob.glob("data/comments_*.json")
    if not json_files:
        print("No comments_*.json files found in data/ directory.")
        print("Please run scrape_facebook.py first.")
        exit(1)

    latest_file = max(json_files, key=os.path.getmtime)
    print(f"Using latest scraped file: {latest_file}")

    analyze_feedback_with_schema(latest_file)