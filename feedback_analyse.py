import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()
os.makedirs("data", exist_ok=True)


def analyze_feedback_with_schema(feedback_file, output_file="data/processed_feedback.csv"):
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    if not os.path.exists(feedback_file):
        print(f"Error: Scraped file '{feedback_file}' not found.")
        return

    review_df = pd.read_csv(feedback_file).head(20)
    analysis_results = []
    
    print(f"--> Ingesting {len(review_df)} reviews from scraped file for Darija analysis...")
    
    for idx, (_, row) in enumerate(review_df.iterrows()):
        if 'comment_text' in row and pd.notna(row['comment_text']):
            text_payload = str(row['comment_text'])
        elif 'text' in row and pd.notna(row['text']):
            text_payload = str(row['text'])
        else:
            continue
            
        current_id = str(row['id'])
        
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert telecom classification engine specializing in Moroccan Darija (both Arabic script and Arabizi/Latin script).\n"
                        "Analyze the text provided by the user.\n"
                        "Return a JSON object containing a single root array named 'results'.\n"
                        "Evaluate the sentiment, category, and risk score carefully based on Moroccan dialect nuances:\n\n"
                        "Classification Rules:\n"
                        "1. If text complains about price, subscriptions, or high billing (e.g., 'غالي', 'بزاف', 'ثمن', 'فلوس', 'غالي بزاف'), use:\n"
                        "   feedback_category='pricing', sentiment='negative', complaint_intensity=3.\n"
                        "2. If text expresses neutral questions, general remarks, or satisfaction (e.g., 'واش غادي يلقاوهم', 'زوين', 'مزيان'), use:\n"
                        "   feedback_category='service_quality', sentiment='neutral', complaint_intensity=1.\n"
                        "3. If text mentions bad customer support, networks, or unhelpful service (e.g., 'سيرفيس عيان', 'ماجاوبونيش', 'دعم', 'ريزو عيان'), use:\n"
                        "   feedback_category='customer_support', sentiment='negative', complaint_intensity=5.\n"
                        "4. If text expresses explicit intent to switch or cancel service (e.g., 'غادي نلغي', 'شركاء خرين', 'نبدل', 'انوي', 'اتصالات'), use:\n"
                        "   feedback_category='churn_intent', sentiment='negative', complaint_intensity=5.\n\n"
                        "Output Format Blueprint:\n"
                        "{\n"
                        '  "results": [\n'
                        '    {"customer_id": "PLACEHOLDER", "feedback_category": "service_quality", "sentiment": "neutral", "complaint_intensity": 1}\n'
                        '  ]\n'
                        "}"
                    )
                },
                {
                    "role": "user", 
                    "content": f"Analyze this profile metadata item:\nID: {current_id}\nText: \"{text_payload}\""
                }
            ],
            temperature=0.0
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            for item in analysis.get('results', []):
                item['customer_id'] = current_id
                item['raw_text'] = text_payload
                analysis_results.append(item)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse LLM response for ID {current_id}. Skipping.")

    if analysis_results:
        result_df = pd.DataFrame(analysis_results)
        result_df.to_csv(output_file, index=False)
        print(f"\nAnalysis complete! Processed feedback saved to '{output_file}'.")
        print(tabulate(result_df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No valid analysis results were generated.")


if __name__ == "__main__":
    file_path = "data/scrape_facebook.csv" 
    analyze_feedback_with_schema(file_path)