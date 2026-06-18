# feedback_analysis.py
import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

def analyze_feedback_with_schema(feedback_file, output_file="processed_feedback.csv", batch_size=20):
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Ingest and explicitly slice to test the first 20 rows
    review_df = pd.read_csv(feedback_file).head(20)
    analysis_results = []
    
    print(f"--> Ingesting {len(review_df)} reviews for schema validation test...")
    
    for _, row in review_df.iterrows():
        text_col = 'feedback_text' if 'feedback_text' in row else 'feedback'
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert telecom classification engine. Analyze the array of texts provided by the user.\n"
                        "Return a JSON object containing a single root array named 'results'.\n"
                        "Evaluate each item step-by-step and map them strictly to these rules:\n"
                        "1. If text mentions 'cost', 'expensive', or 'monthly cost', use: feedback_category='pricing', sentiment='negative', complaint_intensity=3.\n"
                        "2. If text mentions 'works well' or 'no major complaints', use: feedback_category='service_quality', sentiment='positive', complaint_intensity=1.\n"
                        "3. If text mentions 'customer support' or 'issue was not solved', use: feedback_category='customer_support', sentiment='negative', complaint_intensity=5.\n"
                        "4. If text mentions 'switching' or 'not satisfied', use: feedback_category='churn_intent', sentiment='negative', complaint_intensity=5.\n\n"
                        "Output Format Blueprint:\n"
                        "{\n"
                        '  "results": [\n'
                        '    {"customer_id": "CUST_000001", "feedback_category": "pricing", "sentiment": "negative", "complaint_intensity": 3}\n'
                        '  ]\n'
                        "}"
                    )
                },
                {"role": "user", "content": f"Analyze this payload: {text_col}: \"{row[text_col]}\""}
            ],
            temperature=0.0
        )
        try:
            customer_id = row['customer_id'] if 'customer_id' in row else f"CUST_{_+1:06d}"
            analysis = json.loads(response.choices[0].message.content)
            for item in analysis.get('results', []):
                item['customer_id'] = customer_id  # Ensure we have the customer_id in the output
                analysis_results.append(item)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse LLM response for customer_id {row['customer_id']}. Skipping this entry.")
        
    # Convert the results into a DataFrame and save to CSV
    if analysis_results:
        result_df = pd.DataFrame(analysis_results)
        result_df.to_csv(output_file, index=False)
        print(f"Analysis complete! Processed feedback saved to '{output_file}'.")
        print(tabulate(result_df.head(), headers='keys', tablefmt='psql'))
    else:
        print("No valid analysis results were generated.")

if __name__ == "__main__":
    file_path = "customer_feedback.csv"
    analyze_feedback_with_schema(file_path)