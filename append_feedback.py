import pandas as pd


def append_feedback(train_path="data/train.csv", feedback_path="data/customer_feedback.csv", output_path="data/train_with_feedback.csv"):
    df_train = pd.read_csv(train_path)
    df_feedback = pd.read_csv(feedback_path)

    if 'feedback_text' not in df_feedback and 'feedback' in df_feedback:
        df_feedback = df_feedback.rename(columns={'feedback': 'feedback_text'})

    merge_columns = ['customer_id', 'feedback_text', 'feedback_category', 'sentiment', 'complaint_intensity']
    feedback_columns = [col for col in merge_columns if col in df_feedback.columns]

    merged_df = df_train.merge(
        df_feedback[feedback_columns],
        on='customer_id',
        how='left'
    )

    merged_df.to_csv(output_path, index=False)
    return merged_df


if __name__ == '__main__':
    result = append_feedback()
    print(f"Merged feedback into train data: {len(result)} rows written to 'data/train_with_feedback.csv'.")
        

