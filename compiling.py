# import pandas as pd
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt

# def compile_results(folder_name):
#     output_files = [f for f in os.listdir(folder_name) if f.startswith('output_FINAL TRANSCRIPT') and f.endswith('.xlsx')]

#     compiled_results_sentiment = pd.DataFrame()
#     compiled_results_magnitude = pd.DataFrame()

#     for file in output_files:
#         counts_df = pd.read_excel(os.path.join(folder_name, file))

#         date = file.replace('output_FINAL TRANSCRIPT', '').replace('.xlsx', '')

#         average_scores_sentiment = counts_df.groupby('Key Word Category')['Sentiment Score'].mean().reset_index()
#         average_scores_sentiment.columns = ['Key Word Category', f'Sentiment Score {date}']
#         average_scores_magnitude = counts_df.groupby('Key Word Category')['Magnitude'].mean().reset_index()
#         average_scores_magnitude.columns = ['Key Word Category', f'Magnitude {date}']

#         if compiled_results_sentiment.empty:
#             compiled_results_sentiment = average_scores_sentiment
#         else:
#             compiled_results_sentiment = pd.merge(compiled_results_sentiment, average_scores_sentiment, on='Key Word Category', how='outer')

#         if compiled_results_magnitude.empty:
#             compiled_results_magnitude = average_scores_magnitude
#         else:
#             compiled_results_magnitude = pd.merge(compiled_results_magnitude, average_scores_magnitude, on='Key Word Category', how='outer')

#     compiled_results_sentiment.to_excel(f'{folder_name}/Compiled_results_sentiment.xlsx', index=False)
#     compiled_results_magnitude.to_excel(f'{folder_name}/Compiled_results_magnitude.xlsx', index=False)

#     visualize_results(compiled_results_sentiment, 'Average Sentiment Scores by Keyword Category', folder_name)
#     visualize_results(compiled_results_magnitude, 'Average Magnitude Scores by Keyword Category', folder_name)

# def visualize_results(df, title, folder_name):
#     df_melt = df.melt('Key Word Category', var_name='Date', value_name='Score')
#     plt.figure(figsize=(15, 10))
#     sns.lineplot(x='Key Word Category', y='Score', hue='Date', data=df_melt)
#     plt.title(title)
#     plt.xticks(rotation=90)
#     plt.savefig(f'{folder_name}/{title}.png')
#     plt.close()

# #Keep changing the foldername here
# compile_results('ads')
# --
from google.cloud import storage

def create_folder(bucket_name, folder_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/")
    blob.upload_from_string('')

create_folder('sentiment-files', 'test-folder')


