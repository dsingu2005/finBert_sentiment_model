import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import gcsfs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize
from google.auth import compute_engine
from google.auth import default
from google.cloud import datastore
from google.cloud import storage
import shutil
import pickle

# nltk.download('vader_lexicon')
with open('finbert_model.pkl', 'rb') as f:
    model, tokenizer = pickle.load(f)
# model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finBERT')
# tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finBERT')

# model = AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
# tokenizer = AutoTokenizer.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')


sia = SentimentIntensityAnalyzer()

def split_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = softmax(outputs.logits, dim=1)
    return probabilities[0]

#https://console.cloud.google.com/storage/browser/sentiment-files/sentiment_magnitude?hl=en&project=sentiment-analysis-379200&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
# def process_file(filename):
#     credentials, _ = default()
#     fs = gcsfs.GCSFileSystem(project='sentiment-analysis-379200', token=credentials)
#     # fs = gcsfs.GCSFileSystem(project='sentiment-analysis-379200')
#     with fs.open('gs://sentiment-files/sentiment_magnitude/keywords.xlsx') as f:
#         keywords_df = pd.read_excel(f)
#     with fs.open(f'gs://sentiment-files/sentiment_magnitude/{filename}.xlsx') as f:
#         ads_df = pd.read_excel(f)

#     # Create a directory in the GCS bucket
#     fs.mkdirs(f'gs://sentiment-files/scores_magnitude{filename}')

#     for column in ads_df.columns:
#         counts_df = pd.DataFrame(columns=['Key Word Category', 'Keyword', 'Paragraph', 'Sentiment Score', 'Magnitude'])

#         for index, row in keywords_df.iterrows():
#             keyword = row['Key Words/Topics']
#             category = row['Key Word Category']

#             paragraphs = ads_df[column].apply(lambda x: str(x) if keyword.lower() in str(x).lower() else None).dropna()

#             for paragraph in paragraphs:
#                 chunks = split_text(paragraph, 1024)
#                 for chunk in chunks:
#                     probabilities = analyze_sentiment(chunk)
#                     sentiment_score = (probabilities[1] + (probabilities[2] * 2) + (probabilities[0] * 3)) - 2
                    
#                     sentences = sent_tokenize(chunk)
#                     magnitudes = []
#                     for sentence in sentences:
#                         sentence_probabilities = analyze_sentiment(sentence)
#                         sentence_sentiment_score = (sentence_probabilities[1] + (sentence_probabilities[2] * 2) + (sentence_probabilities[0] * 3)) - 2
#                         sentence_magnitude = abs(sia.polarity_scores(sentence)['compound'])
#                         magnitudes.append(sentence_magnitude)
                    
#                     total_magnitude = sum(magnitudes)
                    
#                     new_row = {'Key Word Category': category, 'Keyword': keyword, 'Paragraph': chunk, 'Sentiment Score': sentiment_score.item(), 'Magnitude': total_magnitude}
#                     counts_df = pd.concat([counts_df, pd.DataFrame([new_row])], ignore_index=True)

#         counts_df.to_excel(f'gs://sentiment-files/scores_magnitude/{filename}/output_{column}.xlsx', index=False)


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



# def compile_results(folder_name):
#     fs = gcsfs.GCSFileSystem(project='sentiment-analysis-379200')
#     output_files = fs.ls(f'gs://sentiment-files/scores_magnitude/{folder_name}')

#     compiled_results_sentiment = pd.DataFrame()
#     compiled_results_magnitude = pd.DataFrame()

#     for file in output_files:
#         with fs.open(file) as f:
#             counts_df = pd.read_excel(f)

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

#     compiled_results_sentiment.to_excel(f'gs://sentiment-files/scores_magnitude/{folder_name}/Compiled_results_sentiment.xlsx', index=False)
#     compiled_results_magnitude.to_excel(f'gs://sentiment-files/scores_magnitude/{folder_name}/Compiled_results_magnitude.xlsx', index=False)

#     visualize_results(compiled_results_sentiment, 'Average Sentiment Scores by Keyword Category', folder_name)
#     visualize_results(compiled_results_magnitude, 'Average Magnitude Scores by Keyword Category', folder_name)
    # credentials, _ = default()
    # fs = gcsfs.GCSFileSystem(project='sentiment-analysis-379200', token=credentials)

    # with fs.open('gs://sentiment-files/sentiment_magnitude/keywords.xlsx') as f:
        # keywords_df = pd.read_excel(f)
    # with fs.open(f'gs://sentiment-files/sentiment_magnitude/{filename}.xlsx') as f:
        # ads_df = pd.read_excel(f)
    
def process_file(filename):

    keywords_df = pd.read_excel('sentiment/12_20/keywords.xlsx')
    xls = pd.ExcelFile(f'sentiment/12_20/{filename}.xlsx')

    for sheet_name in xls.sheet_names:
        ads_df = pd.read_excel(xls, sheet_name=sheet_name)

        # Create a directory locally
        if not os.path.exists(f'./scores_magnitude/{sheet_name}'):
            os.makedirs(f'./scores_magnitude/{sheet_name}')

        for column in ads_df.columns:
            counts_df = pd.DataFrame(columns=['Key Word Category', 'Keyword', 'Paragraph', 'Sentiment Score', 'Sentiment Magnitude'])

            for index, row in keywords_df.iterrows():
                keyword = row['Key Words/Topics']
                category = row['Key Word Category']

                paragraphs = ads_df[column].apply(lambda x: str(x) if keyword.lower() in str(x).lower() else None).dropna()

                for paragraph in paragraphs:
                    chunks = split_text(paragraph, 1024)
                    for chunk in chunks:
                        probabilities = analyze_sentiment(chunk)
                        sentiment_score = (probabilities[1] + (probabilities[2] * 2) + (probabilities[0] * 3)) - 2
                        
                        sentences = sent_tokenize(chunk)
                        magnitudes = []
                        for sentence in sentences:
                            sentence_probabilities = analyze_sentiment(sentence)
                            sentence_sentiment_score = (sentence_probabilities[1] + (sentence_probabilities[2] * 2) + (sentence_probabilities[0] * 3)) - 2
                            sentence_magnitude = abs(sia.polarity_scores(sentence)['compound'])
                            magnitudes.append(sentence_magnitude)
                        
                        total_magnitude = sum(magnitudes)
                        
                        new_row = {'Key Word Category': category, 'Keyword': keyword, 'Paragraph': chunk, 'Sentiment Score': sentiment_score.item(), 'Sentiment Magnitude': total_magnitude}
                        counts_df = pd.concat([counts_df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the output file with the date of the earnings call
            counts_df.to_excel(f'./scores_magnitude/{sheet_name}/output_{column}.xlsx', index=False)

def compile_results():
    base_dir = './scores_magnitude'
    folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    compiled_results_sentiment = pd.DataFrame()
    compiled_results_magnitude = pd.DataFrame()

    for folder in folders:
        output_files = [f for f in os.listdir(folder) if f.startswith('output_') and f.endswith('.xlsx')]

        for output_file in output_files:
            df = pd.read_excel(os.path.join(folder, output_file))

            average_scores_sentiment = df.groupby('Key Word Category')['Sentiment Score'].mean().reset_index()
            average_scores_magnitude = df.groupby('Key Word Category')['Sentiment Magnitude'].mean().reset_index()

            if compiled_results_sentiment.empty:
                compiled_results_sentiment = average_scores_sentiment
            else:
                compiled_results_sentiment = pd.concat([compiled_results_sentiment, average_scores_sentiment], ignore_index=True)

            if compiled_results_magnitude.empty:
                compiled_results_magnitude = average_scores_magnitude
            else:
                compiled_results_magnitude = pd.concat([compiled_results_magnitude, average_scores_magnitude], ignore_index=True)

    compiled_results_sentiment = compiled_results_sentiment.groupby('Key Word Category')['Sentiment Score'].mean().reset_index()
    compiled_results_magnitude = compiled_results_magnitude.groupby('Key Word Category')['Sentiment Magnitude'].mean().reset_index()

    compiled_results_sentiment.to_excel(os.path.join(base_dir, 'Compiled_results_sentiment.xlsx'), index=False)
    compiled_results_magnitude.to_excel(os.path.join(base_dir, 'Compiled_results_magnitude.xlsx'), index=False)


# def visualize_results(df, title, folder_name):
#     df_melt = df.melt('Key Word Category', var_name='Date', value_name='Score')
#     plt.figure(figsize=(15, 10))
#     sns.lineplot(x='Key Word Category', y='Score', hue='Date', data=df_melt)
#     plt.title(title)
#     plt.xticks(rotation=90)
#     plt.savefig(f'{title}.png')
#     plt.close()

#     # Upload the plot to GCS
#     fs = gcsfs.GCSFileSystem(project='sentiment-analysis-379200')
#     fs.put(f'{title}.png', f'gs://sentiment-files/scores_magnitude/{folder_name}/{title}.png')

def visualize_results(df, title, folder_name):
    df_melt = df.melt('Key Word Category', var_name='Date', value_name='Score')
    plt.figure(figsize=(15, 10))
    sns.lineplot(x='Key Word Category', y='Score', hue='Date', data=df_melt)
    plt.title(title)
    plt.xticks(rotation=90)
    
    # Save the plot locally
    if not os.path.exists(f'./scores_magnitude/{folder_name}'):
        os.makedirs(f'./scores_magnitude/{folder_name}')
    plt.savefig(f'./scores_magnitude/{folder_name}/{title}.png')
    
    plt.close()

def process_folder(folder_name):
    storage_client = storage.Client()
    bucket_name = "sentiment-files"
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=f'scores_magnitude/{folder_name}/')
    urls = []
    for blob in blobs:
        if blob.name.endswith('.png'):  # Check if the blob name ends with .png
            print(f'Processing file: {blob.name}')
            url = blob.public_url  # Get the public URL of the blob
            urls.append(url)
    return urls

# process_file('ADS GR')
# compile_results()