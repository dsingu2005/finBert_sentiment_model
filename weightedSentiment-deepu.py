import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def read_file(file_path):
    return pd.read_excel(file_path)

def calculate_weighted_sentiment(file_path):
    # Read the files
    sentiment_df = read_file(file_path)
    keywords_df = read_file("12_20/keywords.xlsx")

    # Calculate the count of each keyword within its category
    keyword_counts = keywords_df.groupby('Key Word Category')['Key Words/Topics'].value_counts()

    # Calculate the percentage of each keyword within its category
    keyword_percentages = keyword_counts.groupby(level=0).apply(lambda x: x / x.sum())

    # Normalize the keyword percentages
    scaler = MinMaxScaler()
    keyword_percentages = scaler.fit_transform(keyword_percentages.values.reshape(-1, 1))

    # Convert the keyword percentages to a DataFrame
    keyword_percentages_df = pd.DataFrame(keyword_percentages, columns=['Keyword Percentage'], index=keyword_counts.index)

    # Reset the index of the keyword percentages DataFrame
    keyword_percentages_df.reset_index(inplace=True)

    # Merge the sentiment DataFrame with the keyword percentages DataFrame
    sentiment_df = pd.merge(sentiment_df, keyword_percentages_df, left_on=['Key Word Category', 'Keyword'], right_on=['Key Word Category', 'Key Words/Topics'])

    # Calculate the weighted sentiment score
    sentiment_df['Weighted Sentiment Score'] = sentiment_df['Sentiment Score'] * sentiment_df['Keyword Percentage']

    # Save the DataFrame with the new column to the same Excel file
    sentiment_df.to_excel(file_path, index=False)

def identify_quarters_in_directory(directory):
    files_by_company = {}

    # Group files by company
    for filename in os.listdir(directory):
        match = re.search(r'CC_(.+)_Q(\d{1})(\d{4})', filename)
        if match:
            company = match.group(1)
            quarter_year = 'Q' + match.group(2) + match.group(3)
            if company not in files_by_company:
                files_by_company[company] = []
            files_by_company[company].append((filename, quarter_year))

    for company, file_quarter_pairs in files_by_company.items():
        print(f'Processing files for company: {company}')  # Print the company name

        plt.clf()
        sentiment_scores = []

        # Sort file_quarter_pairs based on quarter and year
        file_quarter_pairs.sort(key=lambda x: (int(x[1][2:]), int(x[1][1:2])))
        for filename, quarter_year in file_quarter_pairs:
            df = pd.read_excel(os.path.join(directory, filename))
            total_sentiment_score = df['Weighted Sentiment Score'].sum()  # Calculate the sum instead of the average
            sentiment_scores.append((quarter_year, total_sentiment_score))

        quarters, scores = zip(*sentiment_scores)
        plt.plot(quarters, scores)
        plt.xlabel('Quarter')
        plt.ylabel('Average Weighted Sentiment Score')
        plt.title(f'Average Weighted Sentiment Score per Quarter for {company}')
        plt.savefig(os.path.join(directory, f'{company}_WeightedSentiment.png'))
        plt.show()

# # Define the directory path
# directory_path = "29 Companies"

# # Get a list of all files in the directory
# all_files = os.listdir(directory_path)

# # Loop through all files
# for file in all_files:
#     # Check if the file is an .xlsx file
#     if file.endswith(".xlsx"):
#         # Construct the full file path
#         file_path = os.path.join(directory_path, file)
#         # Call the function with the file path
#         calculate_weighted_sentiment(file_path)

identify_quarters_in_directory("29 Companies")