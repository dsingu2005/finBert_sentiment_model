import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import re
import matplotlib.pyplot as plt

def read_file(file_path):
    return pd.read_excel(file_path)

def calculate_weighted_sentiment(file_path):
    sentiment_df = read_file(file_path)
    keywords_df = read_file("12_20/keywords.xlsx")

    # Merge sentiment data with keywords and categories
    sentiment_df = pd.merge(sentiment_df, keywords_df, left_on=['Key Word Category', 'Keyword'], right_on=['Key Word Category', 'Key Words/Topics'])

    # Calculate the weighted sentiment score
    sentiment_df['Weighted Sentiment Score'] = sentiment_df['Sentiment Score'] * sentiment_df['Keyword Percentage']

    # Save the DataFrame with the new column to the same Excel file
    sentiment_df.to_excel(file_path, index=False)

def identify_quarters_in_directory(directory):
    files_by_company = {}

    for filename in os.listdir(directory):
        match = re.search(r'CC_(.+)_Q(\d{1})(\d{4})', filename)
        if match:
            company = match.group(1)
            quarter_year = 'Q' + match.group(2) + match.group(3)
            if company not in files_by_company:
                files_by_company[company] = []
            files_by_company[company].append((filename, quarter_year))

    for company, file_quarter_pairs in files_by_company.items():
        print(f'Processing files for company: {company}')

        plt.clf()
        sentiment_scores_by_category = {}

        # Sort file_quarter_pairs based on quarter and year
        file_quarter_pairs.sort(key=lambda x: (int(x[1][2:]), int(x[1][1:2])))

        for filename, quarter_year in file_quarter_pairs:
            df = pd.read_excel(os.path.join(directory, filename))
            calculate_weighted_sentiment(os.path.join(directory, filename))

            # Group by Key Word Category and sum the Weighted Sentiment Score
            category_scores = df.groupby('Key Word Category')['Weighted Sentiment Score'].sum()
            
            # Update the sentiment_scores_by_category dictionary
            for category, score in category_scores.items():
                if category not in sentiment_scores_by_category:
                    sentiment_scores_by_category[category] = []
                sentiment_scores_by_category[category].append((quarter_year, score))

        # Plot sentiment scores for each category
        for category, scores in sentiment_scores_by_category.items():
            quarters, category_scores = zip(*scores)
            plt.plot(quarters, category_scores, label=category)

        plt.xlabel('Quarter')
        plt.ylabel('Total Weighted Sentiment Score')
        plt.title(f'Total Weighted Sentiment Score per Quarter for {company}')
        plt.legend()
        plt.savefig(os.path.join(directory, f'{company}_WeightedSentiment_Categories.png'))
        plt.show()

def plot_individual_sentiment_scores(directory):
    files_by_company = {}

    for filename in os.listdir(directory):
        match = re.search(r'CC_(.+)_Q(\d{1})(\d{4})', filename)
        if match:
            company = match.group(1)
            quarter_year = 'Q' + match.group(2) + match.group(3)
            if company not in files_by_company:
                files_by_company[company] = []
            files_by_company[company].append((filename, quarter_year))

    for company, file_quarter_pairs in files_by_company.items():
        print(f'Processing files for company: {company}')

        plt.clf()
        sentiment_scores_by_category = {}

        # Sort file_quarter_pairs based on quarter and year
        file_quarter_pairs.sort(key=lambda x: (int(x[1][2:]), int(x[1][1:2])))

        for filename, quarter_year in file_quarter_pairs:
            df = pd.read_excel(os.path.join(directory, filename))

            # Group by Key Word Category and average the Sentiment Score
            category_scores = df.groupby('Key Word Category')['Sentiment Score'].mean()

            # Update the sentiment_scores_by_category dictionary
            for category, score in category_scores.items():
                if category in ["Financial metric - All", "Macro", "Sector trend"]:
                    if category not in sentiment_scores_by_category:
                        sentiment_scores_by_category[category] = []
                    sentiment_scores_by_category[category].append((quarter_year, score))

        # Plot sentiment scores for each category
        for category, scores in sentiment_scores_by_category.items():
            quarters, category_scores = zip(*scores)
            plt.plot(quarters, category_scores, label=category)

        plt.xlabel('Quarter')
        plt.ylabel('Average Sentiment Score')
        plt.title(f'Average Sentiment Score per Quarter for {company}')
        plt.legend()
        plt.savefig(os.path.join(directory, f'{company}_Sentiment_Categories.png'))
        plt.show()

# Define the directory path
directory_path = "29 Companies"

# Plot individual sentiment scores for the keyword categories
plot_individual_sentiment_scores(directory_path)

# Define the directory path
# directory_path = "29 Companies"

# # Identify quarters in the directory and generate graphs
# identify_quarters_in_directory(directory_path)
