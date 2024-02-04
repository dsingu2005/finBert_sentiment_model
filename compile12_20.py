import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def get_quarter(month):
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4

def identify_quarters_in_directory(directory):
    plt.clf()
    files_with_dates = []
    sentiment_scores = []
    files_by_quarter = {}

    for filename in os.listdir(directory):
        match = re.search(r'(\d{4})-(\d{2})-\d{2}', filename)
        if match:
            year, month = map(int, match.groups())
            quarter = get_quarter(month)
            files_with_dates.append((filename, year, quarter))

    files_with_dates.sort(key=lambda x: (x[1], x[2]))

    for filename, year, quarter in sorted(files_with_dates, key=lambda x: x[0]):
        df = pd.read_excel(os.path.join(directory, filename))
        avg_sentiment_score = df['Sentiment Score'].mean()

        if f'Q{quarter} {year}' not in files_by_quarter:
            files_by_quarter[f'Q{quarter} {year}'] = []
        files_by_quarter[f'Q{quarter} {year}'].append(filename)

        if len(files_by_quarter[f'Q{quarter} {year}']) > 1:
            next_quarter = f'Q{quarter % 4 + 1} {year + (quarter // 4)}'
            if next_quarter not in files_by_quarter:
                files_by_quarter[next_quarter] = [files_by_quarter[f'Q{quarter} {year}'].pop(0)]
                sentiment_scores.append((next_quarter, avg_sentiment_score))
            else:
                sentiment_scores.append((f'Q{quarter} {year}', avg_sentiment_score))
        else:
            sentiment_scores.append((f'Q{quarter} {year}', avg_sentiment_score))

    quarters, scores = zip(*sentiment_scores)
    plt.plot(quarters, scores)
    plt.xlabel('Quarter')
    plt.ylabel('Average Sentiment Score')
    plt.title(f'Average Sentiment Score per Quarter for {os.path.basename(directory)}')
    plt.savefig(os.path.join(directory, f'{os.path.basename(directory)}_output.png'))
    plt.show()

    for quarter, filenames in files_by_quarter.items():
        if len(filenames) > 1:
            print(f'Warning: The company {os.path.basename(directory)} has multiple files for {quarter}: {", ".join(filenames)}')

folders = next(os.walk("29 Companies"))[1]

folders.sort()

for folder in folders[:29]:
    identify_quarters_in_directory(os.path.join("29 Companies", folder))