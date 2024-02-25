#import statements
import os
import pandas as pd
from PIL import Image
import io
from google.cloud import datastore

#parsing

# data = pd.read_excel("test1.xlsx")


client = datastore.Client("sentiment-analysis-379200")

#1. parse the name of the file --> create the entity based on the name and location (ex: TRI CN)
folder = os.listdir('xml')
for file in folder:
    splice1 = file.split()
    company_location = splice1[0].split('_')[1] + '_' +  splice1[1].split('_')[0]
    splice2 = splice1[1].split('_')
    quarter_year = splice2[1]

    #create entity here based on company_location var value
    key = client.key(company_location)
    task = datastore.Entity(key)

    #2. parse the contens of the xml file. below is a template of the queries
    '''
    quarter (with year) || keyword cat || keyword || para || sentiment score || sentiment magnitude || Key words/Topic || 
    keyword percentage || weighted sentiment
    '''
    #a. split into rows
    file_location = os.path.join('xml', file)
    file_content = pd.read_excel(file_location)

    for index,row in file_content.iterrows():

        #non-compulsory data
        try:
            if row.iloc[5]:
                keywords_and_topic = row.iloc[5]
        except:
            keywords_and_topic = '-'

        try:
            if row.iloc[6]:
                keyword_percentage = row.iloc[6]
        except:
            keyword_percentage = '-'
        
        try:
            if row.iloc[7]:
                weighted_sentiment = row.iloc[7]
        except:
            weighted_sentiment = '-'
        


        #b. for each row, split by columns and enter the data
        data = {"index: ": index,
                "Quarter: ": quarter_year,        
                "Key Word Category: ": row.iloc[0],
                "Keyword: ": row.iloc[1],
                "Paragraph: ": row.iloc[2],
                "Sentiment Score: ": row.iloc[3],
                "Sentiment Magnitude: ": row.iloc[4],
                "Key Words/Topic: ": keywords_and_topic,
                "Keyword Percentage: ": keyword_percentage,
                "Weighted Sentiment: ": weighted_sentiment }
        task.update(data)
        client.put(task)

#3. put the data
