# FinBERT Sentiment Model for Earnings Calls

## Overview

This repository contains the working code and data files for 29 companies (as of 2/3/2024), along with a sentiment model that processes earnings calls and stores the results in Google Cloud's Datastore.

## Features

- **Sentiment Analysis:** The primary functionality of the FinBERT model is to conduct sentiment analysis on earnings call transcripts, identifying sentiments such as positive, negative, or neutral. We have integrated the FinBERT model into our project to analyze earnings calls.

- **Fine-tuned for Finance:** The model is fine-tuned using financial text data, enhancing its effectiveness in understanding nuances and context specific to the financial domain.

- **Easy Integration:** Our system allows for the integration of a front-end application, facilitating the analysis of data for specific fields.

## Acknowledgments

The FinBERT model is built upon the BERT architecture and utilizes the Hugging Face Transformers library. We express our gratitude to the open-source community and contributors to these projects.

## Further Documentation

Documentation regarding our use case, as well as the research behind it, can be found under the 'Documentation' directory within the main GitHub repository.
