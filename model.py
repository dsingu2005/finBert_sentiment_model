import torch
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

torch.save(model, 'finbert_model.pkl')