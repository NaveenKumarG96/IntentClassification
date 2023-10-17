import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from custom_classifier import SequenceClassifier

# Load the BERT model and tokenizer 
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


label_mapping = {"Churn": 0, "Escalation": 1,'Churn and Escalation':2, "No Intent Found": 3, }


custom_classifier = torch.load('../data/custom.pth').eval()

def preprocess_data(data, tokenizer):
    input_texts = data
    

    tokenized_data = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    return input_ids, attention_mask


def classify_text(text):
    # Tokenize the text
    input_ids, attention_mask = preprocess_data(text,tokenizer)

    # Get BERT embeddings
    with torch.no_grad():
        bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state

    # Pass BERT embeddings through the custom classifier
    custom_outputs = custom_classifier(bert_embeddings)

    
    _, predicted_class = custom_outputs.max(1)
    print(custom_outputs)
    
    intent_labels = {0: "Churn", 1: "Escalation", 2: "Churn and Escalation'" , 3:'No Intent Found'}
    predicted_label = intent_labels[predicted_class.item()]

    return predicted_label 


text_input = "I'm  leaving"
predicted_intent = classify_text(text_input)
print(f"Predicted Intent: {predicted_intent}")

