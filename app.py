from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import training
from custom_classifier import SequenceClassifier


app = FastAPI() 

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# label_mapping = {"Churn": 0, "Escalation": 1,'Churn and Escalation':2, "No Intent Found": 3, }



classifier = torch.load('./data/custom.pth').eval()


class QueryRequest(BaseModel):
    query: str

# Define a function to perform text classification
def classify_text(text):
    inputs = tokenizer(text, 
        padding=True,
        truncation=True,
        return_tensors="pt",)

    #BERT embeddings
    with torch.no_grad():
        bert_outputs = bert_model(**inputs)
        bert_embeddings = bert_outputs.last_hidden_state

    # Passing BERT embeddings through the custom classifier
    outputs = classifier(bert_embeddings)

    
    _, predicted_class = outputs.max(1)
    print(outputs)

    intent_labels = {0: "Churn", 1: "Escalation", 2: "Churn and Escalation" , 3:'No Intent Found'}
    predicted_label = intent_labels[predicted_class.item()]

    return predicted_label 

@app.get("/")
async def get_index():
    return FileResponse("form.html")

# Route to do intent classifier post call
@app.post("/intent/")
async def classify(query_data: QueryRequest):
    user_query = query_data.query
    predicted_intent = classify_text(user_query) 

    return {"intent": predicted_intent}

@app.post("/api/")
async def classify_api(query: str = Form(...)):

    predicted_intent = classify_text(query)
    return {"intent": predicted_intent}