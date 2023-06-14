from typing import Union

from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()



@app.post("/sentiment/{text}")
def pl(text:str):
    classifier = pipeline('sentiment-analysis')
    return classifier(text)

@app.post("/question/{text}/context/{cont}/}")
def qa(text:str, cont:str):
    question_answer = pipeline("question-answering")
    return question_answer(
    question= text,
    context= cont)

@app.post("/translate/{text}")
def trans(text:str):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    return translator(text)

@app.get("/")
def read_root():
    return pl("/sentiment/{text}")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
