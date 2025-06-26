
from fastapi import FastAPI, UploadFile, Form
from transformers import ViltProcessor, ViltForQuestionAnswering
from huggingface_hub import login
from PIL import Image
import io
import torch
import os

# Login securely using environment variable
login(token=os.getenv("HF_TOKEN"))

app = FastAPI()

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.get("/")
def root():
    return {"message": "Welcome to the VQA API. Use /predict to ask questions about an image."}

@app.post("/predict")
async def predict(file: UploadFile, question: str = Form(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    idx = outputs.logits.argmax(-1).item()
    answer = model.config.id2label[idx]

    return {"question": question, "answer": answer}
