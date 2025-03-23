from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define fixed categories
CATEGORIES = [
    "Technology", "Education", "Gaming", "Lifestyle", "Comedy",
    "Music", "Finance", "News", "Health", "Travel"
]

# Define request body structure
class ChannelDescription(BaseModel):
    description: str

# API route for classification
@app.post("/classify/")
async def classify_channel(data: ChannelDescription):
    result = classifier(data.description, candidate_labels=CATEGORIES)
    predicted_category = result["labels"][0]  # The top predicted category
    return {"description": data.description, "predicted_category": predicted_category}