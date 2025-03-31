import streamlit as st
from PIL import Image
import io
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face API details
API_URL_GENDER = "https://api-inference.huggingface.co/models/rizvandwiki/gender-classification"
API_URL_DETECTOR = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def query_gender(image):
    image = image.convert('RGB')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    response = requests.post(API_URL_GENDER, headers=headers, data=image_bytes)
    return response

def query_detector(image_bytes):
    response = requests.post(API_URL_DETECTOR, headers=headers, data=image_bytes)
    return response.json()

# The rest of the functions remain unchanged...

