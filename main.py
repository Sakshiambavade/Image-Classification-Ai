import streamlit as st
from PIL import Image
import io
import pandas as pd
import requests
import os

# Load API key from environment variables or Streamlit secrets
API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_default_api_key_here")

# Hugging Face API details
API_URL_GENDER = "https://api-inference.huggingface.co/models/rizvandwiki/gender-classification"
API_URL_DETECTOR = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Function to query gender classification API
@st.cache_data
def query_gender(image):
    try:
        image = image.convert('RGB')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        response = requests.post(API_URL_GENDER, headers=headers, data=image_bytes)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

# Function to query gender classification API
@st.cache_data
def query_gender(image_bytes):
    try:
        response = requests.post(API_URL_GENDER, headers=headers, data=image_bytes)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

def gender_classification():
    st.title("Gender Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Convert image to bytes
        image_bytes = io.BytesIO()
        image.convert('RGB').save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # Call the cached function with bytes instead of an image object
        with st.spinner('Classifying...'):
            result = query_gender(image_bytes.getvalue())

        if result:
            df = pd.DataFrame(result)
            st.write("API Response:")
            st.table(df)

            top_result = df.loc[df['score'].idxmax()]
            label = top_result['label']
            st.write(f"The person in the image is likely to be **{label}** with a score of {top_result['score']:.2f}.")
        else:
            st.write("Failed to process the image.")

def ai_image_detector():
    st.title("AI Image Detector")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner('Analyzing...'):
            result = query_detector(image_bytes)

        if result:
            df = pd.DataFrame(result)
            st.write("API Response:")
            st.table(df)
            
            if not df.empty:
                top_result = df.loc[df['score'].idxmax()]
                label = top_result['label']
                st.write(f"The image is likely **{label}** with a score of {top_result['score']:.2f}.")
            else:
                st.write("No results to display.")
        else:
            st.write("Failed to get a valid response from the API.")

def is_artificial_detector():
    st.title("Is Image Artificial?")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner('Analyzing...'):
            result = query_detector(image_bytes)

        if result:
            is_artificial = any(item['label'] == 'artificial' and item['score'] > 0.20 for item in result)
            st.write("The image may be artificially generated." if is_artificial else "The image is likely human.")
        else:
            st.write("Failed to get a valid response from the API.")

def main():
    st.set_page_config(page_title="AI Image Tools", page_icon=":robot:")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Gender Classification", "AI Image Detector", "Is Image Artificial?"])

    if selection == "Gender Classification":
        gender_classification()
    elif selection == "AI Image Detector":
        ai_image_detector()
    elif selection == "Is Image Artificial?":
        is_artificial_detector()

if __name__ == "__main__":
    main()
