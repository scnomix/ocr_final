# ocr_service/config.py
import streamlit as st
# ——— Gemini / Google GenAI settings ———
# API_KEY      = st.secrets["GEMINI_API_KEY"]
API_KEY="AIzaSyDeNz2IhcaMblFvMOR5eImvdqo-RhyoykU"
GEMINI_MODEL = "gemini-2.5-flash"

# ——— General defaults ———
PDF_IMAGE_DPI    = 300
PAGES_TO_PROCESS = 2
IMAGES_FOLDER    = "temp_images"
