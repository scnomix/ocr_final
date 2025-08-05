# Placeholder for ocr_service/classifier.py
from enum import Enum, auto
import time
from google import genai
from .config import API_KEY, GEMINI_MODEL
from .utils.pdf_utils import pdf_to_images

# Initialize the Gemini client
tacy_client = genai.Client(api_key=API_KEY)

class DocumentType(Enum):
    NATIONAL_ID = auto()
    COMMERCIAL_REGISTRATION = auto()
    TAX_CARD = auto()
    FINANCIAL_SUMMARY = auto()
    ISCORE_COMPANY = auto()
    ISCORE_INDIVIDUAL = auto()


def classify_pdf(pdf_path: str) -> DocumentType:
    """
    Classify a PDF by sending its first page image to Gemini.
    Returns one of the DocumentType enum values.
    """
    # 1. Convert only the first page to an image
    images = pdf_to_images(pdf_path, output_folder=".", dpi=150, max_pages=1)
    if not images:
        raise FileNotFoundError(f"No pages converted from {pdf_path}")
    image_path = images[0]

    # 2. Upload to Gemini and wait until ACTIVE
    gem_file = tacy_client.files.upload(file=image_path)
    while not getattr(gem_file, "state", None) or gem_file.state.name != "ACTIVE":
        time.sleep(0.5)
        gem_file = tacy_client.files.get(name=gem_file.name)

    # 3. Prompt Gemini for classification
    prompt = (
        "Classify the type of this document. "
        "Choose exactly one of: NATIONAL_ID, COMMERCIAL_REGISTRATION, TAX_CARD, FINANCIAL_SUMMARY, ISCORE_COMPANY, ISCORE_INDIVIDUAL. "
        "Return only the label (no extra text)."
    )
    response = tacy_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[gem_file, prompt]
    )
    label = response.text.strip().upper()

    # 4. Map the label to our enum
    try:
        return DocumentType[label]
    except KeyError:
        raise ValueError(f"Unrecognized document type from Gemini: '{label}'")