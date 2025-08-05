# ocr_service/ocr.py

# import time
# from google import genai
# from .config import API_KEY, GEMINI_MODEL

# client = genai.Client(api_key=API_KEY)

# def ocr_image_with_gemini(image_path: str) -> str:
#     """
#     Uploads `image_path` to Gemini and polls until text is extracted.
#     """
#     gemini_file = client.files.upload(file=image_path)
#     while not getattr(gemini_file, "state", None) or gemini_file.state.name != "ACTIVE":
#         time.sleep(0.5)
#         gemini_file = client.files.get(name=gemini_file.name)

#     prompt = (
#         "Extract **all visible text** from this commercial-registration page. "
#         "Return only the extracted text, no commentary."
#     )
#     response = client.models.generate_content(
#         model=GEMINI_MODEL,
#         contents=[gemini_file, prompt]
#     )
#     return response.text

# def ocr_images(image_paths: list[str]) -> list[str]:
#     texts = []
#     for path in image_paths:
#         texts.append(ocr_image_with_gemini(path))
#     return texts

import time
import json
import re
from google import genai
from .config import API_KEY, GEMINI_MODEL
from .classifier import DocumentType
from .config import PDF_IMAGE_DPI, PAGES_TO_PROCESS, IMAGES_FOLDER
from .utils.pdf_utils import pdf_to_images

client = genai.Client(api_key=API_KEY)


def ocr_image_with_gemini(image_path: str) -> str:
    """
    Uploads `image_path` to Gemini and polls until text is extracted.
    """
    gemini_file = client.files.upload(file=image_path)
    while not getattr(gemini_file, "state", None) or gemini_file.state.name != "ACTIVE":
        time.sleep(0.5)
        gemini_file = client.files.get(name=gemini_file.name)

    prompt = (
        "Extract **all visible text** from this commercial-registration page. "
        "Return only the extracted text, no commentary."
    )
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[gemini_file, prompt]
    )
    return response.text


def ocr_images(image_paths: list[str], doc_type: DocumentType = None) -> list[str] | dict:
    """
    If doc_type is COMMERCIAL_REGISTRATION, do the two-step extraction:
      1) OCR both pages
      2) Per-page prompts to pull out exactly the fields you care about
      3) Aggregate into JSON and return that dict
    Otherwise, just OCR every image and return list of raw texts.
    """
    texts = [ocr_image_with_gemini(p) for p in image_paths]

    if doc_type == 'COMMERCIAL_REGISTRATION':
        # page‐1: extract fields 1–15
        page1_kv = extract_page1_fields(texts[0] if len(texts) > 0 else "")
        # page‐2: extract paid capital
        page2_kv = extract_page2_fields(texts[1] if len(texts) > 1 else "")
        # aggregate to JSON
        return aggregate_fields_to_json(page1_kv, page2_kv)

    # fallback: return raw OCR texts
    return texts


def extract_page1_fields(text: str) -> str:
    prompt = (
        "You are given the OCR text of the first page of a commercial-registration document.\n"
        "Extract the following fields as 'key: value' lines exactly:\n"
        "1. commercial register: text after 'مستخرج سجل تجاري رقم' in the header.\n"
        "2. commercial name arabic: from the second column before 'Trade mark english'.\n"
        "3. Trade mark arabic: same as commercial name arabic.\n"
        "4. Trade mark english: immediately after the Arabic name.\n"
        "5. business activity: under point (ب) in the fourth column.\n"
        "6. commercial establish date: under point (ب) in the first column.\n"
        "7. commencial end date: labeled 'ساري الى' in the first column.\n"
        "8. term: number next to 'المدة' in column five.\n"
        "9. commercial expire date: the later date in column five.\n"
        "10. issued start date: text after 'تحرر في' at the top.\n"
        "11. issued end date: issued start date plus 3 years.\n"
        "12. under law: text after 'قانون رقم' in the second column.\n"
        "13. issue authorithy: top-right header, e.g. 'مكتب استثمار الجيزة'.\n"
        "14. tax card: Arabic number after 'الرقم القومي للمنشأة'.\n"
        "15. unified register: Arabic number after 'الرقم الموحد للسجل التجاري'.\n"
        "Return only key-value lines, no extra commentary.\n"
        "===BEGIN TEXT===\n"
        f"{text}\n"
        "===END TEXT===\n"
    )
    resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt])
    return resp.text.strip()


def extract_page2_fields(text: str) -> str:
    prompt = (
        "You are given the OCR text of the second page of a commercial-registration document.\n"
        "Extract only 'paid capital: value' by finding text after 'مقدار راس المال'.\n"
        "Return only that key-value line, no extra commentary.\n"
        "===BEGIN TEXT===\n"
        f"{text}\n"
        "===END TEXT===\n"
    )
    resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt])
    return resp.text.strip()


def aggregate_fields_to_json(kv1: str, kv2: str) -> dict:
    combined = kv1 + "\n" + kv2
    agg_prompt = (
        "You are given key-value lines from two pages of a commercial-registration document:\n"
        f"{combined}\n"
        "Convert these into a JSON object with these keys in this exact order:\n"
        "[\"commercial register\",\"commercial name arabic\",\"Trade mark arabic\","
        "\"Trade mark english\",\"business activity\",\"commercial establish date\","
        "\"commencial end date\",\"term\",\"commercial expire date\",\"issued start date\","
        "\"issued end date\",\"under law\",\"issue authorithy\",\"tax card\","
        "\"unified register\",\"paid capital\"].\n"
        "If any key is missing, set its value to an empty string. Return only valid JSON."
    )
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=[agg_prompt])
    raw = resp.text
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    return json.loads(clean)

