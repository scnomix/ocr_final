# ocr_service/extractors/iscore_individual.py
from typing import Dict, Union
import json
import re
import fitz  # PyMuPDF
from google import genai
from ..config import API_KEY, GEMINI_MODEL
from .base import BaseExtractor
from datetime import datetime

# Initialize shared Gemini client
gemini_client = genai.Client(api_key=API_KEY)

class ScorePersonalExtractor(BaseExtractor):
    """
    Extracts detailed personal credit score report from a multi-page PDF using a three-step workflow:
      1. Raw extraction: Gemini returns 'Key: Value' lines for required fields.
      2. JSON conversion: Gemini maps lines into structured JSON.
      3. JSON refinement: Gemini corrects Arabic text and identity_data structure.
    """

    def build_raw_prompt(self, full_report: str) -> str:
        return f"""
Extract these fields from the personal credit score report, one per line, in 'Key: Value':
Report Number:
Name (full Arabic name as it is):
Address(make sure to return the Arabic Address it might contain digits of course)
Credit Score:
Identity Data under 'بيانات تحقيق شخصية' as 'ID Type: Value' lines:
Credit Summary fields under 'ملخص محتوى التقرير للتسهيلات الائتمانية':
  Currency:
  Number of Facilities:
  Total Credit Limits:
  Total Outstanding:
  Total Monthly Installments:
For each facility table 'ﻲﻧﺎﻤﺘﺋا ﻞﻴﻬﺴﺘﻟا {{index}}', list 'facility_index, facility_code, facility_type, credit_limit, bank_code' make sure to return the facility_type:

Full Report Text:
{full_report}
===END===
Return only the 'Key: Value' lines, no commentary.
"""

    def build_json_prompt(self, raw_lines: str) -> str:
        return f"""
Convert these key:value lines into a JSON object with exactly these keys:
- report_number (string)
- profile (object with name,address, credit_score)
- identity_data (object with id_type: id_number)
- credit_summary (object with currency, number_of_facilities, total_credit_limits, total_outstanding, total_monthly_installments)
- facilities (array of objects, each with facility_index, facility_code, facility_type, credit_limit, bank_code)

Ensure:
- Identity data becomes a single object, not array.
- Arabic text is preserved correctly.
- bank_code contains only the alphanumeric code.
- Convert all Arabic numerals and dates to English digits; format dates as YYYY-MM-DD.

Raw Lines:
{raw_lines}
===END===
Return only the JSON object, no commentary.
"""

    def build_refine_prompt(self, extracted_json: str) -> str:
        return f"""
Here is the JSON extracted:
{extracted_json}

Please:
- Correct any garbled Arabic in profile.name and address.
- Ensure identity_data is a flat object: {{id_type: id_number}}.
- Confirm credit_summary has separate fields.
- Return only the corrected JSON object, no commentary.
"""

    def extract(self, pdf_path: str) -> Dict[str, Union[str, dict, list]]:
        # 1. Extract text
        doc = fitz.open(pdf_path)
        pages = [doc.load_page(i).get_text() for i in range(doc.page_count)]
        doc.close()
        full_report = "\n\n".join(pages)

        # 2. Raw extraction
        raw = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[self.build_raw_prompt(full_report)]
        ).text.strip()

        # 3. JSON conversion
        json_text = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[self.build_json_prompt(raw)]
        ).text.strip()
        json_text = re.sub(r"^```\w*|```$", "", json_text).strip()
        data = json.loads(json_text)

        # 4. JSON refinement
        refined_text = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[self.build_refine_prompt(json.dumps(data, ensure_ascii=False))]
        ).text.strip()
        refined_text = re.sub(r"^```\w*|```$", "", refined_text).strip()
        refined = json.loads(refined_text)

        return refined
