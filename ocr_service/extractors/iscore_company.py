# Placeholder for ocr_service/extractors/iscore_company.py
# ocr_service/extractors/iscore_company.py
from typing import Dict, Union, List
import json
import re
import fitz  # PyMuPDF
from google import genai
from ..config import API_KEY, GEMINI_MODEL
from .base import BaseExtractor
from datetime import datetime

# Initialize shared Gemini client
gemini_client = genai.Client(api_key=API_KEY)

class ScoreCompanyExtractor(BaseExtractor):
    """
    Extracts detailed corporate credit score report from a multi-page PDF using a three-step Gemini workflow:
      1. Raw extraction: Gemini returns 'Key: Value' lines for each section.
      2. JSON conversion: Gemini maps those lines into structured JSON.
      3. JSON refinement: Gemini corrects Arabic text and table structures.
    """

    def build_raw_prompt(self, full_report: str) -> str:
        return f"""
Extract these fields from the corporate credit score report, one per line in the format 'Key: Value':

Report Number:
Company Name:
Address:
Credit Score:
Corporate Profile table under 'ﺔﻴﺼﺨﺸﻟا ﻖﻴﻘﺤﺗ تﺎﻧﺎﻴﺑ':
Business Risk Summary under 'ى ﻤﻟا يﺰﻛﺮﻤﻟا ﻚﻨﺒﻟا راﺮﻘﻟ ﺎﻘﺒﻃ ةﺄﺸﻨﻤﻟا تﺎﻧﺎﻴﺑ':
Identity Data (under 'بيانات تحقيق شخصية'):
Credit Summary (under 'ملخص محتوى التقرير للتسهيلات الائتمانية'):
Facility tables 'ﻲﻧﺎﻤﺘﺋا ﻞﻴﻬﺴﺘﻟا {{index}}': facility_index, facility_code, facility_type, credit_limit, bank_code

Full Report Text:
{full_report}
===END===
Return only the 'Key: Value' lines, no commentary.
"""  

    def build_json_prompt(self, raw_lines: str) -> str:
        return f"""
Convert these key:value lines into a JSON object with exactly these keys:
- report_number (string)
- company_profile (object parsed from Corporate Profile table)
- business_risk_summary (object parsed from Business Risk Summary table)
- profile (object with company name, address, credit_score)
- identity_data (object key:id)
- credit_summary (object with currency,number_of_facilities,total_credit_limits,total_outstanding,total_monthly_installments)
- facilities (array of objects with facility_index,facility_code,facility_type,credit_limit,bank_code)

Ensure bank_code contains only alphanumeric code, Arabic text preserved, numerals to English, dates YYYY-MM-DD.
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
- Fix any garbled Arabic in company_profile and profile fields.
- Ensure 'identity_data' is a flat object {{id_type: id_number}}.
- Structure 'company_profile' and 'business_risk_summary' as nested objects with correct keys.
- Return only the corrected JSON object, no commentary.
"""

    def extract(self, pdf_path: str) -> Dict[str, Union[str, dict, list]]:
        # 1. Extract full text
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
        refined = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[self.build_refine_prompt(json.dumps(data, ensure_ascii=False))]
        ).text.strip()
        refined = re.sub(r"^```\w*|```$", "", refined).strip()
        return json.loads(refined)
