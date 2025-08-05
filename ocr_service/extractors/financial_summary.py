# Placeholder for ocr_service/extractors/financial_summary.py
from typing import List, Union
import json
import re
from google import genai
from datetime import datetime
from ..config import API_KEY, GEMINI_MODEL
from .base import BaseExtractor

# Shared Gemini client
gemini_client = genai.Client(api_key=API_KEY)

class FinancialSummaryExtractor(BaseExtractor):
    """
    Extracts key fields from a multi-page financial summary report (CBE).

    Workflow:
      1. Raw extraction: list fields as 'Key: Value' lines.
      2. JSON conversion: map lines into clean JSON.
      3. Regex fallback for Finance List if needed.
      4. Date normalization.

    Fields:
      - Client Name
      - CBE Code
      - CBE Tenor
      - Print Date
      - Governorate Name
      - Industry
      - Finance List (array of ints) which is the banks where the client deals with.
    """

    def build_raw_prompt(self, text: str) -> str:
        return f"""
Extract the following fields from the Central Bank of Egypt financial summary report, one per line in the format 'Key: Value':

Client Name:
CBE Code:
CBE Tenor will be found like in the header the مركز مجمع اعميل نهاية شهر note this should be the end of the month (eg.so if نهاية شهر 8/2022 return "2022-08-31"):
Print Date:
Governorate Code (next to the governorate Name) (just before it):
Governorate Name:
Industry Code (next to the industry(just before it)):
Industry :
Finance List which is the banks where the client deals with it will be found in table you should extract all the numbers in it the table header is بنوك التعامل:

Use English digits for numbers and dates. Finance List should list all numeric amounts (in thousands) separated by commas.

===BEGIN REPORT TEXT===
{text}
===END REPORT TEXT===
Return only the lines of 'Key: Value'.
"""  

    def build_json_prompt(self, raw_lines: str) -> str:
        return f"""
Convert these key-value lines into a JSON object with exactly these keys:
Client Name, CBE Code, CBE Tenor, Print Date, Governorate Code,Governorate Name, Industry Code,Industry, Finance List.

- Use English digits and 'YYYY-MM-DD' for dates.
- Finance List should become an array of integers.
- Omit any unknown fields.

===RAW KEY-VALUE LINES===
{raw_lines}
===END RAW LINES===
Return only the JSON object, no commentary.
"""

    def extract(self, pages_text: List[str]) -> Union[dict, List[dict]]:
        # Combine pages
        combined = "\n\n".join(pages_text)

        # Step 1: raw extraction
        raw_prompt = self.build_raw_prompt(combined)
        raw_resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[raw_prompt]
        )
        raw_lines = raw_resp.text.strip()

        # Step 2: JSON conversion
        json_prompt = self.build_json_prompt(raw_lines)
        json_resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[json_prompt]
        )
        json_text = json_resp.text.strip()
        # Remove code fences
        json_text = re.sub(r"^```\w*|```$", "", json_text).strip()

        # Attempt to parse JSON
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse Financial Summary JSON. Raw: {json_text}")

        # Regex fallback: ensure Finance List is array of ints
        if 'Finance List' in data and not isinstance(data['Finance List'], list):
            nums = re.findall(r"\d+", raw_lines.partition('Finance List:')[-1])
            data['Finance List'] = [int(n) for n in nums]

        # Normalize dates
        for key in ('CBE Tenor', 'Print Date'):
            val = data.get(key, '')
            if val:
                # try ISO first
                try:
                    dt = datetime.fromisoformat(val)
                except ValueError:
                    # try slash format
                    try:
                        dt = datetime.strptime(val, '%Y/%m/%d')
                    except ValueError:
                        continue
                data[key] = dt.strftime('%Y-%m-%d')

        return data
