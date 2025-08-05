# ocr_service/extractors/national_id.py
from typing import List, Tuple, Union
import json
import re
from google import genai
from datetime import datetime, timedelta
from ..config import API_KEY, GEMINI_MODEL
from .base import BaseExtractor

# Initialize a shared Gemini client
gemini_client = genai.Client(api_key=API_KEY)

class NationalIDExtractor(BaseExtractor):
    """
    Extracts fields from Egyptian national ID PDFs. Works through:
      1. Classify pages as FRONT, BACK, or BOTH (if front and back on one page).
      2. Pair pages into records, handling any order and multiple IDs.
      3. Combine front/back text, then use Gemini to extract JSON directly.
      4. Post-process gender, dates, profession and fill missing expiry date.
    """

    def classify_page(self, page_text: str) -> str:
        """
        Ask Gemini to label a page's OCR text.
        """
        prompt = f"""
You are given the OCR text from one page of an Egyptian national ID card:
{page_text}
Classify this page as exactly one of: FRONT, BACK, or BOTH.
Return only that label, with no extra text.
"""
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt]
        )
        return response.text.strip().upper()

    def build_record_text(self, front: str, back: str) -> str:
        """
        Label and combine front/back text for JSON prompt.
        """
        return f"""
===BEGIN FRONT===
{front}
===END FRONT===
===BEGIN BACK===
{back}
===END BACK===
"""

    def build_json_prompt(self, record_text: str) -> str:
        """
        Ask Gemini to extract all fields into a clean JSON object.
        """
        return f"""
You are given combined OCR text from the FRONT and BACK of an Egyptian national ID card:
{record_text}

Extract the following fields and return a JSON object with exactly these keys:
  full_name (line 1), note the first name will be found in one line doesn't contain any thing else so you should concate it with the parent name.
  gender ('Male' or 'Female'),
  date_of_birth (YYYY-MM-DD), national_id_number (14 digits),
  issue_date (YYYY-MM-DD), expiration_date (YYYY-MM-DD),
  address, profession.

- Map Arabic 'ذكر' to 'Male' and 'انثى' to 'Female'.
- Convert all Arabic numerals to English digits.
- If 'expiration_date' is missing, you will add 7 years to 'issue_date' after parsing.
- Ignore any machine-readable zone (MRZ) or scanner footer text.

Return only the JSON object, with no code fences or extra commentary.
"""

    def extract(self, pages_text: List[str]) -> Union[dict, List[dict]]:
        # 1. Classify pages
        classified: List[Tuple[int, str, str]] = []
        for idx, text in enumerate(pages_text):
            label = self.classify_page(text)
            classified.append((idx, label, text))

        # 2. Pair pages into records
        records: List[Tuple[str, str]] = []
        used = set()
        # Handle any BOTH pages first
        for idx, label, text in classified:
            if label == 'BOTH':
                records.append((text, text))
                used.add(idx)
        # Collect fronts and backs
        fronts = [(i, t) for i, l, t in classified if l == 'FRONT' and i not in used]
        backs  = [(i, t) for i, l, t in classified if l == 'BACK' and i not in used]
        # Match fronts to closest backs
        for f_idx, f_text in fronts:
            if backs:
                b_idx, b_text = min(backs, key=lambda x: abs(x[0] - f_idx))
                records.append((f_text, b_text))
                used.update({f_idx, b_idx})
                backs = [b for b in backs if b[0] != b_idx]
        # If no records matched, fallback to single combined record
        if not records:
            combined = "\n\n".join(text for _, _, text in classified)
            records = [(combined, combined)]

        # 3. Extract each record
        outputs = []
        for front_text, back_text in records:
            record_text = self.build_record_text(front_text, back_text)
            prompt = self.build_json_prompt(record_text)
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt]
            )
            raw = resp.text.strip()
            # Remove code fences if any
            raw = re.sub(r"^```\w*|```$", "", raw).strip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError(f"JSON parse failed. Raw response: {raw}")

            # 4. Post-process expiration_date
            if not data.get('expiration_date') and data.get('issue_date'):
                dt = datetime.strptime(data['issue_date'], '%Y-%m-%d') + timedelta(days=7*365)
                data['expiration_date'] = dt.strftime('%Y-%m-%d')

            outputs.append(data)

        return outputs[0] if len(outputs) == 1 else outputs
