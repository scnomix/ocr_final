# Placeholder for ocr_service/extractors/tax_card.py
# ocr_service/extractors/tax_card.py
from typing import List, Union
import json
import re
from google import genai
from datetime import datetime
from ..config import API_KEY, GEMINI_MODEL
from .base import BaseExtractor

# Shared Gemini client
gemini_client = genai.Client(api_key=API_KEY)

class TaxCardExtractor(BaseExtractor):
    """
    Extractor for Tax Card PDFs. Treats the entire document as one record.

    Fields to extract:
      - Country
      - Ministry
      - Authority
      - Tax Center
      - Company Name
      - Address
      - Activity
      - Tax ID Number
      - Card Issuance Date
      - Card Expiry Date
      - Card Number
      - Document Type
      - Usage Restriction
      - Lost/Found Instructions
      - Contact for Lost/Stolen Cards
    """
    def extract(self, pages_text: List[str]) -> Union[dict, List[dict]]:
        # Combine all pages text
        combined_text = "\n\n".join(pages_text)

        # Build prompt for Gemini
        prompt = f"""
You are given OCR text from a Tax Card document. Extract the following fields and return a JSON object with exactly these keys (no extra keys or commentary):
  Country
  Ministry
  Authority
  Tax Center
  Company Name
  Address
  Activity
  Tax ID Number
  Card Issuance Date (format YYYY-MM-DD)
  Card Expiry Date (YYYY-MM-DD)
  Card Number
  Document Type
  Usage Restriction
  Lost/Found Instructions
  Contact for Lost/Stolen Cards

Use English digits for all numbers and dates. If a field is missing, set its value to an empty string.

===BEGIN TAX CARD TEXT===
{combined_text}
===END TAX CARD TEXT===
Return only the JSON object.
"""        
        # Call Gemini
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt]
        )
        raw = response.text.strip()
        # Remove backticks or code fences
        raw = re.sub(r"^```\w*|```$", "", raw).strip()
        
        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse Tax Card JSON. Raw: {raw}")

        # Normalize date formats
        for date_key in ("Card Issuance Date", "Card Expiry Date"):
            if data.get(date_key):
                try:
                    # Attempt to parse common formats
                    dt = datetime.fromisoformat(data[date_key])
                    data[date_key] = dt.strftime('%Y-%m-%d')
                except ValueError:
                    pass

        return data
