# Placeholder for ocr_service/extractors/commercial_registration.py

from ..config import GEMINI_MODEL
from google import genai
import time
from .base import BaseExtractor
import re
import json

class CommercialRegistrationExtractor(BaseExtractor):
    """
    Extractor for commercial registration documents.
    """
    def __init__(self):
        self.client = genai.Client(api_key="AIzaSyDeNz2IhcaMblFvMOR5eImvdqo-RhyoykU")  # CLIENT should be configured globally or injected

    def build_prompt(self, combined_text: str, fields: list[str]) -> str:
        lines = [
            "You are given the combined OCR text of a multi-page commercial-registration document.",
            "Ignore any footer-like text (page numbers, disclaimers).",
            "Dates in yyyy-mm-dd, numbers in English digits.",
            "Extract only these fields as JSON keys:"  
        ]
        for i, key in enumerate(fields, 1):
            lines.append(f"{i}. {key}")
        lines.append("===BEGIN TEXT===")
        lines.append(combined_text)
        lines.append("===END TEXT===")
        return "\n".join(lines)

    def extract(self, pages_text: list[str]) -> dict:
  
        # 1. Combine pages
        combined = "\n\n".join(pages_text)

        # 2. Define required fields
        fields = [
            "commercial register",
            "commercial name arabic",
            "Trade mark arabic",
            "Trade mark english",
            "business activity",
            "commercial establish date",
            "commencial end date",
            "term",
            "commercial expire date",
            "issued start date",
            "issued end date",
            "under law",
            "issue authorithy",
            "tax card",
            "tax file",
            "tax card expiray date",
            "unified register",
            "facility number",
            "paid capital"
        ]

        # 3. Build prompt
        prompt = self.build_prompt(combined, fields)

        # 4. Call Gemini
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        # 5. Pull out just the generated text
        raw = response.text

        # 6. Strip markdown fences if present
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

        # 7. Parse and return JSON
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from Gemini output:\n{stripped}") from e



