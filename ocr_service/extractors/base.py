# Placeholder for ocr_service/extractors/base.py
from abc import ABC, abstractmethod
from ..classifier import DocumentType

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, pages_text: list[str]) -> dict:
        """
        Given OCR text per page, return a dict of structured fields.
        """
        pass


def get_extractor_for(doc_type: DocumentType) -> BaseExtractor:
    """
    Factory: return the appropriate extractor for a document type.
    """
    from .national_id import NationalIDExtractor
    from .commercial_registration import CommercialRegistrationExtractor
    from .tax_card import TaxCardExtractor
    from .financial_summary import FinancialSummaryExtractor
    from .iscore_company import ScoreCompanyExtractor
    from .iscore_individual import ScorePersonalExtractor

    mapping: dict[DocumentType, type[BaseExtractor]] = {
        DocumentType.NATIONAL_ID: NationalIDExtractor,
        DocumentType.COMMERCIAL_REGISTRATION: CommercialRegistrationExtractor,
        DocumentType.TAX_CARD: TaxCardExtractor,
        DocumentType.FINANCIAL_SUMMARY: FinancialSummaryExtractor,
        DocumentType.ISCORE_COMPANY: ScoreCompanyExtractor,
        DocumentType.ISCORE_INDIVIDUAL: ScorePersonalExtractor,
    }
    extractor_cls = mapping.get(doc_type)
    if not extractor_cls:
        raise ValueError(f"No extractor defined for {doc_type}")
    return extractor_cls()