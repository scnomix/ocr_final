# ocr_service/pipeline.py

# from .classifier import classify_pdf, DocumentType
# from .config    import PDF_IMAGE_DPI, PAGES_TO_PROCESS, IMAGES_FOLDER
# from .utils.pdf_utils import pdf_to_images
# from .ocr       import ocr_images
# from .extractors.base import get_extractor_for

# def process_document(pdf_path: str) -> dict:
#     # 1. classify
#     doc_type = classify_pdf(pdf_path)

#     # 2. render pages → images
#     images = pdf_to_images(pdf_path, IMAGES_FOLDER, dpi=PDF_IMAGE_DPI, max_pages=PAGES_TO_PROCESS)

#     # 3. OCR
#     pages_text = ocr_images(images)

#     # 4. extract fields
#     extractor = get_extractor_for(doc_type)
#     return extractor.extract(pages_text)
# ocr_service/pipeline.py

from .classifier      import classify_pdf, DocumentType
from .config          import PDF_IMAGE_DPI, PAGES_TO_PROCESS, IMAGES_FOLDER
from .utils.pdf_utils import pdf_to_images
from .ocr             import ocr_images
from .extractors.base import get_extractor_for

def process_document(pdf_path: str) -> dict:
    # 1. classify
    doc_type = classify_pdf(pdf_path)
    print(doc_type)
    extractor = get_extractor_for(doc_type)

    # 2. For personal or company credit-score, pass PDF directly
    if doc_type in (DocumentType.ISCORE_INDIVIDUAL, DocumentType.ISCORE_COMPANY):
        return extractor.extract(pdf_path)

    # 3. Otherwise, do PDF→images→OCR
    images = pdf_to_images(pdf_path, IMAGES_FOLDER, dpi=PDF_IMAGE_DPI, max_pages=PAGES_TO_PROCESS)
    if doc_type !='COMMERCIAL_REGISTRATION':
        pages_text = ocr_images(images)
    else:
        pages_text = ocr_images(images,'COMMERCIAL_REGISTRATION')
        return pages_text
    # 4. extract fields from text
    return extractor.extract(pages_text)
