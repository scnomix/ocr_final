# Placeholder for ocr_service/utils/pdf_utils.py
# utils/pdf_utils.py

import fitz
import os

def pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 150, max_pages: int = 2) -> list[str]:
    """
    Converts up to `max_pages` of the PDF into JPEGs at `dpi`.
    Returns list of image paths.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot find PDF file: {pdf_path}")

    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_to_do = min(max_pages, doc.page_count)
    image_paths = []

    for i in range(pages_to_do):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        pix.save(img_path, output="jpg")
        image_paths.append(img_path)

    doc.close()
    return image_paths
