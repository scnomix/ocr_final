import streamlit as st
from pathlib import Path
import base64
import fitz  # PyMuPDF
import streamlit.components.v1 as components
import tempfile
import json

from ocr_service.pipeline import process_document
from ocr_service.classifier import DocumentType

st.set_page_config(page_title="OCR & Data Extraction", layout="wide")
st.title("📄 OCR & Data Extraction")
st.write("Upload a PDF and get its classification and extracted data instantly.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Only start processing when the user clicks
    if st.button("Submit PDF for Processing"):
        # Use an ephemeral temp directory for all file writes
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            pdf_path = tmpdir / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.getbuffer())

            # Render PDF pages to base64-encoded PNGs
            doc = fitz.open(stream=pdf_path.read_bytes(), filetype="pdf")
            page_imgs = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                page_imgs.append(f"data:image/png;base64,{b64}")
            doc.close()

            n_pages = len(page_imgs)
            js_pages = json.dumps(page_imgs)

            # Layout: wide viewer + extraction
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("PDF Viewer")
                html = f"""
<style>
  .viewer-container {{ width:100%; max-width:800px; margin:auto; }}
  .controls {{ text-align:center; margin-bottom:8px; }}
  .controls button {{ margin:0 4px; padding:6px 10px; }}
</style>
<div class="viewer-container">
  <div class="controls">
    <button id="prevBtn">← Prev</button>
    <span id="pageDisplay">1/{n_pages}</span>
    <button id="nextBtn">Next →</button>
    <button id="rotateBtn">⟳ Rotate</button>
  </div>
  <div style="text-align:center;">
    <img id="docImg" src="{page_imgs[0]}" style="max-width:100%; transform:rotate(0deg);" />
  </div>
</div>
<script>
(function(){{
  const pages = {js_pages};
  let idx = 0, rot = 0;
  const img = document.getElementById("docImg");
  const disp = document.getElementById("pageDisplay");

  document.getElementById("prevBtn").onclick = () => {{
    if(idx>0) idx--;
    img.src = pages[idx];
    disp.textContent = `${{idx+1}}/{n_pages}`;
  }};
  document.getElementById("nextBtn").onclick = () => {{
    if(idx<pages.length-1) idx++;
    img.src = pages[idx];
    disp.textContent = `${{idx+1}}/{n_pages}`;
  }};
  document.getElementById("rotateBtn").onclick = () => {{
    rot = (rot+90)%360;
    img.style.transform = `rotate(${{rot}}deg)`;
  }};
}})();
</script>
"""
                components.html(html, height=800, scrolling=False)

            with col2:
                st.subheader("Extraction Results")
                with st.spinner("Classifying..."):
                    doc_type = process_document.__globals__["classify_pdf"](str(pdf_path))
                st.markdown(f"**Document Type:** `{doc_type.name}`")

                with st.spinner("Extracting..."):
                    result = process_document(str(pdf_path))
                st.json(result)
