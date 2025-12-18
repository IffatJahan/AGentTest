# import pytesseract
# from pdf2image import convert_from_path
pdf_path = r"E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\\3. Appendix.docx.pdf"
#
# # optional (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # 1. Convert PDF to images
# pages = convert_from_path(pdf_path, dpi=300)  # higher DPI = better accuracy
#
# full_text = ""
#
# # 2. OCR each page
# for i, page in enumerate(pages):
#     text = pytesseract.image_to_string(page, lang="eng")
#     full_text += f"\n--- Page {i+1} ---\n"
#     full_text += text
#
# # 3. Save output
# with open("report_text.txt", "w", encoding="utf-8") as f:
#     f.write(full_text)
#
# print("Done! Text saved to report_text.txt")
#
"""
   Maintain document structure and hierarchy
   """
import fitz

doc = fitz.open(pdf_path)
structured_chunks = []

for page_num, chunk in enumerate(text_chunks):
    page = doc[page_num]
    structured_chunk = {
        'content': chunk['text'],
        'metadata': {
            'page_number': page_num + 1,
            'section_headers': extract_headers(page),
            'font_sizes': extract_font_sizes(page),
            'paragraphs': identify_paragraphs(chunk['text']),
            'tables': extract_tables(page),  # If any
            'images': extract_image_captions(page)
        }
    }
    structured_chunks.append(structured_chunk)

doc.close()
















# from google.cloud import vision
# import io
# from PIL import Image
#
# client = vision.ImageAnnotatorClient()
#
# with open('scanned.pdf', 'rb') as pdf:
#     pages = convert_from_bytes(pdf.read())
#
# full_text = ""
# for page in pages:
#    image = vision.Image(content=page.tobytes)
#    response = client.document_text_detection(image=image)
#    full_text += response.text