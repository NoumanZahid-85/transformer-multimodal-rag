import re
import fitz
import pdfplumber
from PIL import Image
import io

ARXIV_HEADER = "arXiv:"
SECTION_PATTERN = re.compile(r"^(\d+\.?\d*\.?\d*)\s+[A-Z]")


def _is_arxiv_or_page_number(text, block_y0, page_height):
    if ARXIV_HEADER in text:
        return True
    if block_y0 > page_height * 0.85 and len(text.strip()) < 10:
        return True
    if block_y0 < page_height * 0.05 and len(text.strip()) < 15:
        return True
    return False


def _looks_like_heading(text, font_size, is_bold):
    if not text:
        return False
    if len(text) < 4:
        return False
    if SECTION_PATTERN.match(text):
        return True
    section_keywords = [
        "Abstract", "Introduction", "Background", "Conclusion",
        "References", "Appendix", "Related Work", "Method",
        "Results", "Discussion", "Experiments", "Training",
        "Model Architecture", "Attention", "Positional",
        "Why Self-Attention", "Applications", "Encoder", "Decoder"
    ]
    if any(text.startswith(kw) for kw in section_keywords):
        return True
    if font_size > 12:
        return True
    if font_size > 10.5 and is_bold:
        return True
    return False


def extract_text_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    last_heading = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        page_heading = None
        page_text_lines = []
        heading_found_here = False

        for block in blocks:
            if block["type"] != 0:
                continue
            block_y0 = block["bbox"][1]
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                font_size = spans[0]["size"]
                is_bold = "Bold" in spans[0]["font"] or "Semibold" in spans[0]["font"]
                line_text = "".join(s["text"] for s in spans).strip()

                if not line_text:
                    continue
                if _is_arxiv_or_page_number(line_text, block_y0, page_height):
                    continue

                if _looks_like_heading(line_text, font_size, is_bold):
                    page_heading = line_text
                    heading_found_here = True
                else:
                    page_text_lines.append(line_text)

        effective_heading = page_heading if page_heading else last_heading
        if page_heading:
            last_heading = page_heading

        para = [l for l in page_text_lines if not (len(l) < 5 and any(c.isdigit() for c in l))]
        full_text = " ".join(para)
        paragraphs = [p.strip() for p in full_text.split(". ") if p.strip()]
        current_chunk = []
        current_len = 0

        for para_text in paragraphs:
            para_len = len(para_text)
            if current_len + para_len > 1000 and current_chunk:
                content = ". ".join(current_chunk) + "."
                if len(content) > 100:
                    chunks.append({
                        "content": content,
                        "type": "text",
                        "source_page": page_num + 1,
                        "heading": effective_heading
                    })
                current_chunk = []
                current_len = 0
            current_chunk.append(para_text)
            current_len += para_len

        if current_chunk:
            content = ". ".join(current_chunk) + "."
            if len(content) > 100:
                chunks.append({
                    "content": content,
                    "type": "text",
                    "source_page": page_num + 1,
                    "heading": effective_heading
                })

    doc.close()
    return chunks


def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")
            images.append({
                "image": image,
                "source_page": page_num + 1,
                "index": img_idx
            })

    doc.close()
    return images


def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted = page.extract_tables()
            for table_data in extracted:
                if table_data and len(table_data) > 1:
                    rows = []
                    for row in table_data:
                        cleaned = [cell.strip() if cell else "" for cell in row]
                        rows.append(cleaned)
                    if any(cell for row in rows for cell in row):
                        tables.append({
                            "data": rows,
                            "source_page": page_num + 1
                        })
    return tables
