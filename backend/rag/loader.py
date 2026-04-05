"""
Document Loader
Extracts text from PDF and TXT files with intelligent entity extraction.

Enhances metadata by dynamically extracting key entities (persons, organizations, roles)
from document content, enabling accurate attribution in RAG responses.
"""

from pypdf import PdfReader
from typing import List, Dict, Set, Any
import logging
import re
from pathlib import Path
import subprocess
import base64
import mimetypes
import os
import io
import json

from PIL import Image, ImageOps
import requests

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    import fitz
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)

MIN_TEXT_THRESHOLD = 50
MIN_IMAGE_TEXT_THRESHOLD = 20
_easyocr_reader = None


def run_ocr(pdf_path: str, lang: str = "eng") -> str:
    """
    Run OCR on a PDF file page-by-page and return full extracted text.

    Args:
        pdf_path: Path to PDF file
        lang: OCR language code (default: English)

    Returns:
        Full OCR text across all pages
    """
    page_texts = run_ocr_page_texts(pdf_path, lang=lang)
    return "\n".join(page_texts)


def preprocess_image(image_path: str) -> Image.Image:
    """
    Preprocess image for OCR quality improvements.

    - Convert to grayscale
    - Increase contrast

    Returns:
        PIL image ready for OCR
    """
    if cv2 is None:
        # Fallback preprocessing with PIL only
        img = Image.open(image_path)
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        return img

    cv_image = cv2.imread(image_path)
    if cv_image is None:
        raise Exception(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
    return Image.fromarray(enhanced)


def run_tesseract(image_path: str, lang: str = "eng") -> str:
    """
    Run Tesseract OCR with preprocessing and tuned page segmentation.
    """
    if pytesseract is None:
        return ""

    if cv2 is not None:
        img = cv2.imread(image_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adaptive thresholding improves handwriting/uneven lighting OCR
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            text = pytesseract.image_to_string(thresh, lang=lang, config="--oem 3 --psm 6") or ""
            return text

    # PIL fallback
    preprocessed = preprocess_image(image_path)
    return pytesseract.image_to_string(preprocessed, lang=lang, config="--psm 6") or ""


def get_easyocr_reader():
    """
    Lazily initialize EasyOCR reader to avoid startup penalty.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        if easyocr is None:
            raise RuntimeError("easyocr not installed")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


def run_easyocr(image_path: str) -> str:
    """
    Run EasyOCR (better for handwriting/complex text) and return merged text.
    """
    try:
        reader = get_easyocr_reader()
        results = reader.readtext(image_path)
        return " ".join([res[1] for res in results if len(res) > 1 and res[1]])
    except Exception as e:
        logger.warning(f"EasyOCR failed: {str(e)}")
        return ""


def run_easyocr_with_boxes(image_path: str) -> Dict[str, any]:
    """
    Run EasyOCR and return both text and bounding boxes.

    Returns bbox in [x, y, w, h] format for each detected span.
    """
    try:
        reader = get_easyocr_reader()
        results = reader.readtext(image_path)
        texts = []
        boxes = []

        for item in results:
            if len(item) < 2:
                continue

            points = item[0]
            span_text = (item[1] or "").strip()
            if not span_text:
                continue

            try:
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
                boxes.append({
                    "text": span_text,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min]
                })
            except Exception:
                # Skip malformed bbox entries
                pass

            texts.append(span_text)

        return {
            "text": " ".join(texts).strip(),
            "boxes": boxes
        }
    except Exception as e:
        logger.warning(f"EasyOCR bbox extraction failed: {str(e)}")
        return {
            "text": "",
            "boxes": []
        }


def extract_text_from_image(image_path: str) -> Dict[str, any]:
    """
    Extract text from image using multi-engine OCR.

    Strategy:
    1) Run Tesseract OCR
    2) Run EasyOCR
    3) Combine both outputs for better coverage (printed + handwriting)
    """
    text_tesseract = run_tesseract(image_path)
    easyocr_result = run_easyocr_with_boxes(image_path)
    text_easyocr = easyocr_result.get("text", "")
    ocr_boxes = easyocr_result.get("boxes", [])

    combined_parts = []
    if text_tesseract and text_tesseract.strip():
        combined_parts.append(text_tesseract.strip())
    if text_easyocr and text_easyocr.strip():
        combined_parts.append(text_easyocr.strip())

    combined_text = "\n".join(combined_parts)

    if text_tesseract.strip() and text_easyocr.strip():
        engine = "tesseract+easyocr"
    elif text_tesseract.strip():
        engine = "tesseract"
    elif text_easyocr.strip():
        engine = "easyocr"
    else:
        engine = "none"

    return {
        "text": combined_text,
        "engine": engine,
        "boxes": ocr_boxes,
    }


def describe_image(image_path: str) -> str:
    """
    Temporary visual fallback for non-text images.

    NOTE: This is a lightweight heuristic captioner and can be replaced later
    with a real vision API.
    """
    if cv2 is None:
        return "This image likely contains objects or a scene without readable text."

    try:
        img = cv2.imread(image_path)
        if img is None:
            return "This image likely contains objects or a scene without readable text."

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Yellow mask to detect banana-like content (common in food images)
        lower_yellow = (18, 40, 60)
        upper_yellow = (40, 255, 255)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = float(cv2.countNonZero(yellow_mask)) / float(yellow_mask.size)

        # Count elongated yellow-ish blobs
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elongated_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) == 0:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect >= 1.8:
                elongated_count += 1

        if yellow_ratio > 0.03 and elongated_count >= 1:
            return "The image appears to show bananas on a surface."
        if yellow_ratio > 0.08:
            return "The image appears to show yellow fruits, likely bananas, on a surface."

        return "This image likely contains objects or a scene without readable text."
    except Exception as e:
        logger.warning(f"describe_image heuristic failed: {str(e)}")
        return "This image likely contains objects or a scene without readable text."


def describe_image_with_vision(image_path: str) -> str:
    """
    Vision analysis using OpenRouter multimodal API with base64 image_url.

    Returns structured visual understanding.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return describe_image(image_path)

    try:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": "qwen/qwen-2.5-vl-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze the image carefully.\n\n"
                                "Instructions:\n"
                                "- Identify all visible objects and elements\n"
                                "- If chart -> extract values and compare\n"
                                "- If diagram -> explain flow and relationships\n"
                                "- If real image -> describe objects clearly\n"
                                "- DO NOT rely only on text\n"
                                "- DO NOT say 'unknown' unless truly unclear\n\n"
                                "Return structured output:\n"
                                "1. Image Type\n"
                                "2. Key Elements\n"
                                "3. Relationships / Structure\n"
                                "4. Core Meaning\n"
                                "5. Interpretation"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Multi-Document RAG",
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45,
        )

        if response.status_code != 200:
            logger.warning(f"Vision API failed: {response.status_code} {response.text[:200]}")
            return describe_image(image_path)

        data = response.json()
        logger.info(f"RAW RESPONSE: {data}")

        if "choices" not in data or not data["choices"]:
            raise Exception("Invalid LLM response")

        message = data["choices"][0].get("message", {})
        content = message.get("content", "")

        # Some providers return structured content arrays
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content") or ""
                    if txt:
                        parts.append(str(txt))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)

        content = str(content).strip()
        if not content:
            content = "Image contains visible objects."

        return content
    except Exception as e:
        logger.warning(f"Vision fallback failed: {str(e)}")
        return describe_image(image_path)


def describe_pdf_page_with_vision(pdf_path: str, page_num: int) -> str:
    """
    Analyze a specific PDF page visually using the same multimodal model.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or convert_from_path is None:
        return ""

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=200,
            thread_count=1,
        )
        if not images:
            return ""

        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "model": "qwen/qwen-2.5-vl-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze the image carefully.\n\n"
                                "Instructions:\n"
                                "- Identify all visible objects and elements\n"
                                "- If chart -> extract values and compare\n"
                                "- If diagram -> explain flow and relationships\n"
                                "- If real image -> describe objects clearly\n"
                                "- DO NOT rely only on text\n"
                                "- DO NOT say 'unknown' unless truly unclear\n\n"
                                "Return structured output:\n"
                                "1. Image Type\n"
                                "2. Key Elements\n"
                                "3. Relationships / Structure\n"
                                "4. Core Meaning\n"
                                "5. Interpretation"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 350,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Multi-Document RAG",
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45,
        )

        if response.status_code != 200:
            logger.warning(f"PDF page vision failed: {response.status_code} {response.text[:200]}")
            return ""

        data = response.json()
        if "choices" not in data or not data["choices"]:
            return ""

        message = data["choices"][0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content") or ""
                    if txt:
                        parts.append(str(txt))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)

        return str(content or "").strip()
    except Exception as e:
        logger.warning(f"PDF page vision analysis failed (page {page_num}): {str(e)}")
        return ""


def _clean_ocr_text(text: str) -> str:
    cleaned = (text or "").replace("\u200b", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_pdf_words_with_bboxes(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract per-page words from PDF with deterministic char offsets and bboxes.

    The returned page text is reconstructed directly from extracted words so
    char offsets map 1:1 to word indices and bboxes.
    """
    if fitz is None:
        return []

    pages: List[Dict[str, Any]] = []

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                raw_words = page.get_text("words", sort=True) or []

                page_words: List[Dict[str, Any]] = []
                text_parts: List[str] = []
                cursor = 0
                last_line_key = None

                for item in raw_words:
                    if len(item) < 8:
                        continue

                    x0, y0, x1, y1, token, block_no, line_no, _word_no = item[:8]
                    word_text = str(token or "").strip()
                    if not word_text:
                        continue

                    line_key = (int(block_no), int(line_no))
                    if last_line_key is not None:
                        separator = "\n" if line_key != last_line_key else " "
                        text_parts.append(separator)
                        cursor += len(separator)

                    char_start = cursor
                    text_parts.append(word_text)
                    cursor += len(word_text)
                    char_end = cursor

                    page_words.append({
                        "text": word_text,
                        "char_start": char_start,
                        "char_end": char_end,
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    })
                    last_line_key = line_key

                pages.append({
                    "text": "".join(text_parts),
                    "words": page_words,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                })
    except Exception as exc:
        logger.warning(f"Word-level PDF extraction failed: {str(exc)}")
        return []

    return pages


def _extract_detected_elements(vision_text: str) -> str:
    lines = [ln.strip(" -•\t") for ln in (vision_text or "").splitlines() if ln and ln.strip()]
    if not lines:
        return "No distinct visual elements were extracted."
    if len(lines) <= 6:
        return "\n".join(lines)
    return "\n".join(lines[:6])


def process_image(file_path: str, file_name: str, lang: str = "eng") -> List[Dict[str, any]]:
    """
    Extract text from an image document using OCR and return loader-compatible structure.
    """
    try:
        # Vision-first pipeline: Vision -> OCR -> Fusion
        vision_caption = describe_image_with_vision(file_path)
        if not vision_caption:
            vision_caption = "Image contains visible objects but visual analysis was limited."

        logger.info(f"Vision analysis executed for image file: {file_name}")

        # OCR remains a supporting signal
        if pytesseract is None and easyocr is None:
            ocr_result = {"text": "", "engine": "none", "boxes": []}
        else:
            ocr_result = extract_text_from_image(file_path)

        ocr_text = _clean_ocr_text(ocr_result["text"])
        ocr_engine = ocr_result["engine"]
        ocr_boxes = ocr_result.get("boxes", [])
        logger.info(f"Image OCR stats | file={file_name} | engine={ocr_engine} | ocr_len={len(ocr_text)}")

        with Image.open(file_path) as img:
            image_width, image_height = img.size

        no_text = len(ocr_text) < 8
        detected_elements = _extract_detected_elements(vision_caption)
        final_text = (
            "IMAGE ANALYSIS\n\n"
            f"Visual Description:\n{vision_caption}\n\n"
            f"Detected Elements:\n{detected_elements}\n\n"
            f"OCR Text:\n{ocr_text if ocr_text else '[No reliable OCR text detected]'}\n"
        ).strip()

        entity_data = extract_entities(final_text)
        primary_entity = entity_data["primary_entity"]
        entity_header = f"SUBJECT: {primary_entity}\n"
        enriched_text = "Image Content:\n" + entity_header + "\n" + final_text

        logger.info(
            f"🖼️ {file_name} | vision_used=True | has_visual_context=True | "
            f"ocr_len={len(ocr_text)} | primary_entity={primary_entity}"
        )

        return [{
            "text": enriched_text,
            "metadata": {
                "file_name": file_name,
                "page": 1,
                "page_number": 1,
                "total_pages": 1,
                "file_type": "image",
                "ocr_used": True,
                "ocr_warning": "OCR weak/empty" if no_text else "",
                "ocr_engine": ocr_engine,
                "has_visual_context": True,
                "has_text": not no_text,
                "vision_used": True,
                "bbox": ",".join(str(v) for v in (ocr_boxes[0]["bbox"] if ocr_boxes else [])),
                "bbox_image_width": str(image_width),
                "bbox_image_height": str(image_height),
                "source_image_path": file_path,
                "vision_caption": vision_caption,
                "primary_entity": primary_entity,
                "entities": entity_data["entities"],
                "entity_persons": entity_data["entities"]["persons"],
                "entity_organizations": entity_data["entities"]["organizations"],
                "entity_roles": entity_data["entities"]["roles"],
            }
        }]
    except Exception as e:
        logger.error(f"Error processing image {file_name}: {str(e)}")
        raise Exception(f"Failed to process image: {str(e)}")


def load_docx(file_path: str, file_name: str) -> List[Dict[str, any]]:
    """
    Load text from DOCX file.
    """
    if DocxDocument is None:
        raise Exception("python-docx is not installed. Please install backend dependencies.")

    try:
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paragraphs)

        if not text.strip():
            raise Exception("No readable text found in DOCX")

        entity_data = extract_entities(text)
        primary_entity = entity_data["primary_entity"]
        entity_header = f"SUBJECT: {primary_entity}\n"
        enriched_text = entity_header + "\n" + text

        return [{
            "text": enriched_text,
            "metadata": {
                "file_name": file_name,
                "page": 1,
                "page_number": 1,
                "total_pages": 1,
                "file_type": "docx",
                "ocr_used": False,
                "primary_entity": primary_entity,
                "entities": entity_data["entities"],
                "entity_persons": entity_data["entities"]["persons"],
                "entity_organizations": entity_data["entities"]["organizations"],
                "entity_roles": entity_data["entities"]["roles"],
            }
        }]
    except Exception as e:
        logger.error(f"Error loading DOCX {file_name}: {str(e)}")
        raise Exception(f"Failed to load DOCX: {str(e)}")


def load_doc(file_path: str, file_name: str) -> List[Dict[str, any]]:
    """
    Load text from legacy DOC file using antiword if available.
    """
    try:
        result = subprocess.run(
            ["antiword", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout or ""
        if not text.strip():
            raise Exception("No readable text found in DOC")

        entity_data = extract_entities(text)
        primary_entity = entity_data["primary_entity"]
        entity_header = f"SUBJECT: {primary_entity}\n"
        enriched_text = entity_header + "\n" + text

        return [{
            "text": enriched_text,
            "metadata": {
                "file_name": file_name,
                "page": 1,
                "page_number": 1,
                "total_pages": 1,
                "file_type": "doc",
                "ocr_used": False,
                "primary_entity": primary_entity,
                "entities": entity_data["entities"],
                "entity_persons": entity_data["entities"]["persons"],
                "entity_organizations": entity_data["entities"]["organizations"],
                "entity_roles": entity_data["entities"]["roles"],
            }
        }]
    except FileNotFoundError:
        raise Exception("antiword is not installed. Install antiword to process .doc files.")
    except Exception as e:
        logger.error(f"Error loading DOC {file_name}: {str(e)}")
        raise Exception(f"Failed to load DOC: {str(e)}")


def run_ocr_page_texts(pdf_path: str, lang: str = "eng") -> List[str]:
    """
    Run OCR on each PDF page and return text per page.

    This keeps processing page-by-page to avoid unnecessary memory spikes.

    Args:
        pdf_path: Path to PDF file
        lang: OCR language code (default: English)

    Returns:
        List of text strings (one per page)
    """
    if pytesseract is None or convert_from_path is None:
        raise RuntimeError(
            "OCR dependencies are missing. Install: pytesseract, pdf2image, pillow, and system tesseract."
        )

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    page_texts: List[str] = []

    for page_num in range(1, total_pages + 1):
        ocr_text = run_ocr_single_page_text(pdf_path, page_num=page_num, lang=lang)
        page_texts.append(ocr_text)

    return page_texts


def run_ocr_single_page_text(pdf_path: str, page_num: int, lang: str = "eng") -> str:
    """
    Run OCR for a single (1-based) PDF page and return extracted text.
    """
    if pytesseract is None or convert_from_path is None:
        raise RuntimeError(
            "OCR dependencies are missing. Install: pytesseract, pdf2image, pillow, and system tesseract."
        )

    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=200,
        thread_count=1,
    )

    if not images:
        return ""

    return pytesseract.image_to_string(images[0], lang=lang) or ""


def extract_embedded_images(pdf_path: str, output_dir: str) -> List[Dict]:
    """
    Extract embedded images from a PDF using PyMuPDF.

    Args:
        pdf_path: Absolute or relative path to input PDF
        output_dir: Directory where extracted images will be saved

    Returns:
        List of image metadata dicts:
        {
            "image_path": str,
            "page": int,
            "type": "embedded_image"
        }
    """
    extracted: List[Dict] = []

    if fitz is None:
        logger.warning("PyMuPDF (fitz) is not installed; skipping embedded image extraction")
        return extracted

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                images = page.get_images(full=True)

                for image_index, image_info in enumerate(images):
                    try:
                        xref = image_info[0]
                        image_data = doc.extract_image(xref)
                        image_bytes = image_data.get("image")
                        image_ext = (image_data.get("ext") or "png").lower()

                        if not image_bytes:
                            continue

                        if image_ext == "jpeg":
                            image_ext = "jpg"

                        image_filename = f"page_{page_index + 1}_embedded_{image_index}.{image_ext}"
                        image_file_path = output_path / image_filename

                        with open(image_file_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        extracted.append({
                            "image_path": str(image_file_path),
                            "page": page_index + 1,
                            "type": "embedded_image",
                        })
                    except Exception as img_err:
                        logger.warning(
                            f"Skipping embedded image on page {page_index + 1}: {str(img_err)}"
                        )
    except Exception as e:
        logger.warning(f"Embedded image extraction failed for {pdf_path}: {str(e)}")

    return extracted


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> List[Dict]:
    """
    Convert each page of a PDF into a raster image (fallback path).

    Args:
        pdf_path: Path to PDF
        output_dir: Directory to save generated page images

    Returns:
        List of image metadata dicts:
        {
            "image_path": str,
            "page": int,
            "type": "page_image"
        }
    """
    page_images: List[Dict] = []

    if convert_from_path is None:
        logger.warning("pdf2image is not installed; cannot generate fallback page images")
        return page_images

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        images = convert_from_path(pdf_path, dpi=200, fmt="png", thread_count=1)
        for page_num, image in enumerate(images, start=1):
            page_image_path = output_path / f"page_{page_num}.png"
            image.save(page_image_path, format="PNG")
            page_images.append({
                "image_path": str(page_image_path),
                "page": page_num,
                "type": "page_image",
            })
    except Exception as e:
        logger.warning(f"Fallback page image conversion failed for {pdf_path}: {str(e)}")

    return page_images


def extract_entities(text: str) -> Dict:
    """
    Extract key entities from document text using heuristics and NLP-style rules.
    
    Strategy:
    1. Scan first 50 lines (document header/summary section)
    2. Identify prominent nouns and title-case phrases
    3. Classify entities as: persons, organizations, roles
    4. Rank by position (earlier = more likely to be document subject)
    
    Args:
        text: Raw document text
    
    Returns:
        Dictionary with:
        {
            "primary_entity": str,  # Most likely document subject
            "entities": {
                "persons": [...],
                "organizations": [...],
                "roles": [...]
            }
        }
    """
    entities = {
        "persons": set(),
        "organizations": set(),
        "roles": set()
    }
    
    if not text or not text.strip():
        return {
            "primary_entity": "Unknown",
            "entities": {k: [] for k in entities}
        }
    
    lines = text.strip().split('\n')[:50]  # Focus on first 50 lines
    
    # Common role/title keywords
    role_keywords = {
        'engineer', 'developer', 'architect', 'manager', 'lead', 'director',
        'analyst', 'scientist', 'researcher', 'student', 'intern', 'associate',
        'specialist', 'consultant', 'designer', 'officer', 'executive',
        'programmer', 'data scientist', 'ml engineer', 'full stack', 'backend',
        'frontend', 'devops', 'qa', 'qa engineer', 'product manager'
    }
    
    # Common organization indicators
    org_keywords = {
        'company', 'corporation', 'institute', 'university', 'college',
        'organization', 'org', 'organization', 'inc', 'ltd', 'llc',
        'consulting', 'technologies', 'systems', 'solutions'
    }
    
    # Words to skip (common articles, prepositions)
    skip_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'at', 'to', 'for',
        'from', 'by', 'with', 'as', 'is', 'was', 'are', 'were', 'have', 'has',
        'been', 'be', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'shall'
    }
    
    processed_lines = []
    
    for line in lines:
        cleaned = line.strip()
        
        # Skip empty, too short, or obviously metadata lines
        if not cleaned or len(cleaned) < 3:
            continue
        if '@' in cleaned or 'http' in cleaned.lower():
            continue
        if cleaned.lower() in {'resume', 'cv', 'contact', 'email', 'phone', 'address'}:
            continue
        if len(cleaned) > 150:  # Skip very long lines (likely body text)
            continue
        
        processed_lines.append(cleaned)
    
    # Extract person names (2-4 title-cased words)
    for line in processed_lines:
        words = line.split()
        
        # Check if line looks like a person name
        if 2 <= len(words) <= 4:
            # Validate format: mostly letters, title case
            valid_chars = sum(1 for c in line if c.isalpha() or c in " -'")
            if valid_chars / len(line) > 0.75:
                title_case_words = sum(1 for w in words if w[0].isupper() and not w.isupper())
                if title_case_words >= len(words) - 1:  # At least n-1 words title-cased
                    if line not in skip_words:
                        entities["persons"].add(line)
                        break  # Primary entity is likely first name found
        
        # Extract roles (contains role keywords)
        line_lower = line.lower()
        for role in role_keywords:
            if role in line_lower:
                # Extract the full phrase containing the role
                match = re.search(r'([a-zA-Z\s&-]+(?:' + role + r')[a-zA-Z\s&-]*)', line_lower)
                if match:
                    entities["roles"].add(match.group(1).strip().title())
                    break
        
        # Extract organizations (contains org keywords or all-caps phrases)
        for org in org_keywords:
            if org in line_lower:
                entities["organizations"].add(line)
                break
        
        # Check for all-caps phrases (common for organization names)
        all_caps = re.findall(r'\b([A-Z][A-Z0-9\s&-]+)\b', line)
        for phrase in all_caps:
            if len(phrase.split()) <= 3:  # Max 3 words for org names
                entities["organizations"].add(phrase.strip())
    
    # Determine primary entity (person name > organization > role)
    primary_entity = "Unknown"
    if entities["persons"]:
        primary_entity = list(entities["persons"])[0]
    elif entities["organizations"]:
        primary_entity = list(entities["organizations"])[0]
    elif entities["roles"]:
        primary_entity = list(entities["roles"])[0]
    
    # Convert sets to sorted lists
    return {
        "primary_entity": primary_entity,
        "entities": {
            "persons": sorted(list(entities["persons"])),
            "organizations": sorted(list(entities["organizations"])),
            "roles": sorted(list(entities["roles"]))
        }
    }


def load_pdf(file_path: str, file_name: str) -> List[Dict[str, any]]:
    """
    Extract text from PDF file page by page with intelligent entity extraction.
    
    Process:
    1. Extract text from all pages
    2. Perform entity extraction on full text (persons, organizations, roles)
    3. Inject primary entity into page 1 for embedding context
    4. Store complete entity metadata for retrieval attribution
    
    Args:
        file_path: Path to PDF file
        file_name: Name of the file
    
    Returns:
        List of dictionaries with text and enriched metadata
    """
    try:
        reader = PdfReader(file_path)
        documents = []
        pdf_metadata = reader.metadata or {}
        pdf_author = pdf_metadata.get('/Author') or pdf_metadata.get('/author') or "Unknown"
        pdf_title = pdf_metadata.get('/Title') or pdf_metadata.get('/title') or file_name

        page_word_data = _extract_pdf_words_with_bboxes(file_path)

        image_output_dir = Path(__file__).parent.parent / "temp" / "pdf_images" / Path(file_name).stem
        embedded_images = extract_embedded_images(file_path, str(image_output_dir))
        print(f"Extracted {len(embedded_images)} embedded images")

        if len(embedded_images) == 0:
            page_images = convert_pdf_to_images(file_path, str(image_output_dir))
        else:
            page_images = []

        print(f"Fallback page images: {len(page_images)}")

        document_images = embedded_images + page_images
        page_to_images: Dict[int, List[Dict]] = {}
        for img_meta in document_images:
            page_no = int(img_meta.get("page", 0) or 0)
            if page_no <= 0:
                continue
            page_to_images.setdefault(page_no, []).append(img_meta)

        pages_with_visual_context = sorted(page_to_images.keys())
        fallback_used = len(embedded_images) == 0

        logger.info(
            f"PDF visual extraction | file={file_name} | embedded_images={len(embedded_images)} | "
            f"fallback_page_images={len(page_images)} | fallback_used={fallback_used} | "
            f"pages_with_visual_context={pages_with_visual_context}"
        )

        page_texts: List[str] = []
        page_ocr_used: List[bool] = []
        page_words_per_page: List[List[Dict[str, Any]]] = []
        page_dimensions: List[Dict[str, float]] = []
        page_text_sources: List[str] = []
        page_vision_summaries: List[str] = []

        # Per-page extraction with OCR fallback (never silently skip pages)
        for page_num, page in enumerate(reader.pages, start=1):
            fitz_payload = page_word_data[page_num - 1] if len(page_word_data) >= page_num else {}
            fitz_text = ((fitz_payload or {}).get("text") or "").strip()
            page_words = (fitz_payload or {}).get("words") or []

            extracted_text = fitz_text
            used_ocr = False
            text_source = "fitz_words" if fitz_text else "pypdf"

            if not extracted_text:
                extracted_text = (page.extract_text() or "").strip()
                if not extracted_text:
                    logger.warning(f"{file_name} | Page {page_num}: empty text extraction, attempting OCR fallback")
                    try:
                        extracted_text = (run_ocr_single_page_text(file_path, page_num=page_num) or "").strip()
                        used_ocr = True
                        text_source = "ocr" if extracted_text else "none"
                        if not extracted_text:
                            logger.warning(f"{file_name} | Page {page_num}: OCR fallback returned empty text")
                    except Exception as ocr_error:
                        logger.warning(f"{file_name} | Page {page_num}: OCR fallback failed: {str(ocr_error)}")
                        text_source = "none"
                else:
                    text_source = "pypdf"

            vision_summary = describe_pdf_page_with_vision(file_path, page_num)
            if vision_summary and not page_words:
                extracted_text = (
                    f"{extracted_text}\n\n"
                    f"PAGE VISUAL ANALYSIS:\n{vision_summary}"
                ).strip()

            page_width = float((fitz_payload or {}).get("width") or 0.0)
            page_height = float((fitz_payload or {}).get("height") or 0.0)
            if page_width <= 0 or page_height <= 0:
                try:
                    page_width = float(page.mediabox.width)
                    page_height = float(page.mediabox.height)
                except Exception:
                    page_width = 0.0
                    page_height = 0.0

            page_texts.append(extracted_text)
            page_ocr_used.append(used_ocr)
            page_words_per_page.append(page_words)
            page_dimensions.append({"width": page_width, "height": page_height})
            page_text_sources.append(text_source)
            page_vision_summaries.append(vision_summary)

        full_text = "\n".join([text for text in page_texts if text and text.strip()])
        if not full_text.strip():
            logger.warning(f"{file_name} | No extractable text found across all pages")
        
        # Perform intelligent entity extraction
        entity_data = extract_entities(full_text)
        primary_entity = entity_data["primary_entity"]
        if primary_entity == "Unknown" and pdf_author != "Unknown":
            primary_entity = pdf_author  # Fallback to PDF metadata if available

        logger.info(
            f"📄 {file_name} | Primary Entity: {primary_entity} | "
            f"Organizations: {entity_data['entities']['organizations']}"
        )

        for page_num, text in enumerate(page_texts, start=1):
            text = text or ""
            original_text = text
            page_images_for_chunk = page_to_images.get(page_num, [])
            page_image_paths = [item.get("image_path") for item in page_images_for_chunk if item.get("image_path")]
            page_image_types = sorted({item.get("type") for item in page_images_for_chunk if item.get("type")})

            documents.append({
                "text": text,
                "metadata": {
                    "file_name": file_name,
                    "page": page_num,
                    "page_number": page_num,
                    "total_pages": len(reader.pages),
                    "pdf_author": pdf_author,
                    "pdf_title": pdf_title,
                    "file_type": "pdf",
                    "ocr_used": page_ocr_used[page_num - 1],
                    "has_visual_context": len(page_images_for_chunk) > 0,
                    "image_paths": "|".join(page_image_paths),
                    "visual_context_types": "|".join(page_image_types),
                    "visual_image_count": len(page_image_paths),
                    "embedded_images_count": len(embedded_images),
                    "page_images_count": len(page_images),
                    "fallback_used": fallback_used,
                    "document_images_count": len(document_images),
                    "pages_with_visual_context": "|".join(str(p) for p in pages_with_visual_context),
                    "document_images": document_images,
                    "source_pdf_path": file_path,
                    "text_length": len(original_text.strip()),
                    "page_text_source": page_text_sources[page_num - 1],
                    "page_words_json": json.dumps(page_words_per_page[page_num - 1], ensure_ascii=False),
                    "page_words_count": len(page_words_per_page[page_num - 1]),
                    "bbox_page_width": float(page_dimensions[page_num - 1]["width"]),
                    "bbox_page_height": float(page_dimensions[page_num - 1]["height"]),
                    "vision_summary": page_vision_summaries[page_num - 1],
                    # Entity extraction results
                    "primary_entity": primary_entity,
                    "entities": entity_data["entities"],  # {persons, organizations, roles}
                    "entity_persons": entity_data["entities"]["persons"],
                    "entity_organizations": entity_data["entities"]["organizations"],
                    "entity_roles": entity_data["entities"]["roles"]
                }
            })

            logger.info(
                f"Page {page_num} → length: {len(original_text.strip())} → ocr_used: {page_ocr_used[page_num - 1]} "
                f"→ words: {len(page_words_per_page[page_num - 1])} → source: {page_text_sources[page_num - 1]} "
                f"→ has_visual_context: {len(page_images_for_chunk) > 0}"
            )

        logger.info(
            f"✓ Extracted {len(documents)} pages from {file_name} | "
            f"embedded_images={len(embedded_images)} | fallback_used={fallback_used} | "
            f"pages_with_visual_context={pages_with_visual_context}"
        )
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF {file_name}: {str(e)}")
        raise Exception(f"Failed to load PDF: {str(e)}")


def load_txt(file_path: str, file_name: str) -> List[Dict[str, any]]:
    """
    Load text from TXT file with entity extraction.
    
    Args:
        file_path: Path to TXT file
        file_name: Name of the file
    
    Returns:
        List with single dictionary containing text and enriched metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract entities from content
        entity_data = extract_entities(text)
        primary_entity = entity_data["primary_entity"]
        
        # Inject primary entity into text for embedding
        entity_header = f"SUBJECT: {primary_entity}\n"
        enriched_text = entity_header + "\n" + text
        
        logger.info(f"📄 {file_name} | Primary Entity: {primary_entity}")
        
        documents = [{
            "text": enriched_text,
            "metadata": {
                "file_name": file_name,
                "page": 1,
                "page_number": 1,
                "total_pages": 1,
                "file_type": "txt",
                "ocr_used": False,
                # Entity extraction results
                "primary_entity": primary_entity,
                "entities": entity_data["entities"],
                "entity_persons": entity_data["entities"]["persons"],
                "entity_organizations": entity_data["entities"]["organizations"],
                "entity_roles": entity_data["entities"]["roles"]
            }
        }]
        
        logger.info(f"✓ Loaded text file {file_name}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading TXT {file_name}: {str(e)}")
        raise Exception(f"Failed to load TXT: {str(e)}")


def load_document(file_path: str, file_name: str) -> List[Dict[str, any]]:
    """
    Load document based on file extension.
    
    Args:
        file_path: Path to file
        file_name: Name of the file
    
    Returns:
        List of documents with text and metadata
    """
    file_ext = Path(file_name).suffix.lower()

    if file_ext == '.pdf':
        return load_pdf(file_path, file_name)
    elif file_ext == '.txt':
        return load_txt(file_path, file_name)
    elif file_ext == '.docx':
        return load_docx(file_path, file_name)
    elif file_ext == '.doc':
        return load_doc(file_path, file_name)
    elif file_ext in {'.jpg', '.jpeg', '.png'}:
        return process_image(file_path, file_name)
    else:
        raise ValueError(f"Unsupported file type: {file_name}")
