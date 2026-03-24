"""
Document Loader
Extracts text from PDF and TXT files with intelligent entity extraction.

Enhances metadata by dynamically extracting key entities (persons, organizations, roles)
from document content, enabling accurate attribution in RAG responses.
"""

from pypdf import PdfReader
from typing import List, Dict, Set
import logging
import re
from pathlib import Path
import subprocess

from PIL import Image, ImageOps

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


def extract_text_from_image(image_path: str) -> Dict[str, str]:
    """
    Extract text from image using multi-engine OCR.

    Strategy:
    1) Run Tesseract OCR
    2) Run EasyOCR
    3) Combine both outputs for better coverage (printed + handwriting)
    """
    text_tesseract = run_tesseract(image_path)
    text_easyocr = run_easyocr(image_path)

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
        "engine": engine
    }


def process_image(file_path: str, file_name: str, lang: str = "eng") -> List[Dict[str, any]]:
    """
    Extract text from an image document using OCR and return loader-compatible structure.
    """
    if pytesseract is None:
        raise Exception("pytesseract is not installed. Please install OCR dependencies.")

    try:
        ocr_warning = ""

        # OCR pipeline: Tesseract -> EasyOCR
        ocr_result = extract_text_from_image(file_path)
        text = ocr_result["text"]
        ocr_engine = ocr_result["engine"]

        has_visual_context = False

        # Soft fallback only when OCR produced no usable text
        if not text or not text.strip():
            ocr_warning = "OCR empty; using soft fallback text"
            text = "Handwritten content detected but partially extracted."

        # Always keep extracted text available for downstream LLM reasoning
        final_text = text.strip()

        entity_data = extract_entities(final_text)
        primary_entity = entity_data["primary_entity"]
        entity_header = f"SUBJECT: {primary_entity}\n"
        enriched_text = "Image Content:\n" + entity_header + "\n" + final_text

        if ocr_warning:
            logger.warning(f"🖼️ {file_name} | OCR warning: {ocr_warning}")
        else:
            logger.info(f"🖼️ {file_name} | OCR text extracted | Primary Entity: {primary_entity}")

        return [{
            "text": enriched_text,
            "metadata": {
                "file_name": file_name,
                "page_number": 1,
                "total_pages": 1,
                "file_type": "image",
                "ocr_used": True,
                "ocr_warning": ocr_warning or "",
                "ocr_engine": ocr_engine,
                "has_visual_context": has_visual_context,
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
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=200,
            thread_count=1,
        )

        if not images:
            page_texts.append("")
            continue

        ocr_text = pytesseract.image_to_string(images[0], lang=lang) or ""
        page_texts.append(ocr_text)

    return page_texts


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

        # First pass: standard text extraction via PyPDF
        pypdf_page_texts = [(page.extract_text() or "") for page in reader.pages]
        pypdf_full_text = "\n".join(pypdf_page_texts)

        # OCR fallback only when needed
        ocr_used = False
        if not pypdf_full_text or len(pypdf_full_text.strip()) < MIN_TEXT_THRESHOLD:
            logger.info(f"OCR fallback triggered for {file_name}")
            try:
                page_texts = run_ocr_page_texts(file_path)
                full_text = "\n".join(page_texts)
                ocr_used = True
            except Exception as e:
                logger.error(f"OCR failed for {file_name}: {str(e)}")
                # Keep PyPDF text as final fallback, even if sparse
                page_texts = pypdf_page_texts
                full_text = pypdf_full_text
                ocr_used = False
        else:
            logger.info(f"Text extracted normally for {file_name}")
            page_texts = pypdf_page_texts
            full_text = pypdf_full_text

        if not full_text or not full_text.strip():
            raise Exception("No text could be extracted from PDF (PyPDF + OCR fallback both empty)")
        
        # Perform intelligent entity extraction
        entity_data = extract_entities(full_text)
        primary_entity = entity_data["primary_entity"]
        if primary_entity == "Unknown" and pdf_author != "Unknown":
            primary_entity = pdf_author  # Fallback to PDF metadata if available

        logger.info(
            f"📄 {file_name} | Primary Entity: {primary_entity} | OCR used: {ocr_used} | "
            f"Organizations: {entity_data['entities']['organizations']}"
        )

        for page_num, text in enumerate(page_texts, start=1):
            text = text or ""

            if page_num == 1:
                # Inject primary entity into page 1 text for embedding context
                # This ensures the embedding captures the entity association
                entity_header = f"SUBJECT: {primary_entity}\n"
                
                metadata_lines = [entity_header]
                if pdf_title and pdf_title != file_name:
                    metadata_lines.append(f"DOCUMENT: {pdf_title}")
                if pdf_author and pdf_author != "Unknown":
                    metadata_lines.append(f"SOURCE: {pdf_author}")

                text = "\n".join(metadata_lines) + "\n\n" + text
            
            if text.strip():  # Only add non-empty pages
                documents.append({
                    "text": text,
                    "metadata": {
                        "file_name": file_name,
                        "page_number": page_num,
                        "total_pages": len(reader.pages),
                        "pdf_author": pdf_author,
                        "pdf_title": pdf_title,
                        "file_type": "pdf",
                        "ocr_used": ocr_used,
                        # Entity extraction results
                        "primary_entity": primary_entity,
                        "entities": entity_data["entities"],  # {persons, organizations, roles}
                        "entity_persons": entity_data["entities"]["persons"],
                        "entity_organizations": entity_data["entities"]["organizations"],
                        "entity_roles": entity_data["entities"]["roles"]
                    }
                })
        
        if not documents:
            raise Exception("All extracted pages are empty after processing")

        logger.info(f"✓ Extracted {len(documents)} pages from {file_name} (ocr_used={ocr_used})")
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
