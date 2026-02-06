"""PDF to XML conversion module using pdfminer.six."""
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Optional
import logging

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

logger = logging.getLogger(__name__)


def pdf_to_xml(pdf_path: str | Path, output_xml_path: Optional[str | Path] = None) -> str:
    """
    Convert PDF file to XML format.
    
    Args:
        pdf_path: Path to the PDF file
        output_xml_path: Optional path to save XML file. If None, only returns XML string.
    
    Returns:
        XML string representation of the PDF content.
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF file is corrupted or empty
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty: {pdf_path}")
    
    logger.info(f"Converting PDF to XML: {pdf_path}")
    
    # Extract text from PDF
    output_string = io.StringIO()
    
    try:
        with open(pdf_path, 'rb') as pdf_file:
            # Extract text with layout parameters for better formatting
            laparams = LAParams(
                detect_vertical=True,
                all_texts=True,
                boxes_flow=None
            )
            
            extract_text_to_fp(
                pdf_file,
                output_string,
                laparams=laparams,
                codec='utf-8',
                maxpages=0,
                caching=True
            )
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise ValueError(f"PDF parsing error: {str(e)}")
    
    extracted_text = output_string.getvalue()
    output_string.close()
    
    if not extracted_text or len(extracted_text.strip()) < 10:
        raise ValueError("PDF contains no extractable text")
    
    # Create XML structure
    root = ET.Element("Document")
    
    # Add metadata
    meta = ET.SubElement(root, "Meta")
    ET.SubElement(meta, "Filename").text = escape_xml(str(pdf_path.name))
    ET.SubElement(meta, "SizeBytes").text = str(pdf_path.stat().st_size)
    ET.SubElement(meta, "ContentType").text = "application/pdf"
    
    # Add pages (currently grouping all text in a single element)
    # For more granular page-by-page extraction, you would need to use PDFPage
    pages = ET.SubElement(root, "Pages")
    page = ET.SubElement(pages, "Page")
    page.set("index", "1")
    
    # Add content - escape special characters
    content = ET.SubElement(page, "Content")
    content.text = escape_xml(extracted_text.strip())
    
    # Convert to pretty XML string
    xml_string = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
    
    # Remove XML declaration if it exists (minidom adds it by default)
    xml_string = '\n'.join(line for line in xml_string.split('\n') if not line.strip().startswith('<?xml'))
    xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
    
    logger.info(f"Successfully converted PDF to XML ({len(xml_string)} characters)")
    
    # Save to file if path provided
    if output_xml_path:
        output_path = Path(output_xml_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_string, encoding='utf-8')
        logger.info(f"Saved XML to: {output_path}")
    
    return xml_string


def escape_xml(text: str) -> str:
    """
    Escape special XML characters in text.
    
    Args:
        text: Raw text string
        
    Returns:
        XML-escaped string
    """
    if text is None:
        return ""
    
    # Replace special XML characters in correct order
    # & must be first to avoid double-escaping
    # Using character codes to avoid string literal issues
    result = text
    amp_entity = chr(38) + "amp;"
    lt_entity = chr(38) + "lt;"
    gt_entity = chr(38) + "gt;"
    quot_entity = chr(38) + "quot;"
    apos_entity = chr(38) + "apos;"
    
    result = result.replace(chr(38), amp_entity)
    result = result.replace(chr(60), lt_entity)
    result = result.replace(chr(62), gt_entity)
    result = result.replace(chr(34), quot_entity)
    result = result.replace(chr(39), apos_entity)
    
    return result


def validate_xml(xml_string: str) -> bool:
    """
    Validate XML string is well-formed.
    
    Args:
        xml_string: XML string to validate
        
    Returns:
        True if XML is valid, False otherwise
    """
    try:
        ET.fromstring(xml_string)
        return True
    except ET.ParseError as e:
        logger.warning(f"XML validation failed: {e}")
        return False