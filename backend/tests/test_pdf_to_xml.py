"""Unit tests for pdf_to_xml module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from app.pdf_to_xml import pdf_to_xml, validate_xml, extract_text_from_pdf


class TestExtractTextFromPDF:
    """Test text extraction from PDF."""
    
    @patch('app.pdf_to_pdf.HighLevelExtractor')
    def test_extract_text_from_pdf_success(self, mock_extractor_class):
        """Test successful text extraction from PDF."""
        # Setup mock
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_pages.return_value = [
            MagicMock(get_text=lambda: "Page 1 text"),
            MagicMock(get_text=lambda: "Page 2 text"),
        ]
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4')
            pdf_path = Path(tmp.name)
        
        try:
            # Execute
            pages = extract_text_from_pdf(pdf_path)
            
            # Assert
            assert len(pages) == 2
            assert pages[0] == "Page 1 text"
            assert pages[1] == "Page 2 text"
            mock_extractor.extract_pages.assert_called_once()
        finally:
            pdf_path.unlink()
    
    @patch('app.pdf_to_pdf.HighLevelExtractor')
    def test_extract_text_from_pdf_error(self, mock_extractor_class):
        """Test error handling in PDF text extraction."""
        # Setup mock to raise exception
        mock_extractor_class.side_effect = Exception("PDF read error")
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_path = Path(tmp.name)
        
        try:
            # Execute and assert
            with pytest.raises(Exception, match="PDF read error"):
                extract_text_from_pdf(pdf_path)
        finally:
            pdf_path.unlink()


class TestPDFToXML:
    """Test PDF to XML conversion."""
    
    @patch('app.pdf_to_xml.extract_text_from_pdf')
    def test_pdf_to_xml_success(self, mock_extract):
        """Test successful PDF to XML conversion."""
        # Setup mock
        mock_extract.return_value = ["Page 1 content", "Page 2 content"]
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4')
            pdf_path = Path(tmp.name)
        
        try:
            # Execute
            xml_string = pdf_to_xml(pdf_path)
            
            # Assert
            assert xml_string is not None
            assert isinstance(xml_string, str)
            assert "<?xml version" in xml_string
            assert "<Document>" in xml_string
            assert "<Pages>" in xml_string
            assert "<Page index=\"1\">Page 1 content</Page>" in xml_string
            assert "<Page index=\"2\">Page 2 content</Page>" in xml_string
            assert "</Pages>" in xml_string
            assert "</Document>" in xml_string
        finally:
            pdf_path.unlink()
    
    @patch('app.pdf_to_xml.extract_text_from_pdf')
    def test_pdf_to_xml_empty_pages(self, mock_extract):
        """Test PDF to XML conversion with empty pages."""
        # Setup mock
        mock_extract.return_value = []
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_path = Path(tmp.name)
        
        try:
            # Execute
            xml_string = pdf_to_xml(pdf_path)
            
            # Assert
            assert xml_string is not None
            assert "<Document>" in xml_string
            assert "<Pages>" in xml_string
            assert "</Pages>" in xml_string
            assert "</Document>" in xml_string
        finally:
            pdf_path.unlink()
    
    @patch('app.pdf_to_xml.extract_text_from_pdf')
    def test_pdf_to_xml_special_characters(self, mock_extract):
        """Test PDF to XML conversion with special characters."""
        # Setup mock with special characters
        mock_extract.return_value = ["Text with < > & ' \" special chars"]
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_path = Path(tmp.name)
        
        try:
            # Execute
            xml_string = pdf_to_xml(pdf_path)
            
            # Assert - special characters should be escaped
            assert "<" in xml_string
            assert ">" in xml_string
            assert "&" in xml_string
            assert "'" in xml_string
            assert """ in xml_string
            
            # Validate it's valid XML
            assert validate_xml(xml_string)
        finally:
            pdf_path.unlink()


class TestValidateXML:
    """Test XML validation."""
    
    def test_validate_xml_valid(self):
        """Test validation of valid XML."""
        valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Document>
    <Pages>
        <Page index="1">Content</Page>
    </Pages>
</Document>"""
        
        # Assert
        assert validate_xml(valid_xml) is True
    
    def test_validate_xml_invalid(self):
        """Test validation of invalid XML."""
        invalid_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Document>
    <Pages>
        <Page index="1">Content
    </Pages>
</Document>"""  # Missing closing tag
        
        # Assert
        assert validate_xml(invalid_xml) is False
    
    def test_validate_xml_empty(self):
        """Test validation of empty string."""
        empty_xml = ""
        
        # Assert
        assert validate_xml(empty_xml) is False
    
    def test_validate_xml_malformed(self):
        """Test validation of malformed XML."""
        malformed_xml = "<?xml version><UnclosedTag"
        
        # Assert
        assert validate_xml(malformed_xml) is False