"""
Data Models for SEC Filings and Extraction Results
====================================================

This module defines the core data structures used throughout the MD&A
extraction pipeline. These dataclasses provide type-safe containers
for filing metadata, extraction results, and error tracking.

Data Models:
- Filing: Metadata about a single SEC filing (CIK, company, date, form type)
- ExtractionResult: Results from MD&A extraction including text and tables
- ProcessingError: Error information for failed extractions

Design Principles:
- Use Python dataclasses for clean, immutable-ish data structures
- Provide computed properties for common derived values
- Include serialization methods for logging and export
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class Filing:
    """
    Represents a 10-K or 10-Q SEC filing and its metadata.

    This dataclass captures the essential identifying information for a
    single SEC filing. It is created during the metadata parsing phase
    and used throughout the extraction pipeline.

    Attributes:
        cik: Central Index Key (10-digit SEC identifier)
        company_name: Legal name of the filing company
        filing_date: Date the filing was submitted to SEC
        form_type: Type of filing ("10-K", "10-K/A", "10-Q", "10-Q/A")
        file_path: Local path to the filing document
    """
    cik: str
    company_name: str
    filing_date: datetime
    form_type: str  # "10-K", "10-K/A", "10-Q", or "10-Q/A"
    file_path: Path

    def __post_init__(self):
        """Post-initialization processing to normalize CIK format."""
        # CIK must be exactly 10 digits, zero-padded on the left
        # This ensures consistent identification across all filings
        self.cik = self.cik.zfill(10)

    @property
    def is_amended(self) -> bool:
        """
        Check if this is an amended filing (/A suffix).

        Amended filings (10-K/A or 10-Q/A) supersede the original
        filing and may contain corrections or additional disclosures.
        """
        return "/A" in self.form_type or self.form_type.endswith("A")


@dataclass
class ExtractionResult:
    """
    Results from MD&A extraction including text, tables, and metadata.

    This dataclass is the primary output of the extraction pipeline,
    containing the extracted MD&A text along with identified tables,
    cross-references, and extraction metadata.

    Attributes:
        filing: The Filing object with source document metadata
        mdna_text: The extracted MD&A text content
        tables: List of Table objects identified within the MD&A
        cross_references: List of CrossReference objects found
        extraction_metadata: Additional extraction details (positions, warnings, etc.)
    """
    filing: Filing
    mdna_text: str
    tables: List[Any]  # List of Table objects from table_parser
    cross_references: List[Any]  # List of CrossReference objects
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """
        Check if extraction was successful.

        Returns True if MD&A text was extracted (non-empty).
        """
        return bool(self.mdna_text)

    @property
    def statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics as a dictionary.

        Useful for logging, reporting, and aggregating extraction
        results across multiple filings.
        """
        return {
            "cik": self.filing.cik,
            "filing_date": self.filing.filing_date.isoformat(),
            "form_type": self.filing.form_type,
            "word_count": self.extraction_metadata.get("word_count", 0),
            "table_count": len(self.tables),
            "cross_reference_count": len(self.cross_references),
            "has_warnings": bool(self.extraction_metadata.get("warnings", [])),
        }


@dataclass
class ProcessingError:
    """
    Represents an error encountered during processing.

    Used to track and report errors that occur during extraction,
    allowing for batch processing to continue while logging failures.

    Attributes:
        file_path: Path to the file that caused the error
        error_type: Category of error (e.g., "ParseError", "IOError")
        error_message: Detailed description of what went wrong
        timestamp: When the error occurred
    """
    file_path: Path
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to dictionary for logging and serialization.

        Returns a JSON-serializable dictionary with all error details.
        """
        return {
            "file": str(self.file_path),
            "type": self.error_type,
            "message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }