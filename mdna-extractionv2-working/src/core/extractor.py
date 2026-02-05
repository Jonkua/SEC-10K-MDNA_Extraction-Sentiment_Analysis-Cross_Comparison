"""
Enhanced MD&A Extractor with True In-Place Table Preservation
==============================================================

This module provides the core functionality for extracting Management's Discussion
and Analysis (MD&A) sections from SEC 10-K and 10-Q filings. The key innovation is
the "in-place" preservation of tables and structured data, meaning tables remain
embedded within the text at their original positions rather than being extracted
separately.

Key Features:
- Automatic detection of HTML vs. plain text documents
- Structure-preserving HTML-to-text conversion
- Intelligent table detection using multiple heuristics
- Cross-reference resolution for referenced documents
- Metadata extraction (CIK, company name, filing date, form type)
- Support for incorporation by reference scenarios

The extraction process follows these steps:
1. Read and detect document format (HTML or text)
2. Parse filing metadata from document headers
3. Locate MD&A section boundaries using pattern matching
4. Process content while preserving table structures
5. Identify and catalog tables for metadata
6. Resolve any cross-references
7. Save the extracted content with metadata header
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

# Internal module imports for configuration and utilities
from ...config.patterns import compile_patterns
from ...src.core.file_handler import FileHandler
from ...src.parsers.section_parser import SectionParser
from ...src.parsers.table_parser import TableParser, Table
from ...src.parsers.cross_reference_parser import CrossReferenceParser
from ...src.parsers.text_normalizer import TextNormalizer
from ...src.utils.logger import get_logger, log_error
from ...src.models.filing import Filing, ExtractionResult
from ...config.settings import MAX_ERRORS_PER_FILE
import html

# Initialize module-level logger for tracking extraction progress and errors
logger = get_logger(__name__)


class MDNAExtractor:
    """
    Main class for extracting MD&A sections with true in-place table preservation.

    This extractor is designed to handle the complex structure of SEC filings,
    which often contain embedded tables, cross-references, and varying formats.
    The "in-place" preservation strategy ensures that financial tables remain
    contextually positioned within the MD&A narrative, which is crucial for
    downstream sentiment analysis and understanding.

    Attributes:
        output_dir: Directory where extracted MD&A sections will be saved
        file_handler: Utility for reading files with automatic encoding detection
        section_parser: Parser for locating MD&A section boundaries
        table_parser: Parser for identifying and extracting table structures
        cross_ref_parser: Parser for finding and resolving cross-references
        normalizer: Utility for text normalization and cleaning
        patterns: Compiled regex patterns for various extraction tasks
        error_count: Tracks errors encountered during processing
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the MDNAExtractor with required components.

        Args:
            output_dir: Path to directory where extracted MD&A files will be saved.
                       The directory will be created if it doesn't exist.
        """
        self.output_dir = Path(output_dir)

        # Initialize file I/O handler with encoding detection capabilities
        self.file_handler = FileHandler()

        # Parser for locating MD&A section start/end positions using regex patterns
        self.section_parser = SectionParser()

        # Parser for extracting and formatting financial tables
        self.table_parser = TableParser()

        # Parser for finding references to other documents (e.g., "See Item 7A")
        self.cross_ref_parser = CrossReferenceParser()

        # Text normalizer for cleaning and standardizing extracted content
        self.normalizer = TextNormalizer()

        # Pre-compiled regex patterns for form types, dates, CIKs, etc.
        self.patterns = compile_patterns()

        # Counter to track and limit errors per file (prevents infinite loops)
        self.error_count = 0

    def extract_from_file(self, file_path: Path, reference_resolver=None) -> Optional[ExtractionResult]:
        """
        Extract MD&A from a single filing file with in-place table preservation.

        This is the main entry point for processing a single SEC filing. The method
        handles the complete extraction pipeline including format detection,
        preprocessing, section location, and content processing.

        Args:
            file_path: Path to the SEC filing document (text or HTML format)
            reference_resolver: Optional resolver for handling incorporation by reference
                              scenarios where MD&A content is in a separate document

        Returns:
            ExtractionResult containing the extracted MD&A text, tables, and metadata,
            or None if extraction fails at any stage

        Processing Pipeline:
            1. Read file content with automatic encoding detection
            2. Detect if document is HTML or plain text
            3. Preprocess content (convert HTML or apply minimal text cleaning)
            4. Extract filing metadata (CIK, company name, date, form type)
            5. Locate MD&A section using pattern-based boundary detection
            6. Check for and handle incorporation by reference
            7. Validate the extracted section meets quality criteria
            8. Process content preserving table structures
            9. Identify tables and cross-references
            10. Save results to output file
        """
        logger.info(f"Processing file: {file_path}")
        self.error_count = 0  # Reset error counter for this file

        try:
            # ===================================================================
            # STEP 1: Read file content with automatic encoding detection
            # The file handler attempts UTF-8, Latin-1, CP1252, then uses chardet
            # ===================================================================
            content = self.file_handler.read_file(file_path)
            if not content:
                log_error(f"Failed to read file: {file_path}")
                return None

            # Keep reference to original (currently unused but available for debugging)
            original_content = content

            # ===================================================================
            # STEP 2: Detect document format by checking for HTML tags
            # SEC filings can be either plain text or HTML formatted
            # ===================================================================
            is_html = self._is_html_document(content)

            # ===================================================================
            # STEP 3: Preprocess content based on detected format
            # HTML requires careful conversion to preserve table structures
            # Plain text needs minimal processing to maintain formatting
            # ===================================================================
            if is_html:
                logger.info("Detected HTML document - using structure-preserving processing")
                # Convert HTML tables to text tables, remove tags, clean entities
                preprocessed = self._convert_html_preserving_structure(content)
            else:
                logger.info("Detected text document - preserving original structure")
                # Only ensure section headers are on new lines, decode entities
                preprocessed = self._minimal_text_preprocessing(content)

            # ===================================================================
            # STEP 4: Parse filing metadata from document header
            # Extracts CIK, company name, filing date, and form type
            # ===================================================================
            filing = self._parse_filing_metadata(preprocessed, file_path)
            if not filing:
                log_error(f"Failed to parse metadata from: {file_path}")
                return None

            # ===================================================================
            # STEP 5: Locate MD&A section boundaries using regex patterns
            # Uses form-type-specific patterns (10-K vs 10-Q have different Item numbers)
            # ===================================================================
            mdna_bounds = self.section_parser.find_mdna_section(preprocessed, filing.form_type)
            if not mdna_bounds:
                log_error(f"MD&A section not found in: {file_path}")
                return None

            start_pos, end_pos = mdna_bounds  # Character positions in preprocessed text

            # Extract the raw MD&A content from the document
            mdna_text = preprocessed[start_pos:end_pos]

            # ===================================================================
            # STEP 6: Check if MD&A is incorporated by reference
            # Some filings reference MD&A in separate documents (e.g., annual reports)
            # ===================================================================
            incorporation_ref = self.section_parser.check_incorporation_by_reference(
                mdna_text, 0, len(mdna_text)
            )

            # ===================================================================
            # STEP 6a: Handle incorporation by reference
            # When MD&A references another document, try to resolve it
            # ===================================================================
            if incorporation_ref:
                logger.warning(f"MD&A incorporated by reference in {file_path}")
                if reference_resolver:
                    try:
                        # Attempt to fetch MD&A content from the referenced document
                        resolved_mdna = reference_resolver.resolve_reference(incorporation_ref, filing)
                        if resolved_mdna:
                            logger.info(f"Successfully resolved MD&A from {incorporation_ref.document_type}")

                            # Process the resolved content through our standard pipeline
                            final_text = self._process_mdna_content(resolved_mdna, is_html=False)
                            tables = self._identify_tables_in_place(final_text)

                            # Build result with special metadata indicating reference resolution
                            result = ExtractionResult(
                                filing=filing,
                                mdna_text=final_text,
                                tables=tables,
                                cross_references=[],
                                extraction_metadata={
                                    "start_pos": 0,
                                    "end_pos": len(final_text),
                                    "word_count": len(final_text.split()),
                                    "table_count": len(tables),
                                    "cross_ref_count": 0,
                                    "warnings": ["MD&A resolved from referenced document"],
                                    "incorporation_by_reference": {
                                        "document_type": incorporation_ref.document_type,
                                        "caption": incorporation_ref.caption,
                                        "page_reference": incorporation_ref.page_reference,
                                        "resolved": True
                                    }
                                }
                            )
                            self._save_extraction_result(result)
                            return result
                    except Exception as e:
                        logger.error(f"Failed to resolve reference: {e}")

                # If we can't resolve the reference, log error and fail
                log_error(f"MD&A incorporated by reference but could not resolve", file_path)
                return None

            # ===================================================================
            # STEP 7: Validate the extracted section
            # Checks for minimum word count, proper section markers, etc.
            # ===================================================================
            validation = self.section_parser.validate_section(preprocessed, start_pos, end_pos, filing.form_type)
            if not validation["is_valid"]:
                log_error(f"Invalid MD&A section in {file_path}: {validation['warnings']}")
                # Allow some errors before giving up entirely
                if self.error_count > MAX_ERRORS_PER_FILE:
                    return None

            # ===================================================================
            # STEP 8: Process MD&A content with structure preservation
            # Line-by-line processing to maintain table formatting
            # ===================================================================
            final_text = self._process_mdna_content(mdna_text, is_html)

            # ===================================================================
            # STEP 9: Identify tables for metadata (tables remain in-place)
            # Creates metadata about table locations without moving them
            # ===================================================================
            tables = self._identify_tables_in_place(final_text)
            logger.info(f"Identified {len(tables)} tables in MD&A section")

            # ===================================================================
            # STEP 10: Find and resolve cross-references
            # Cross-references like "See Item 7A" or "refer to Note 12"
            # ===================================================================
            cross_refs = self.cross_ref_parser.find_cross_references(final_text)
            if cross_refs:
                # Resolve references by finding the target content in the document
                cross_refs = self.cross_ref_parser.resolve_references(
                    cross_refs,
                    preprocessed,
                    self.normalizer
                )
                logger.info(f"Found {len(cross_refs)} cross-references")
            else:
                cross_refs = []

            # ===================================================================
            # STEP 11: Build and save the final extraction result
            # ===================================================================
            result = ExtractionResult(
                filing=filing,
                mdna_text=final_text,
                tables=tables,
                cross_references=cross_refs,
                extraction_metadata={
                    "start_pos": start_pos,      # Position in original document
                    "end_pos": end_pos,          # End position in original document
                    "word_count": validation["word_count"],
                    "table_count": len(tables),
                    "cross_ref_count": len(cross_refs),
                    "warnings": validation["warnings"]
                }
            )

            # Save to output directory with standardized filename
            self._save_extraction_result(result)
            return result

        except Exception as e:
            # Catch-all for unexpected errors to prevent process crashes
            log_error(f"Error processing {file_path}: {str(e)}")
            return None

    def _process_mdna_content(self, text: str, is_html: bool) -> str:
        """
        Process MD&A content with line-by-line structure preservation.

        This is the core of the in-place preservation strategy. Each line is
        classified as either structured content (tables, columnar data) or
        regular text. Structured lines are preserved exactly, while regular
        text is normalized for consistency.

        The two-pass approach:
        1. First pass: Classify and process each line individually
        2. Second pass: Clean up excessive empty lines while keeping paragraph breaks

        Args:
            text: Raw MD&A content extracted from the document
            is_html: Flag indicating if original document was HTML (unused, kept for future)

        Returns:
            Processed text with preserved table structures and normalized paragraphs
        """
        lines = text.split('\n')
        processed_lines: List[str] = []

        # First pass: Process each line based on its classification
        for i, line in enumerate(lines):
            # Check if this line contains structured data (tables, financial figures)
            if self._is_structured_line(line):
                # PRESERVE EXACTLY: Keep all internal spacing that defines column alignment
                # Only remove trailing whitespace to avoid file bloat
                processed_lines.append(line.rstrip())
            else:
                # NORMALIZE: Regular narrative text gets whitespace normalization
                processed_line = self._process_regular_line(line)
                if processed_line is not None:  # None signals line should be skipped
                    processed_lines.append(processed_line)

        # Second pass: Clean up excessive empty lines while preserving paragraph breaks
        result = self._cleanup_empty_lines(processed_lines)

        return '\n'.join(result)

    def _is_structured_line(self, line: str) -> bool:
        """
        Determine if a line is structured content that should be preserved exactly.

        This is a critical classification function that distinguishes between:
        - Structured content (tables, financial data): Preserve exact spacing
        - Regular narrative text: Apply normalization

        The function uses multiple heuristics to detect structured content:
        1. Table delimiters: Lines of dashes, equals, or underscores
        2. Pipe-delimited content: Markdown-style table formatting
        3. Columnar data: Text segments separated by multiple spaces
        4. Financial numbers: Dollar amounts, percentages in columnar layout
        5. Year headers: Common in comparative financial statements

        Args:
            line: A single line of text to classify

        Returns:
            True if the line should be preserved exactly (structured content)
            False if the line can be normalized (regular text)
        """
        # Empty lines are not considered structured - handle them separately
        if not line.strip():
            return False

        # HEURISTIC 1: Table delimiters (horizontal rules)
        # Matches: "-----", "=====", "_____" with optional leading/trailing spaces
        if re.match(r'^\s*[-=_]{3,}\s*$', line):
            return True

        # HEURISTIC 2: Pipe-delimited tables (Markdown-style)
        # Requires at least 2 pipes to avoid false positives
        if line.count('|') >= 2:
            return True

        # HEURISTIC 3: Columnar structure detection
        # Multiple spaces between text segments indicate column alignment
        if re.search(r'\s{2,}', line):
            # Split on 2+ consecutive spaces (column separator)
            segments = re.split(r'\s{2,}', line)
            # Count non-empty segments (actual column values)
            non_empty = [s for s in segments if s.strip()]
            # 2+ segments strongly suggests columnar data
            if len(non_empty) >= 2:
                return True

        # HEURISTIC 4: Numeric columnar data (financial tables)
        # Uses separate method for detailed number pattern analysis
        if self._has_columnar_numbers(line):
            return True

        # HEURISTIC 5: Year headers (e.g., "2022    2021    2020")
        # Common in comparative financial statements
        if re.search(r'\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b', line):
            return True

        # HEURISTIC 6: Financial line items with dollar amounts
        # Pattern: "Revenue    $1,234,567" with column spacing
        if re.search(r'\$\s*[\d,]+', line) and re.search(r'\s{2,}', line):
            return True

        # HEURISTIC 7: Percentage data in tables
        # Pattern: "Growth Rate    15.2%    12.3%" with spacing
        if '%' in line and re.search(r'\s{2,}', line):
            return True

        return False

    def _has_columnar_numbers(self, line: str) -> bool:
        """
        Check if line contains numbers in columnar format.

        Financial tables typically have numbers aligned in columns with
        significant spacing between them. This method detects such patterns
        to identify table rows containing numerical data.

        Recognized number formats:
        - Currency: $1,234.56 or $1234
        - Negative values: (1,234) or (1,234.56)
        - Percentages: 12.5%
        - Abbreviated numbers: 1.2M, 500K, 2.3B
        - Plain numbers with separators: 1,234,567

        Args:
            line: A single line of text to analyze

        Returns:
            True if line contains multiple numbers with significant spacing
            (suggesting columnar data), False otherwise
        """
        # Comprehensive pattern for financial/numeric data formats
        # Matches: $1,234.56, (1,234), 12.5%, 1.2M, etc.
        number_pattern = re.compile(
            r'(?:\$\s*)?\(?[\d,]+(?:\.\d+)?\)?(?:\s*[%KMB])?'
        )

        # Find all numeric matches in the line
        matches = list(number_pattern.finditer(line))

        # Need at least 2 numbers to indicate columnar data
        if len(matches) >= 2:
            # Check spacing between consecutive numbers
            positions = [m.start() for m in matches]
            for i in range(1, len(positions)):
                # Spacing > 10 characters suggests column alignment, not adjacent numbers
                if positions[i] - positions[i-1] > 10:
                    return True

        return False

    def _process_regular_line(self, line: str) -> Optional[str]:
        """
        Process a regular text line (non-table content).

        Regular narrative text is normalized for consistent formatting while
        preserving meaningful structure like paragraph indentation. This ensures
        the extracted MD&A is readable and suitable for NLP processing.

        Processing rules:
        - Preserves indentation up to 4 spaces (for list items, etc.)
        - Collapses multiple internal spaces to single spaces
        - Returns empty string for blank lines (handled by cleanup)

        Args:
            line: A single line of regular (non-structured) text

        Returns:
            Normalized line with preserved indentation, or empty string for blank lines
        """
        # Calculate original indentation level
        # Cap at 4 spaces to avoid excessive indentation from table-like formatting
        indent = len(line) - len(line.lstrip())
        indent = min(indent, 4)

        # Normalize internal whitespace: multiple spaces become single space
        # This handles inconsistent spacing from OCR or HTML conversion
        # split() + join() is more reliable than regex for this purpose
        cleaned = ' '.join(line.split())

        if cleaned:
            # Reconstruct line with normalized content and preserved indentation
            return ' ' * indent + cleaned
        else:
            # Empty line - will be handled by _cleanup_empty_lines
            return ''

    def _cleanup_empty_lines(self, lines: List[str]) -> List[str]:
        """
        Clean up excessive empty lines while preserving paragraph structure.

        This method balances readability with compactness. Too many blank lines
        make the document hard to read, while too few obscure paragraph breaks
        and table boundaries.

        Rules:
        - Single empty line between regular paragraphs (standard paragraph break)
        - Up to 2 empty lines around tables (visual separation for readability)
        - Remove all trailing empty lines (clean file ending)

        Args:
            lines: List of processed lines (may contain excessive empty lines)

        Returns:
            Cleaned list of lines with appropriate empty line spacing
        """
        result: List[str] = []
        empty_count = 0  # Track consecutive empty lines

        for i, line in enumerate(lines):
            if not line.strip():
                # Empty line encountered
                empty_count += 1

                # Determine context: is this near a table?
                # Tables benefit from extra spacing for visual clarity
                is_before_table = (i + 1 < len(lines) and
                                 self._is_structured_line(lines[i + 1]))
                is_after_table = (result and
                                self._is_structured_line(result[-1]))

                # Allow more empty lines around tables for visual separation
                max_empty = 2 if (is_before_table or is_after_table) else 1

                # Only keep empty line if within the allowed limit
                if empty_count <= max_empty:
                    result.append(line)
            else:
                # Non-empty line: reset counter and add to result
                empty_count = 0
                result.append(line)

        # Remove trailing empty lines for clean file ending
        while result and not result[-1].strip():
            result.pop()

        return result

    def _identify_tables_in_place(self, text: str) -> List[Table]:
        """
        Identify tables in the text for metadata purposes.

        This method scans the processed text to locate and catalog embedded
        tables WITHOUT modifying their positions. The tables remain in-place
        in the text; this method only creates metadata about their locations.

        Table metadata is useful for:
        - Understanding document structure
        - Enabling table-specific analysis
        - Providing extraction statistics
        - Supporting downstream table parsing if needed

        Args:
            text: Processed MD&A text with tables in-place

        Returns:
            List of Table objects containing metadata about each identified table
        """
        tables: List[Table] = []
        lines = text.split('\n')

        # Scan through all lines looking for table start positions
        i = 0
        while i < len(lines):
            # Check if this line begins a table
            if self._is_table_start(lines, i):
                # Extract metadata about the table (boundaries, title, etc.)
                table = self._extract_table_metadata(lines, i)
                if table:
                    tables.append(table)
                    # Skip to end of table to avoid re-detecting interior lines
                    # Use getattr for safety in case end_line attribute is missing
                    i = getattr(table, 'end_line', i) + 1
                else:
                    i += 1
            else:
                i += 1

        return tables

    def _is_table_start(self, lines: List[str], index: int) -> bool:
        """
        Check if the current line starts a table.

        Table detection is based on finding consistent structured patterns.
        A single structured line might be incidental, but multiple consecutive
        structured lines strongly indicate a table.

        Detection criteria:
        1. Horizontal delimiter (----, ====) preceded by header content
        2. Two or more consecutive lines with structured formatting
        3. Clear columnar patterns with consistent spacing

        Args:
            lines: List of all text lines in the document
            index: Current line index to check

        Returns:
            True if this line appears to start a table, False otherwise
        """
        # Bounds check
        if index >= len(lines):
            return False

        current = lines[index]

        # Detection method 1: Delimiter line (likely header separator)
        # Table headers often have a separator line immediately below
        if re.match(r'^\s*[-=_]{3,}\s*$', current) and index > 0:
            return True

        # Detection method 2: Consecutive structured lines
        if self._is_structured_line(current):
            # Look ahead to see if more structured lines follow
            structured_count = 1
            for j in range(index + 1, min(index + 3, len(lines))):
                if self._is_structured_line(lines[j]):
                    structured_count += 1

            # Two or more consecutive structured lines = table
            return structured_count >= 2

        return False

    def _extract_table_metadata(self, lines: List[str], start_index: int) -> Optional[Table]:
        """
        Extract metadata about a table without modifying its content.

        This method determines the boundaries of a table and captures metadata
        about it. The table content remains in-place in the original text;
        this method only creates a metadata record.

        Table boundary detection:
        1. Start from the identified table start line
        2. Continue while lines are structured (columnar data)
        3. Include continuation lines (totals, subtotals, etc.)
        4. Require minimum 2 lines to be considered a valid table

        Args:
            lines: List of all text lines in the document
            start_index: Line index where the table begins

        Returns:
            Table object with metadata, or None if table is too small
        """
        # Find table end by continuing while lines are structured
        end_index = start_index

        # Continue through all structured lines
        while end_index < len(lines) and self._is_structured_line(lines[end_index]):
            end_index += 1

        # Include continuation lines (totals, subtotals often follow tables)
        while end_index < len(lines) and self._is_table_continuation(lines[end_index]):
            end_index += 1

        # Tables must have at least 2 lines (header + data or multiple data rows)
        if end_index - start_index < 2:
            return None

        # Extract the table content lines - PRESERVE ORIGINAL FORMATTING
        # This is critical for maintaining column alignment
        table_lines = lines[start_index:end_index]

        # Look for a table title in the preceding lines
        # Titles are typically within 3 lines before the table starts
        title = None
        for i in range(1, min(4, start_index + 1)):
            candidate = lines[start_index - i].strip()
            # Title should be non-empty and not a structured line
            if candidate and not self._is_structured_line(lines[start_index - i]):
                title = candidate
                break

        # Store table with original formatting preserved
        original_text = '\n'.join(table_lines)

        # Create table metadata object
        # Note: 'content' field uses List[List[str]] for interface compatibility,
        # but the actual preserved formatting is in 'original_text'
        parsed_rows = []
        for line in table_lines:
            # Simple row representation for compatibility
            # The real preserved formatting is in original_text
            parsed_rows.append([line])

        return Table(
            content=parsed_rows,           # Compatibility: List[List[str]] format
            start_pos=0,                   # Character position (calculated if needed)
            end_pos=0,                     # End character position
            start_line=start_index,        # Line number where table starts
            end_line=end_index - 1,        # Line number where table ends
            title=title,                   # Detected table title (may be None)
            confidence=0.9,                # High confidence for detected tables
            table_type='preserved',        # Indicates in-place preservation mode
            original_text=original_text    # ACTUAL preserved formatting here
        )

    def _is_table_continuation(self, line: str) -> bool:
        """
        Check if line is a table continuation (totals, subtotals, etc.).

        Financial tables often end with summary rows that may not strictly
        match the columnar structure of the data rows. These continuation
        lines contain keywords like "Total" or "Net" and should be included
        with the table.

        Args:
            line: A single line following a structured table section

        Returns:
            True if line appears to be a table summary/continuation row
        """
        # Empty lines are not continuations
        if not line.strip():
            return False

        # Keywords commonly found in table summary rows
        continuation_keywords = ['total', 'subtotal', 'net', 'gross', 'sum']
        line_lower = line.lower()

        # Check for presence of summary keywords
        has_keyword = any(kw in line_lower for kw in continuation_keywords)

        # Continuation lines should also have numeric content or columnar structure
        # This prevents false positives on narrative text mentioning these words
        has_numbers = bool(re.search(r'\d', line))
        has_structure = re.search(r'\s{3,}', line) is not None  # 3+ spaces = columns

        # Must have keyword AND (numbers OR structure)
        return has_keyword and (has_numbers or has_structure)

    def _convert_html_preserving_structure(self, html_content: str) -> str:
        """
        Convert HTML to text while preserving table and document structure.

        SEC filings are often submitted as HTML documents with complex formatting.
        This method converts HTML to plain text while maintaining the visual
        structure that indicates tables and section boundaries.

        Processing pipeline:
        1. Decode HTML entities (first pass)
        2. Convert <table> elements to formatted text tables
        3. Remove <script> and <style> elements entirely
        4. Convert block elements to line breaks
        5. Strip remaining HTML tags
        6. Clean up residual HTML entities
        7. Normalize whitespace conservatively

        Key principles:
        - HTML tables become text tables with column spacing
        - Block elements (<div>, <p>, etc.) create line breaks
        - Inline elements are stripped, content preserved
        - Tabs are preserved (may indicate table structure)

        Args:
            html_content: Raw HTML content from SEC filing

        Returns:
            Plain text with preserved structural formatting
        """
        # First pass: decode HTML entities (&amp; -> &, &lt; -> <, etc.)
        content = html.unescape(html_content)

        # STEP 1: Convert HTML tables to structured text BEFORE stripping tags
        # This preserves table structure that would otherwise be lost
        content = self._convert_html_tables_to_text(content)

        # STEP 2: Remove script and style blocks completely
        # These contain code/CSS that would pollute the text output
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

        # STEP 3: Convert block elements to line breaks
        # These HTML elements typically indicate visual separation
        block_elements = ['div', 'p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
        for elem in block_elements:
            # Opening tag -> newline (creates line break before content)
            content = re.sub(f'<{elem}[^>]*>', '\n', content, flags=re.IGNORECASE)
            # Closing tag -> removed (content already separated)
            content = re.sub(f'</{elem}>', '', content, flags=re.IGNORECASE)

        # STEP 4: Remove all remaining HTML tags
        # At this point, only inline/formatting tags should remain
        content = re.sub(r'<[^>]+>', '', content)

        # STEP 5: Clean up any remaining HTML entities not caught by unescape
        content = re.sub(r'&nbsp;', ' ', content, flags=re.IGNORECASE)   # Non-breaking space
        content = re.sub(r'&amp;', '&', content, flags=re.IGNORECASE)    # Ampersand
        content = re.sub(r'&lt;', '<', content, flags=re.IGNORECASE)     # Less than
        content = re.sub(r'&gt;', '>', content, flags=re.IGNORECASE)     # Greater than
        content = re.sub(r'&quot;', '"', content, flags=re.IGNORECASE)   # Quote
        content = re.sub(r'&#\d+;', ' ', content)                        # Numeric entities
        content = re.sub(r'&\w+;', ' ', content)                         # Named entities

        # STEP 6: Conservative whitespace cleanup
        # Multiple spaces -> single space (within lines)
        content = re.sub(r' {2,}', ' ', content)
        # Limit consecutive blank lines (3+ -> 2)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        # NOTE: Tabs are preserved as they may indicate table structure

        return content

    def _convert_html_tables_to_text(self, html_str: str) -> str:
        """
        Convert HTML tables to properly formatted text tables.

        This method parses HTML <table> elements and converts them to
        ASCII text tables with proper column alignment. This is critical
        for preserving the structure of financial data tables.

        Process for each table:
        1. Extract all <tr> (row) elements
        2. Extract <td> and <th> (cell) elements from each row
        3. Clean cell content (remove tags, decode entities)
        4. Calculate column widths and format as aligned text
        5. Replace HTML table with formatted text table

        Args:
            html_str: HTML content containing <table> elements

        Returns:
            HTML with <table> elements replaced by text table equivalents
        """
        def process_table(match):
            """
            Inner function to process a single HTML table match.
            Converts the matched HTML table to a formatted text table.
            """
            table_html = match.group(0)

            # Parse all rows from the table
            rows: List[List[str]] = []
            row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)

            for row_match in row_pattern.finditer(table_html):
                cells: List[str] = []
                row_content = row_match.group(1)

                # Extract cells: both <td> (data) and <th> (header) elements
                cell_pattern = re.compile(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', re.IGNORECASE | re.DOTALL)

                for cell_match in cell_pattern.finditer(row_content):
                    cell_text = cell_match.group(1)
                    # Clean cell content:
                    # 1. Remove any nested HTML tags
                    cell_text = re.sub(r'<[^>]+>', '', cell_text)
                    # 2. Decode HTML entities
                    cell_text = html.unescape(cell_text)
                    # 3. Normalize whitespace
                    cell_text = ' '.join(cell_text.split())
                    cells.append(cell_text)

                # Only add row if it has cells
                if cells:
                    rows.append(cells)

            # Format the parsed rows as an aligned text table
            if rows:
                # Add newlines around table for visual separation
                return '\n' + self._format_text_table(rows) + '\n'
            else:
                return ''  # Empty table -> empty string

        # Find and replace all HTML tables in the document
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
        return table_pattern.sub(process_table, html_str)

    def _format_text_table(self, rows: List[List[str]]) -> str:
        """
        Format table rows as aligned text with improved human readability.

        This method converts a 2D array of cell values into a formatted ASCII
        table with proper column alignment. The output is optimized for both
        human readability and downstream NLP processing.

        Formatting features:
        - Column widths automatically calculated to fit content
        - Numeric values are right-aligned (standard accounting format)
        - Text values are left-aligned
        - Header separator line after first row
        - 3-space gaps between columns

        Args:
            rows: 2D list where each inner list represents a table row

        Returns:
            Formatted text table as a string with newlines between rows
        """
        if not rows:
            return ""

        # Calculate column widths based on content
        num_cols = max(len(row) for row in rows)
        col_widths: List[int] = []

        # Find the maximum width needed for each column
        for col_idx in range(num_cols):
            max_width = 0
            for row in rows:
                if col_idx < len(row):
                    max_width = max(max_width, len(row[col_idx]))
            # Don't cap width - preserve full content for accurate representation
            col_widths.append(max_width)

        # Format each row with proper alignment and spacing
        formatted_rows: List[str] = []

        for row_idx, row in enumerate(rows):
            formatted_cells: List[str] = []

            for col_idx in range(num_cols):
                # Get cell value or empty string if row is shorter
                if col_idx < len(row):
                    cell = row[col_idx]  # Don't truncate - preserve all content
                else:
                    cell = ''

                # Alignment logic:
                # - Numbers, currency, percentages -> right-align (accounting standard)
                # - Text content -> left-align
                if cell and re.match(r'^[\d,.\-$%()]+$', cell.strip()):
                    formatted_cells.append(cell.rjust(col_widths[col_idx]))
                else:
                    formatted_cells.append(cell.ljust(col_widths[col_idx]))

            # Join cells with 3-space separator (creates visual column gaps)
            formatted_rows.append('   '.join(formatted_cells))

            # Add header separator after the first row (if table has multiple rows)
            # This creates a visual distinction between headers and data
            if row_idx == 0 and len(rows) > 1:
                separator_parts = []
                for col_idx in range(num_cols):
                    separator_parts.append('-' * col_widths[col_idx])
                formatted_rows.append('   '.join(separator_parts))

        return '\n'.join(formatted_rows)

    def _minimal_text_preprocessing(self, content: str) -> str:
        """
        Minimal preprocessing for text documents.

        Plain text SEC filings have already been formatted for reading, so
        we apply minimal changes to avoid disrupting existing table structures.
        The main goal is to ensure section headers are properly positioned
        for reliable section detection.

        Processing steps:
        1. Decode any HTML entities that may be present
        2. Ensure ITEM headers start on new lines (for section parsing)
        3. Ensure PART headers start on new lines

        Note: Lookahead/lookbehind in regex prevents affecting table content
        that might contain numbers adjacent to these patterns.

        Args:
            content: Raw text content from SEC filing

        Returns:
            Minimally processed text with proper section header positioning
        """
        # Decode HTML entities (some "text" files contain entity-encoded characters)
        content = html.unescape(content)

        # Ensure ITEM headers start on new lines for reliable section detection
        # Pattern: "ITEM 7." or "ITEM 7A." - avoid matching numbers in tables
        # Lookahead/lookbehind prevents "123ITEM" or "ITEM123" matches
        content = re.sub(r'(?<![0-9])\s*(ITEM\s*\d+[A-Z]?\.?)(?![0-9])', r'\n\1', content, flags=re.IGNORECASE)

        # Ensure PART headers start on new lines (PART I, PART II, etc.)
        content = re.sub(r'(?<![0-9])\s*(PART\s*[IVX]+\.?)(?![0-9])', r'\n\1', content, flags=re.IGNORECASE)

        return content

    @staticmethod
    def _is_html_document(content: str) -> bool:
        """
        Detect if the document is HTML-formatted.

        SEC filings can be submitted in either plain text or HTML format.
        This method samples the beginning of the document to check for
        common HTML tags that indicate HTML formatting.

        Args:
            content: Document content to analyze

        Returns:
            True if document appears to be HTML, False for plain text
        """
        # Sample only first 15KB to avoid scanning entire large documents
        sample = content[:15000].lower()

        # Common HTML indicators found in SEC filings
        html_indicators = [
            '<html', '<body', '<div', '<table', '<tr', '<td',
            '<!doctype', '<meta', '<style', '<font'
        ]

        # Document is HTML if ANY indicator is found
        return any(indicator in sample for indicator in html_indicators)

    def _parse_filing_metadata(self, content: str, file_path: Path) -> Optional[Filing]:
        """
        Parse filing metadata from document content.

        SEC filings contain standardized header information that identifies
        the filer and the filing. This method extracts key metadata fields
        needed for organizing and processing the extracted MD&A.

        Metadata fields extracted:
        - CIK (Central Index Key): Unique SEC identifier for the filer
        - Company Name: Legal name of the filing company
        - Filing Date: Date the document was filed with SEC
        - Form Type: Type of filing (10-K, 10-K/A, 10-Q, 10-Q/A)

        Fallback strategies:
        - If metadata not found in document, extract from filename
        - If date not found, use file modification timestamp
        - If company name not found, use "Unknown Company"

        Args:
            content: Preprocessed document content
            file_path: Path to the source file (for fallback metadata)

        Returns:
            Filing object with extracted metadata, or None on error
        """
        try:
            # =================================================================
            # Extract CIK (Central Index Key)
            # CIK is a unique 10-digit identifier assigned by the SEC
            # =================================================================
            cik = None
            cik_patterns = [
                # Pattern 1: Explicit CIK label
                r"(?:CENTRAL\s*INDEX\s*KEY|CIK)[\s:]*(\d{4,10})",
                # Pattern 2: CIK in company info block
                r"(?:COMPANY\s*CONFORMED\s*NAME).*?(?:CENTRAL\s*INDEX\s*KEY|CIK)[\s:]*(\d{4,10})",
                # Pattern 3: Standalone 10-digit number (exact CIK format)
                r"^\s*(\d{10})\s*$",
            ]

            # Try each pattern until CIK is found
            for pattern_str in cik_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                match = pattern.search(content[:10000])  # Check header area only
                if match:
                    cik = match.group(1).zfill(10)  # Pad to 10 digits
                    break

            # Fallback: Extract CIK from filename
            if not cik:
                cik_from_name = re.search(r"(\d{4,10})", file_path.name)
                cik = cik_from_name.group(1).zfill(10) if cik_from_name else "0000000000"

            # =================================================================
            # Extract Company Name
            # Uses TextNormalizer's specialized extraction logic
            # =================================================================
            company_name = self.normalizer.extract_company_name(content)
            if not company_name:
                company_name = "Unknown Company"

            # =================================================================
            # Extract Filing Date
            # Try multiple date formats common in SEC filings
            # =================================================================
            filing_date = None
            date_patterns = [
                # Pattern 1: "Filed as of date" or "Filing Date"
                r"(?:FILED\s*AS\s*OF\s*DATE|Filing\s*Date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                # Pattern 2: 8-digit date near filing type
                r"(?:DATE\s*OF\s*FILING|CONFORMED\s*SUBMISSION\s*TYPE).*?(\d{8})",
                # Pattern 3: Period of report date
                r"(?:PERIOD\s*OF\s*REPORT)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            ]

            # Try each pattern until date is found
            for pattern_str in date_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                match = pattern.search(content[:10000])
                if match:
                    date_str = match.group(1)
                    filing_date = self._parse_date(date_str)
                    if filing_date:
                        break

            # Fallback: Extract date from filename or use file modification time
            if not filing_date:
                date_from_name = re.search(r"(\d{8})", file_path.name)
                if date_from_name:
                    try:
                        filing_date = datetime.strptime(date_from_name.group(1), "%Y%m%d")
                    except ValueError:
                        filing_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                else:
                    filing_date = datetime.fromtimestamp(file_path.stat().st_mtime)

            # =================================================================
            # Extract Form Type (10-K, 10-K/A, 10-Q, 10-Q/A)
            # Uses pre-compiled patterns from config
            # =================================================================
            form_type = "10-K"  # Default assumption

            for pattern in self.patterns["form_type"]:
                match = pattern.search(content[:10000])
                if match:
                    form_type_raw = match.group(1).upper()
                    # Determine if this is a quarterly (10-Q) or annual (10-K) filing
                    # Also check for amendments (/A suffix)
                    if '10-Q' in form_type_raw:
                        if 'A' in form_type_raw or '/A' in form_type_raw:
                            form_type = "10-Q/A"  # Quarterly amendment
                        else:
                            form_type = "10-Q"    # Quarterly report
                    elif '10-K' in form_type_raw:
                        if 'A' in form_type_raw or '/A' in form_type_raw:
                            form_type = "10-K/A"  # Annual amendment
                        else:
                            form_type = "10-K"    # Annual report
                    break

            # =================================================================
            # Build and return Filing object
            # =================================================================
            return Filing(
                cik=cik,
                company_name=company_name,
                filing_date=filing_date,
                form_type=form_type,
                file_path=file_path
            )

        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            return None

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """
        Parse date string to datetime object.

        SEC filings use various date formats. This method tries multiple
        common formats until one succeeds.

        Supported formats:
        - MM/DD/YYYY, MM-DD-YYYY (US format with 4-digit year)
        - MM/DD/YY, MM-DD-YY (US format with 2-digit year)
        - YYYY-MM-DD, YYYY/MM/DD (ISO format)
        - YYYYMMDD (Compact 8-digit format)
        - Month DD, YYYY (Written format: "January 15, 2024")

        Args:
            date_str: Date string in any supported format

        Returns:
            datetime object if parsing succeeds, None otherwise
        """
        # List of date formats to try, ordered by prevalence in SEC filings
        formats = [
            "%m/%d/%Y", "%m-%d-%Y",      # US format with 4-digit year
            "%m/%d/%y", "%m-%d-%y",      # US format with 2-digit year
            "%Y-%m-%d", "%Y/%m/%d",      # ISO format
            "%Y%m%d",                     # Compact 8-digit format
            "%B %d, %Y", "%b %d, %Y"     # Written format (full/abbreviated month)
        ]

        # Try each format until one parses successfully
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        # None of the formats worked
        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _save_extraction_result(self, result: ExtractionResult) -> None:
        """
        Save extraction results to output files.

        This method creates a formatted output file containing the extracted
        MD&A text with a metadata header. The filename follows a standardized
        pattern for easy identification and sorting.

        Output format:
        - Header block with metadata (CIK, company, date, form type)
        - Extracted MD&A text (with tables in-place)
        - Optional cross-references section (if references were resolved)

        Filename pattern:
        (CIK)_(CompanyName)_(FilingDate)_(FormType).txt

        Args:
            result: ExtractionResult containing MD&A text and metadata
        """
        # Build filename components from metadata
        cik = result.filing.cik
        company_name = self.normalizer.sanitize_filename(result.filing.company_name)
        filing_date = result.filing.filing_date.strftime("%Y-%m-%d")
        form_type = result.filing.form_type.replace('/', '-')  # Replace / with - for filename

        # Construct output filename and path
        filename = f"({cik})_({company_name})_({filing_date})_({form_type}).txt"
        output_path = self.output_dir / filename

        # Build output content with metadata header
        final_content: List[str] = []

        # ===== METADATA HEADER =====
        final_content.append("=" * 80)
        final_content.append(f"CIK: {cik}")
        final_content.append(f"Company Name: {result.filing.company_name}")
        final_content.append(f"Filing Date: {filing_date}")
        final_content.append(f"Form Type: {form_type}")
        final_content.append(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        final_content.append("=" * 80)
        final_content.append("")  # Blank line after header

        # ===== MD&A TEXT =====
        # Tables are already embedded in the text at their original positions
        final_content.append(result.mdna_text)

        # ===== CROSS-REFERENCES SECTION (optional) =====
        # Only include if there are successfully resolved references
        if result.cross_references:
            resolved_refs = [ref for ref in result.cross_references if ref.resolved]
            if resolved_refs:
                final_content.append("")
                final_content.append("-" * 80)
                final_content.append("CROSS-REFERENCES")
                final_content.append("-" * 80)
                final_content.append("")

                # Add each resolved reference with its text
                for ref in resolved_refs:
                    if ref.resolution_text:
                        final_content.append(f"[{ref.reference_text}]:")
                        final_content.append(ref.resolution_text)
                        final_content.append("")

        # Write to file
        output_text = '\n'.join(final_content)
        self.file_handler.write_file(output_path, output_text)
        logger.info(f"Saved MD&A to: {output_path}")

    def process_directory(self, input_dir: Path, cik_filter=None) -> Dict[str, Any]:
        """
        Process all files in a directory.

        Batch processing method for extracting MD&A from multiple filings.
        Supports optional CIK filtering to process only specific companies
        or years.

        Args:
            input_dir: Directory containing SEC filing text files
            cik_filter: Optional CIKFilter instance for selective processing

        Returns:
            Dictionary containing processing statistics:
            - total_files: Number of files found
            - successful: Number of successful extractions
            - failed: Number of failed extractions
            - filtered_out: Number of files skipped by CIK filter
            - errors: List of file paths that failed
        """
        # Initialize statistics tracking
        stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "filtered_out": 0,
            "errors": []
        }

        # Find all text files (both .txt and .TXT extensions)
        text_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.TXT"))
        stats["total_files"] = len(text_files)

        logger.info(f"Found {len(text_files)} text files to process")

        # Process each file
        for file_path in text_files:
            # Apply CIK filter if provided
            if cik_filter and cik_filter.has_cik_filters():
                # Quick metadata parse from filename for filtering decision
                cik, year, form_type = self._parse_file_metadata_simple(file_path)

                # Check if this filing should be processed
                if not cik_filter.should_process_filing(cik, form_type, year):
                    stats["filtered_out"] += 1
                    logger.info(f"Filtered out: {file_path.name}")
                    continue

            # Extract MD&A from the filing
            result = self.extract_from_file(file_path)
            if result:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(str(file_path))

        return stats

    def _parse_file_metadata_simple(self, file_path: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Parse basic metadata for CIK filtering from filename.

        Quick extraction of metadata from the filename without reading
        the file content. Used for fast filtering decisions before
        full extraction.

        Args:
            file_path: Path to the filing file

        Returns:
            Tuple of (CIK, year, form_type), any of which may be None
        """
        filename = file_path.name

        # Extract CIK: 4-10 digit number, padded to 10 digits
        cik_match = re.search(r'(\d{4,10})', filename)
        cik = cik_match.group(1).zfill(10) if cik_match else None

        # Extract year: 1994-2029 range (covers historical and future filings)
        year_match = re.search(r'(199[4-9]|20[0-2][0-9])', filename)
        year = int(year_match.group(1)) if year_match else None

        # Extract form type from filename
        form_type = None
        filename_upper = filename.upper()
        if '10-Q' in filename_upper:
            form_type = "10-Q"
        elif '10-K' in filename_upper:
            form_type = "10-K"

        return cik, year, form_type