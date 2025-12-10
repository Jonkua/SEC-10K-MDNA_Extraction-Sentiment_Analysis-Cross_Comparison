"""Enhanced MD&A extractor with true in-place table preservation."""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
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

logger = get_logger(__name__)


class MDNAExtractor:
    """Main class for extracting MD&A sections with true in-place table preservation."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.file_handler = FileHandler()
        self.section_parser = SectionParser()
        self.table_parser = TableParser()
        self.cross_ref_parser = CrossReferenceParser()
        self.normalizer = TextNormalizer()
        self.patterns = compile_patterns()
        self.error_count = 0

    def extract_from_file(self, file_path: Path, reference_resolver=None) -> Optional[ExtractionResult]:
        """
        Extract MD&A from a single filing file with in-place table preservation.
        """
        logger.info(f"Processing file: {file_path}")
        self.error_count = 0

        try:
            # Read file content
            content = self.file_handler.read_file(file_path)
            if not content:
                log_error(f"Failed to read file: {file_path}")
                return None

            # Store original content
            original_content = content

            # Detect document type
            is_html = self._is_html_document(content)

            if is_html:
                logger.info("Detected HTML document - using structure-preserving processing")
                # Convert HTML to structured text while preserving tables
                preprocessed = self._convert_html_preserving_structure(content)
            else:
                logger.info("Detected text document - preserving original structure")
                preprocessed = self._minimal_text_preprocessing(content)

            # Parse filing metadata
            filing = self._parse_filing_metadata(preprocessed, file_path)
            if not filing:
                log_error(f"Failed to parse metadata from: {file_path}")
                return None

            # Find MD&A section boundaries
            mdna_bounds = self.section_parser.find_mdna_section(preprocessed, filing.form_type)
            if not mdna_bounds:
                log_error(f"MD&A section not found in: {file_path}")
                return None

            start_pos, end_pos = mdna_bounds

            # Extract MD&A content with structure preserved
            mdna_text = preprocessed[start_pos:end_pos]

            # Check for incorporation by reference
            incorporation_ref = self.section_parser.check_incorporation_by_reference(
                mdna_text, 0, len(mdna_text)
            )

            if incorporation_ref:
                logger.warning(f"MD&A incorporated by reference in {file_path}")
                if reference_resolver:
                    try:
                        resolved_mdna = reference_resolver.resolve_reference(incorporation_ref, filing)
                        if resolved_mdna:
                            logger.info(f"Successfully resolved MD&A from {incorporation_ref.document_type}")
                            # Process resolved content
                            final_text = self._process_mdna_content(resolved_mdna, is_html=False)
                            tables = self._identify_tables_in_place(final_text)

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

                log_error(f"MD&A incorporated by reference but could not resolve", file_path)
                return None

            # Validate section
            validation = self.section_parser.validate_section(preprocessed, start_pos, end_pos, filing.form_type)
            if not validation["is_valid"]:
                log_error(f"Invalid MD&A section in {file_path}: {validation['warnings']}")
                if self.error_count > MAX_ERRORS_PER_FILE:
                    return None

            # Process MD&A content with in-place structure preservation
            final_text = self._process_mdna_content(mdna_text, is_html)

            # Identify tables for metadata (but they stay in place)
            tables = self._identify_tables_in_place(final_text)
            logger.info(f"Identified {len(tables)} tables in MD&A section")

            # Find and resolve cross-references
            cross_refs = self.cross_ref_parser.find_cross_references(final_text)
            if cross_refs:
                cross_refs = self.cross_ref_parser.resolve_references(
                    cross_refs,
                    preprocessed,
                    self.normalizer
                )
                logger.info(f"Found {len(cross_refs)} cross-references")
            else:
                cross_refs = []

            # Build result
            result = ExtractionResult(
                filing=filing,
                mdna_text=final_text,
                tables=tables,
                cross_references=cross_refs,
                extraction_metadata={
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "word_count": validation["word_count"],
                    "table_count": len(tables),
                    "cross_ref_count": len(cross_refs),
                    "warnings": validation["warnings"]
                }
            )

            self._save_extraction_result(result)
            return result

        except Exception as e:
            log_error(f"Error processing {file_path}: {str(e)}")
            return None

    def _process_mdna_content(self, text: str, is_html: bool) -> str:
        """
        Process MD&A content with line-by-line structure preservation.

        This is the core of the in-place preservation strategy.
        Each line is classified as either structured or regular text.
        """
        lines = text.split('\n')
        processed_lines: List[str] = []

        for i, line in enumerate(lines):
            # Check if this is a structured line (table/columnar data)
            if self._is_structured_line(line):
                # Preserve exactly as-is (keep all internal spacing)
                # Only remove trailing whitespace
                processed_lines.append(line.rstrip())
            else:
                # Regular text line - normalize but preserve paragraph structure
                processed_line = self._process_regular_line(line)
                if processed_line is not None:  # None means skip this line
                    processed_lines.append(processed_line)

        # Clean up excessive empty lines while preserving paragraph breaks
        result = self._cleanup_empty_lines(processed_lines)

        return '\n'.join(result)

    def _is_structured_line(self, line: str) -> bool:
        """
        Determine if a line is structured content that should be preserved exactly.

        Returns True for:
        - Table delimiters (---, ===, ___)
        - Columnar data (multiple segments separated by 2+ spaces)
        - Pipe-delimited content
        - Lines with columnar numeric data
        - Lines that look like financial data with proper spacing
        """
        # Empty lines are not structured
        if not line.strip():
            return False

        # Check for table delimiters (can be very long)
        if re.match(r'^\s*[-=_]{3,}\s*$', line):
            return True

        # Check for pipe delimiters (at least 2 pipes)
        if line.count('|') >= 2:
            return True

        # Check for columnar structure (2+ consecutive spaces suggesting columns)
        # Lower threshold to catch more tables
        if re.search(r'\s{2,}', line):
            # Split on 2+ spaces
            segments = re.split(r'\s{2,}', line)
            # Filter out empty segments
            non_empty = [s for s in segments if s.strip()]
            # If we have at least 2 non-empty segments, it's likely a table
            if len(non_empty) >= 2:
                return True

        # Check for numeric columnar data
        if self._has_columnar_numbers(line):
            return True

        # Check for year headers common in financial tables
        if re.search(r'\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b', line):
            return True

        # Check for financial statement line items with amounts
        if re.search(r'\$\s*[\d,]+', line) and re.search(r'\s{2,}', line):
            return True

        # Check for percentage signs with spacing (common in tables)
        if '%' in line and re.search(r'\s{2,}', line):
            return True

        return False

    def _has_columnar_numbers(self, line: str) -> bool:
        """
        Check if line contains numbers in columnar format.

        Looks for patterns like:
        - Financial numbers with currency symbols
        - Percentages
        - Numbers with thousand separators
        - Numbers in parentheses (negative values)
        """
        # Pattern for financial/numeric data
        number_pattern = re.compile(
            r'(?:\$\s*)?\(?[\d,]+(?:\.\d+)?\)?(?:\s*[%KMB])?'
        )

        matches = list(number_pattern.finditer(line))

        if len(matches) >= 2:
            # Check if numbers are spaced out (suggesting columns)
            positions = [m.start() for m in matches]
            for i in range(1, len(positions)):
                if positions[i] - positions[i-1] > 10:  # Significant spacing
                    return True

        return False

    def _process_regular_line(self, line: str) -> Optional[str]:
        """
        Process a regular text line (non-table content).

        - Preserves indentation (up to 4 spaces)
        - Normalizes internal whitespace
        - Returns None for lines that should be removed
        """
        # Get indentation level (max 4 spaces)
        indent = len(line) - len(line.lstrip())
        indent = min(indent, 4)

        # Clean the line content - but preserve full length
        # Don't truncate the line
        cleaned = ' '.join(line.split())

        if cleaned:
            # Return with preserved indentation
            return ' ' * indent + cleaned
        else:
            # Empty line - will be handled by cleanup
            return ''

    def _cleanup_empty_lines(self, lines: List[str]) -> List[str]:
        """
        Clean up excessive empty lines while preserving paragraph structure.

        Rules:
        - Keep single empty lines between paragraphs
        - Allow up to 2 consecutive empty lines around tables
        - Remove trailing empty lines
        """
        result: List[str] = []
        empty_count = 0

        for i, line in enumerate(lines):
            if not line.strip():
                empty_count += 1
                # Check if next line is a table/structured content
                is_before_table = (i + 1 < len(lines) and
                                 self._is_structured_line(lines[i + 1]))
                # Check if previous line was a table
                is_after_table = (result and
                                self._is_structured_line(result[-1]))

                # Allow 2 empty lines around tables, 1 elsewhere
                max_empty = 2 if (is_before_table or is_after_table) else 1

                if empty_count <= max_empty:
                    result.append(line)
            else:
                empty_count = 0
                result.append(line)

        # Remove trailing empty lines
        while result and not result[-1].strip():
            result.pop()

        return result

    def _identify_tables_in_place(self, text: str) -> List[Table]:
        """
        Identify tables in the text for metadata purposes.

        Tables remain in their original positions - this just creates
        metadata about where they are located.
        """
        tables: List[Table] = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            # Check if this starts a table
            if self._is_table_start(lines, i):
                table = self._extract_table_metadata(lines, i)
                if table:
                    tables.append(table)
                    # Use getattr with default to handle missing end_line attribute
                    i = getattr(table, 'end_line', i) + 1
                else:
                    i += 1
            else:
                i += 1

        return tables

    def _is_table_start(self, lines: List[str], index: int) -> bool:
        """
        Check if the current line starts a table.

        Looks for:
        - Header followed by delimiter
        - Multiple consecutive structured lines
        - Clear columnar pattern
        """
        if index >= len(lines):
            return False

        current = lines[index]

        # Check if it's a delimiter preceded by a header
        if re.match(r'^\s*[-=_]{3,}\s*$', current) and index > 0:
            return True

        # Check if it starts a series of structured lines
        if self._is_structured_line(current):
            # Look ahead for more structured lines
            structured_count = 1
            for j in range(index + 1, min(index + 3, len(lines))):
                if self._is_structured_line(lines[j]):
                    structured_count += 1

            return structured_count >= 2

        return False

    def _extract_table_metadata(self, lines: List[str], start_index: int) -> Optional[Table]:
        """
        Extract metadata about a table without modifying its content.
        """
        # Find table boundaries
        end_index = start_index

        # Continue while lines are structured
        while end_index < len(lines) and self._is_structured_line(lines[end_index]):
            end_index += 1

        # Include any continuation lines (like totals)
        while end_index < len(lines) and self._is_table_continuation(lines[end_index]):
            end_index += 1

        # Require minimum size
        if end_index - start_index < 2:
            return None

        # Extract table lines - PRESERVE ORIGINAL FORMATTING
        table_lines = lines[start_index:end_index]

        # Find potential title (look back up to 3 lines)
        title = None
        for i in range(1, min(4, start_index + 1)):
            candidate = lines[start_index - i].strip()
            if candidate and not self._is_structured_line(lines[start_index - i]):
                title = candidate
                break

        # Store table with original formatting preserved
        original_text = '\n'.join(table_lines)

        # Create table metadata
        # We need to provide content as List[List[str]] for compatibility
        # but the actual formatted content is in original_text
        parsed_rows = []
        for line in table_lines:
            # Create a simple parsed version for the content field
            # The real preserved formatting is in original_text
            parsed_rows.append([line])

        return Table(
            content=parsed_rows,  # Compatibility with List[List[str]] type
            start_pos=0,  # Will be calculated if needed
            end_pos=0,
            start_line=start_index,
            end_line=end_index - 1,
            title=title,
            confidence=0.9,
            table_type='preserved',
            original_text=original_text  # This has the real preserved formatting
        )

    def _is_table_continuation(self, line: str) -> bool:
        """Check if line is a table continuation (totals, subtotals, etc.)."""
        if not line.strip():
            return False

        continuation_keywords = ['total', 'subtotal', 'net', 'gross', 'sum']
        line_lower = line.lower()

        # Check for keywords
        has_keyword = any(kw in line_lower for kw in continuation_keywords)

        # Must also have some numeric content or structure
        has_numbers = bool(re.search(r'\d', line))
        has_structure = re.search(r'\s{3,}', line) is not None

        return has_keyword and (has_numbers or has_structure)

    def _convert_html_preserving_structure(self, html_content: str) -> str:
        """
        Convert HTML to text while preserving table and document structure.

        Key principles:
        - HTML tables are converted to text tables with spacing
        - Block elements create appropriate line breaks
        - Inline elements are removed but content preserved
        """
        # Decode entities first
        content = html.unescape(html_content)

        # Step 1: Convert HTML tables to structured text
        content = self._convert_html_tables_to_text(content)

        # Step 2: Remove scripts and styles completely
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

        # Step 3: Add line breaks for block elements
        block_elements = ['div', 'p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
        for elem in block_elements:
            content = re.sub(f'<{elem}[^>]*>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(f'</{elem}>', '', content, flags=re.IGNORECASE)

        # Step 4: Remove remaining tags
        content = re.sub(r'<[^>]+>', '', content)

        # Step 5: Clean up entities
        content = re.sub(r'&nbsp;', ' ', content, flags=re.IGNORECASE)
        content = re.sub(r'&amp;', '&', content, flags=re.IGNORECASE)
        content = re.sub(r'&lt;', '<', content, flags=re.IGNORECASE)
        content = re.sub(r'&gt;', '>', content, flags=re.IGNORECASE)
        content = re.sub(r'&quot;', '"', content, flags=re.IGNORECASE)
        content = re.sub(r'&#\d+;', ' ', content)
        content = re.sub(r'&\w+;', ' ', content)

        # Step 6: Clean up whitespace conservatively
        # Don't collapse tabs (they might be table delimiters)
        content = re.sub(r' {2,}', ' ', content)  # Multiple spaces to single
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Limit consecutive newlines

        return content

    def _convert_html_tables_to_text(self, html_str: str) -> str:
        """
        Convert HTML tables to properly formatted text tables.

        Preserves table structure using spacing and alignment.
        """
        def process_table(match):
            table_html = match.group(0)

            # Extract rows
            rows: List[List[str]] = []
            row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)

            for row_match in row_pattern.finditer(table_html):
                cells: List[str] = []
                row_content = row_match.group(1)

                # Extract cells (td and th)
                cell_pattern = re.compile(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', re.IGNORECASE | re.DOTALL)

                for cell_match in cell_pattern.finditer(row_content):
                    cell_text = cell_match.group(1)
                    # Clean cell content
                    cell_text = re.sub(r'<[^>]+>', '', cell_text)
                    cell_text = html.unescape(cell_text)
                    cell_text = ' '.join(cell_text.split())
                    cells.append(cell_text)

                if cells:
                    rows.append(cells)

            # Format as text table
            if rows:
                return '\n' + self._format_text_table(rows) + '\n'
            else:
                return ''

        # Replace all HTML tables with text versions
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
        return table_pattern.sub(process_table, html_str)

    def _format_text_table(self, rows: List[List[str]]) -> str:
        """
        Format table rows as aligned text with improved human readability.

        Creates properly spaced columns with alignment and separator lines.
        """
        if not rows:
            return ""

        # Calculate column widths
        num_cols = max(len(row) for row in rows)
        col_widths: List[int] = []

        for col_idx in range(num_cols):
            max_width = 0
            for row in rows:
                if col_idx < len(row):
                    max_width = max(max_width, len(row[col_idx]))
            # Don't cap column width - preserve full content
            col_widths.append(max_width)

        # Format rows with proper spacing
        formatted_rows: List[str] = []

        for row_idx, row in enumerate(rows):
            formatted_cells: List[str] = []

            for col_idx in range(num_cols):
                if col_idx < len(row):
                    cell = row[col_idx]  # Don't truncate cell content
                else:
                    cell = ''

                # Right-align numbers, left-align text
                if cell and re.match(r'^[\d,.\-$%()]+$', cell.strip()):
                    formatted_cells.append(cell.rjust(col_widths[col_idx]))
                else:
                    formatted_cells.append(cell.ljust(col_widths[col_idx]))

            # Join cells with proper spacing
            formatted_rows.append('   '.join(formatted_cells))

            # Add separator line after header row (first row)
            if row_idx == 0 and len(rows) > 1:
                separator_parts = []
                for col_idx in range(num_cols):
                    separator_parts.append('-' * col_widths[col_idx])
                formatted_rows.append('   '.join(separator_parts))

        return '\n'.join(formatted_rows)

    def _minimal_text_preprocessing(self, content: str) -> str:
        """
        Minimal preprocessing for text documents.

        Only fixes critical issues without disrupting structure.
        """
        # Decode entities if present
        content = html.unescape(content)

        # Ensure section headers are on new lines, but preserve internal spacing
        # Use lookahead/lookbehind to avoid affecting table content
        content = re.sub(r'(?<![0-9])\s*(ITEM\s*\d+[A-Z]?\.?)(?![0-9])', r'\n\1', content, flags=re.IGNORECASE)
        content = re.sub(r'(?<![0-9])\s*(PART\s*[IVX]+\.?)(?![0-9])', r'\n\1', content, flags=re.IGNORECASE)

        return content

    @staticmethod
    def _is_html_document(content: str) -> bool:
        """Detect if the document is HTML-formatted."""
        sample = content[:15000].lower()
        html_indicators = [
            '<html', '<body', '<div', '<table', '<tr', '<td',
            '<!doctype', '<meta', '<style', '<font'
        ]
        return any(indicator in sample for indicator in html_indicators)

    def _parse_filing_metadata(self, content: str, file_path: Path) -> Optional[Filing]:
        """Parse filing metadata from document content."""
        try:
            # Extract CIK
            cik = None
            cik_patterns = [
                r"(?:CENTRAL\s*INDEX\s*KEY|CIK)[\s:]*(\d{4,10})",
                r"(?:COMPANY\s*CONFORMED\s*NAME).*?(?:CENTRAL\s*INDEX\s*KEY|CIK)[\s:]*(\d{4,10})",
                r"^\s*(\d{10})\s*$",
            ]

            for pattern_str in cik_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                match = pattern.search(content[:10000])
                if match:
                    cik = match.group(1).zfill(10)
                    break

            if not cik:
                cik_from_name = re.search(r"(\d{4,10})", file_path.name)
                cik = cik_from_name.group(1).zfill(10) if cik_from_name else "0000000000"

            # Extract company name
            company_name = self.normalizer.extract_company_name(content)
            if not company_name:
                company_name = "Unknown Company"

            # Extract filing date
            filing_date = None
            date_patterns = [
                r"(?:FILED\s*AS\s*OF\s*DATE|Filing\s*Date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                r"(?:DATE\s*OF\s*FILING|CONFORMED\s*SUBMISSION\s*TYPE).*?(\d{8})",
                r"(?:PERIOD\s*OF\s*REPORT)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            ]

            for pattern_str in date_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                match = pattern.search(content[:10000])
                if match:
                    date_str = match.group(1)
                    filing_date = self._parse_date(date_str)
                    if filing_date:
                        break

            if not filing_date:
                date_from_name = re.search(r"(\d{8})", file_path.name)
                if date_from_name:
                    try:
                        filing_date = datetime.strptime(date_from_name.group(1), "%Y%m%d")
                    except ValueError:
                        filing_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                else:
                    filing_date = datetime.fromtimestamp(file_path.stat().st_mtime)

            # Extract form type
            form_type = "10-K"  # Default

            for pattern in self.patterns["form_type"]:
                match = pattern.search(content[:10000])
                if match:
                    form_type_raw = match.group(1).upper()
                    if '10-Q' in form_type_raw:
                        if 'A' in form_type_raw or '/A' in form_type_raw:
                            form_type = "10-Q/A"
                        else:
                            form_type = "10-Q"
                    elif '10-K' in form_type_raw:
                        if 'A' in form_type_raw or '/A' in form_type_raw:
                            form_type = "10-K/A"
                        else:
                            form_type = "10-K"
                    break

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
        """Parse date string to datetime object."""
        formats = [
            "%m/%d/%Y", "%m-%d-%Y",
            "%m/%d/%y", "%m-%d-%y",
            "%Y-%m-%d", "%Y/%m/%d",
            "%Y%m%d",
            "%B %d, %Y", "%b %d, %Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _save_extraction_result(self, result: ExtractionResult) -> None:
        """Save extraction results to output files."""
        cik = result.filing.cik
        company_name = self.normalizer.sanitize_filename(result.filing.company_name)
        filing_date = result.filing.filing_date.strftime("%Y-%m-%d")
        form_type = result.filing.form_type.replace('/', '-')

        filename = f"({cik})_({company_name})_({filing_date})_({form_type}).txt"
        output_path = self.output_dir / filename

        final_content: List[str] = []
        final_content.append("=" * 80)
        final_content.append(f"CIK: {cik}")
        final_content.append(f"Company Name: {result.filing.company_name}")
        final_content.append(f"Filing Date: {filing_date}")
        final_content.append(f"Form Type: {form_type}")
        final_content.append(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        final_content.append("=" * 80)
        final_content.append("")

        # MD&A text with tables already in place
        final_content.append(result.mdna_text)

        # Add cross-references if any
        if result.cross_references:
            resolved_refs = [ref for ref in result.cross_references if ref.resolved]
            if resolved_refs:
                final_content.append("")
                final_content.append("-" * 80)
                final_content.append("CROSS-REFERENCES")
                final_content.append("-" * 80)
                final_content.append("")

                for ref in resolved_refs:
                    if ref.resolution_text:
                        final_content.append(f"[{ref.reference_text}]:")
                        final_content.append(ref.resolution_text)
                        final_content.append("")

        output_text = '\n'.join(final_content)
        self.file_handler.write_file(output_path, output_text)
        logger.info(f"Saved MD&A to: {output_path}")

    def process_directory(self, input_dir: Path, cik_filter=None) -> Dict[str, Any]:
        """Process all files in a directory."""
        stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "filtered_out": 0,
            "errors": []
        }

        text_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.TXT"))
        stats["total_files"] = len(text_files)

        logger.info(f"Found {len(text_files)} text files to process")

        for file_path in text_files:
            if cik_filter and cik_filter.has_cik_filters():
                cik, year, form_type = self._parse_file_metadata_simple(file_path)

                if not cik_filter.should_process_filing(cik, form_type, year):
                    stats["filtered_out"] += 1
                    logger.info(f"Filtered out: {file_path.name}")
                    continue

            result = self.extract_from_file(file_path)
            if result:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(str(file_path))

        return stats

    def _parse_file_metadata_simple(self, file_path: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Parse basic metadata for CIK filtering."""
        filename = file_path.name

        cik_match = re.search(r'(\d{4,10})', filename)
        cik = cik_match.group(1).zfill(10) if cik_match else None

        year_match = re.search(r'(199[4-9]|20[0-2][0-9])', filename)
        year = int(year_match.group(1)) if year_match else None

        form_type = None
        filename_upper = filename.upper()
        if '10-Q' in filename_upper:
            form_type = "10-Q"
        elif '10-K' in filename_upper:
            form_type = "10-K"

        return cik, year, form_type