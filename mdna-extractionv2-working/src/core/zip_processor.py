"""
ZIP Archive Processor for SEC Filings
======================================

This module handles processing of ZIP archives containing SEC filings.
SEC EDGAR provides bulk downloads of filings as ZIP archives, and this
processor extracts and processes the contained documents.

Key Features:
- ZIP archive extraction with temporary directory management
- CIK-based filtering to process only specific companies
- Integration with FilingManager for filing selection logic
- Support for 10-K/10-Q form type prioritization
- Comprehensive statistics tracking for batch processing

Processing Pipeline:
1. Extract ZIP contents to temporary directory
2. Filter files by CIK if filter is provided
3. Register filings with FilingManager for selection
4. Process selected filings through MDNAExtractor
5. Clean up temporary files automatically

The processor supports the SEC's filing distribution format where
multiple filings are packaged together in quarterly or monthly
ZIP archives.
"""

import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Internal module imports
from ...src.core.extractor import MDNAExtractor
from ...src.core.file_handler import FileHandler
from ...src.core.filing_manager import FilingManager
from ...src.core.cik_filter import CIKFilter
from ...src.utils.logger import get_logger, log_error
from ...config.settings import VALID_EXTENSIONS, ZIP_EXTENSIONS

# Initialize module logger
logger = get_logger(__name__)


class ZipProcessor:
    """
    Handles processing of ZIP archives containing SEC filings.

    This processor integrates with CIK filtering and FilingManager
    for intelligent filing selection. It manages the complete workflow
    from ZIP extraction through MD&A extraction.

    Attributes:
        output_dir: Directory where extracted MD&A files will be saved
        extractor: MDNAExtractor instance for processing individual filings
        file_handler: FileHandler for file I/O operations
        filing_manager: FilingManager for filing selection logic
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the ZIP processor with output directory.

        Args:
            output_dir: Path where extracted MD&A files will be saved
        """
        self.output_dir = Path(output_dir)

        # Initialize MD&A extractor with same output directory
        self.extractor = MDNAExtractor(output_dir)

        # File handler for any direct file operations
        self.file_handler = FileHandler()

        # Filing manager handles selection logic (10-K vs 10-Q priority, etc.)
        self.filing_manager = FilingManager()

    def process_zip_file(
        self,
        zip_path: Path,
        cik_filter: Optional[CIKFilter] = None
    ) -> Dict[str, any]:
        """
        Process a single ZIP file containing SEC filings.

        This method implements a three-phase approach:
        1. EXTRACTION PHASE: Extract files to temp directory and apply CIK filter
        2. SELECTION PHASE: Use FilingManager to select which filings to process
        3. PROCESSING PHASE: Extract MD&A from selected filings

        The FilingManager handles intelligent selection logic, such as
        preferring 10-K filings over 10-Q for the same company/year,
        and handling amended filings (10-K/A) appropriately.

        Args:
            zip_path: Path to the ZIP archive to process
            cik_filter: Optional CIKFilter for company/year filtering

        Returns:
            Dictionary containing processing statistics:
            - zip_file: Path to the processed ZIP
            - total_files: Number of text files in the archive
            - processed: Number of successfully processed filings
            - failed: Number of failed extractions
            - filtered_out: Number of files skipped by CIK filter
            - errors: List of error details
        """
        logger.info(f"Processing ZIP file: {zip_path}")

        # Initialize statistics tracking for this ZIP
        stats = {
            "zip_file": str(zip_path),
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "filtered_out": 0,
            "errors": []
        }

        try:
            # Open and process the ZIP archive
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get list of all files in the archive
                members = zf.namelist()

                # Filter to only include valid text file extensions (.txt, .TXT)
                text_members = [f for f in members if any(f.endswith(ext) for ext in VALID_EXTENSIONS)]
                stats["total_files"] = len(text_members)
                logger.info(f"Found {len(text_members)} text files in archive")

                # =============================================================
                # PHASE 1: EXTRACTION - Extract files and apply CIK filter
                # Use temporary directory that auto-cleans when done
                # =============================================================
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    candidates = []  # Files that pass filtering

                    for member in text_members:
                        try:
                            # Extract file to temporary directory
                            zf.extract(member, temp_path)
                            file_path = temp_path / member

                            # Apply CIK filter if provided
                            if cik_filter and cik_filter.has_cik_filters():
                                cik, year, form_type = self.extractor._parse_file_metadata_simple(file_path)
                                if not cik_filter.should_process_filing(cik, form_type, year):
                                    stats["filtered_out"] += 1
                                    logger.debug(f"Filtered out by CIK filter: {member} (CIK: {cik})")
                                    continue  # Skip this file

                            # Register with FilingManager for selection logic
                            cik, year, form_type = self.extractor._parse_file_metadata_simple(file_path)
                            if cik and year and form_type:
                                # FilingManager tracks all filings by company/year
                                self.filing_manager.add_filing(file_path, cik, year, form_type)
                                candidates.append(file_path)
                            else:
                                logger.debug(f"Metadata parse failed, skipping registration: {member}")

                        except Exception as e:
                            stats["failed"] += 1
                            stats["errors"].append({"file": member, "error": str(e)})
                            log_error(f"Error extracting {member} from {zip_path}: {e}")

                    # =============================================================
                    # PHASE 2: SELECTION - Let FilingManager choose which to process
                    # This handles 10-K/10-Q priority, amendments, duplicates, etc.
                    # =============================================================
                    selection = self.filing_manager._select_filings_to_process()
                    to_process = set(selection.get("process", []))

                    # =============================================================
                    # PHASE 3: PROCESSING - Extract MD&A from selected filings
                    # =============================================================
                    for file_path in to_process:
                        try:
                            result = self.extractor.extract_from_file(file_path)
                            if result:
                                stats["processed"] += 1
                            else:
                                stats["failed"] += 1
                                stats["errors"].append({"file": file_path.name, "error": "Extraction failed"})
                        except Exception as e:
                            stats["failed"] += 1
                            stats["errors"].append({"file": file_path.name, "error": str(e)})
                            log_error(f"Error processing {file_path.name} from {zip_path}: {e}")

        except zipfile.BadZipFile:
            # Handle corrupted or invalid ZIP files
            log_error(f"Invalid ZIP file: {zip_path}")
            stats["errors"].append({"file": str(zip_path), "error": "Invalid ZIP file"})
        except Exception as e:
            # Catch any other unexpected errors
            log_error(f"Error processing ZIP file {zip_path}: {e}")
            stats["errors"].append({"file": str(zip_path), "error": str(e)})

        # Log summary and return statistics
        logger.info(f"ZIP complete: {stats['processed']} processed, {stats['filtered_out']} filtered, {stats['failed']} failed")
        return stats

    def process_directory(
        self,
        input_dir: Path,
        cik_filter: Optional[CIKFilter] = None
    ) -> Dict[str, any]:
        """
        Process all ZIP files in a directory with optional CIK filtering.

        Batch processing method for handling multiple ZIP archives. This is
        useful for processing quarterly or annual bulk downloads from SEC EDGAR.

        Files are processed in sorted order for reproducibility.

        Args:
            input_dir: Directory containing ZIP archives to process
            cik_filter: Optional CIKFilter for company/year filtering

        Returns:
            Dictionary containing aggregated statistics:
            - total_zips: Number of ZIP files found
            - total_files: Total number of text files across all ZIPs
            - processed: Total successful extractions
            - failed: Total failed extractions
            - filtered_out: Total files skipped by CIK filter
            - zip_stats: List of per-ZIP statistics dictionaries
        """
        # Initialize aggregated statistics
        overall_stats = {
            "total_zips": 0,
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "filtered_out": 0,
            "zip_stats": []  # Detailed stats for each ZIP
        }

        # Find all ZIP files in the directory (both .zip and .ZIP extensions)
        zip_files = []
        for ext in ZIP_EXTENSIONS:
            zip_files.extend(input_dir.glob(f"*{ext}"))

        # Remove duplicates (can occur if extension is already uppercase)
        zip_files = list(set(zip_files))
        overall_stats["total_zips"] = len(zip_files)

        logger.info(f"Found {len(zip_files)} ZIP files to process in {input_dir}")

        # Process each ZIP file in sorted order for reproducibility
        for zip_path in sorted(zip_files):
            # Process the ZIP and get its statistics
            stats = self.process_zip_file(zip_path, cik_filter=cik_filter)

            # Aggregate statistics
            overall_stats["zip_stats"].append(stats)
            overall_stats["total_files"] += stats.get("total_files", 0)
            overall_stats["processed"] += stats.get("processed", 0)
            overall_stats["failed"] += stats.get("failed", 0)
            overall_stats["filtered_out"] += stats.get("filtered_out", 0)

        return overall_stats
