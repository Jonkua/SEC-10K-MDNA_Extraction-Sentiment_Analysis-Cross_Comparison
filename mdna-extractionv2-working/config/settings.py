"""
Configuration Settings for MD&A Extractor
==========================================

This module contains all configurable settings for the MD&A extraction
pipeline. Settings are organized into logical groups:

Directory Settings:
- INPUT_DIR: Where raw SEC filings are located
- OUTPUT_DIR: Where extracted MD&A files are saved
- LOG_DIR: Where log files are written
- CIK_INPUT_DIR: Where CIK filter CSV files are located

Processing Settings:
- File size limits and extensions
- Encoding preferences
- Error handling behavior

Performance Settings:
- Chunk sizes for large file processing
- Cross-reference depth limits

To modify behavior, edit the constants in this file or create a
local_settings.py file that overrides specific values.
"""

from pathlib import Path

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Base directory: project root (two levels up from this config file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Input directory: place SEC filing documents here for processing
INPUT_DIR = BASE_DIR / "input"

# Output directory: extracted MD&A files are saved here
OUTPUT_DIR = BASE_DIR / "output"

# Log directory: processing logs are written here
LOG_DIR = BASE_DIR / "logs"

# CIK input directory: CSV files for CIK-based filtering
CIK_INPUT_DIR = BASE_DIR / "cik_input"

# Auto-create directories if they don't exist
# This prevents errors on first run
for dir_path in [INPUT_DIR, OUTPUT_DIR, LOG_DIR, CIK_INPUT_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# =============================================================================
# FILE EXTENSION SETTINGS
# =============================================================================

# Valid SEC filing extensions (both lowercase and uppercase)
VALID_EXTENSIONS = {".txt", ".TXT"}
ZIP_EXTENSIONS = {".zip", ".ZIP"}
CIK_CSV_EXTENSIONS = {".csv", ".CSV"}

# =============================================================================
# CIK FILTERING SETTINGS
# =============================================================================

# Pattern to extract year from CIK filter CSV filenames
# Example: "sp500_2023.csv" -> extracts "2023"
CIK_CSV_PATTERN = r".*(\d{4}).*\.csv$"

# Form types to process when CIK filtering is enabled
# By default, only process annual reports (10-K)
FORM_TYPE_FILTER = {"10-K"}

# =============================================================================
# TEMPORARY FILE SETTINGS
# =============================================================================

# Use external temp directory to avoid filling internal drive
# When True, temp files go to OUTPUT_DIR/temp_extraction/
USE_EXTERNAL_TEMP = True

# Clean up temp files immediately after each file is processed
# Prevents temp directory from growing too large
CLEANUP_IMMEDIATELY = True

# Name of the temporary directory
TEMP_DIR_NAME = "temp_extraction"

# =============================================================================
# PROCESSING LIMITS
# =============================================================================

# Maximum file size to process (MB)
# Files larger than this are skipped to prevent memory issues
MAX_FILE_SIZE_MB = 250

# Maximum depth for resolving nested cross-references
# Prevents infinite loops from circular references
MAX_CROSS_REFERENCE_DEPTH = 3

# Minimum dimensions for table detection
# Smaller structures are treated as text
TABLE_MIN_COLUMNS = 2
TABLE_MIN_ROWS = 2

# =============================================================================
# TEXT NORMALIZATION SETTINGS
# =============================================================================

# Encoding preference order for reading files
# Try each in order until one works
ENCODING_PREFERENCES = ["utf-8", "latin-1", "cp1252", "ascii"]

# Character to replace control characters with
CONTROL_CHAR_REPLACEMENT = " "

# Pattern for matching multiple whitespace (used in normalization)
MULTIPLE_WHITESPACE_PATTERN = r"\s+"

# =============================================================================
# FILING PRIORITY SETTINGS
# =============================================================================

# Priority order for selecting filings when multiple exist
# Amended filings (/A) take precedence over original filings
FILING_PRIORITY = ["10-K/A", "10-K", "10-Q/A", "10-Q"]

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Log file name
LOG_FILENAME = "mdna_extraction_errors.log"

# Log message format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Date format for log timestamps
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# ERROR HANDLING SETTINGS
# =============================================================================

# Continue processing other files when one fails
CONTINUE_ON_ERROR = True

# Maximum errors to tolerate per file before giving up
MAX_ERRORS_PER_FILE = 10

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Chunk size for reading large files (4MB)
# Larger chunks are faster but use more memory
CHUNK_SIZE = 2048 * 2048  # 4MB chunks for reading large files