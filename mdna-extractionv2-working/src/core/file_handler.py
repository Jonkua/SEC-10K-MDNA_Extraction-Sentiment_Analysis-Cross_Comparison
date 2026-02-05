"""
File Handling Utilities for SEC Filing Processing
==================================================

This module provides robust file I/O operations with automatic encoding
detection, which is essential for processing SEC filings that may use
various character encodings (UTF-8, Latin-1, Windows-1252, etc.).

Key Features:
- Automatic encoding detection using chardet library
- File size limits to prevent memory issues with large files
- Chunked reading for efficient processing of large files
- Directory listing with extension filtering
- Safe file writing with parent directory creation

The encoding detection strategy:
1. Try common encodings (UTF-8, Latin-1, CP1252) in order of preference
2. If all fail, use chardet to detect the actual encoding
3. Fall back gracefully with error logging
"""

import chardet
from pathlib import Path
from typing import Optional, List

# Import configuration settings for file handling
from ...config.settings import (
    ENCODING_PREFERENCES,  # List of encodings to try: ['utf-8', 'latin-1', 'cp1252']
    MAX_FILE_SIZE_MB,      # Maximum file size to process (prevents memory issues)
    CHUNK_SIZE             # Size of chunks for reading large files
)
from ...src.utils.logger import get_logger

# Initialize module logger for file operation tracking
logger = get_logger(__name__)


class FileHandler:
    """
    Handles file I/O operations with encoding detection.

    This class provides methods for reading and writing files with automatic
    encoding detection, which is crucial for processing SEC filings that
    may use different character encodings depending on when they were filed.

    Usage:
        handler = FileHandler()
        content = handler.read_file(Path("filing.txt"))
        handler.write_file(Path("output.txt"), processed_content)
    """

    def read_file(self, file_path: Path) -> Optional[str]:
        """
        Read file content with automatic encoding detection.

        This method implements a multi-step strategy for reading files:
        1. Verify file exists and check size limits
        2. Try preferred encodings (UTF-8, Latin-1, CP1252) in order
        3. If preferred encodings fail, use chardet for detection
        4. Return content or None on failure

        SEC filings from different eras may use different encodings:
        - Modern filings: Usually UTF-8
        - Older filings: Often Latin-1 or Windows-1252
        - Some filings: May have mixed or non-standard encodings

        Args:
            file_path: Path to the file to read

        Returns:
            File content as string, or None if reading fails
        """
        # Verify file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Check file size to prevent memory issues with very large files
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large ({file_size_mb:.1f} MB): {file_path}")
            return None

        # STRATEGY 1: Try preferred encodings in order
        # This is faster than using chardet for every file
        for encoding in ENCODING_PREFERENCES:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                # This encoding didn't work, try the next one
                continue

        # STRATEGY 2: Use chardet to detect encoding
        # This is slower but more reliable for unusual encodings
        try:
            # Read raw bytes for encoding detection
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            if encoding:
                logger.info(f"Detected encoding: {encoding}")
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            else:
                logger.error(f"Could not detect encoding for: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def read_file_chunked(self, file_path: Path) -> Optional[str]:
        """
        Read large file in chunks for memory-efficient processing.

        This method is designed for files that are large but still within
        the size limit. It reads the file in fixed-size chunks to avoid
        loading the entire file into memory at once.

        Process:
        1. Sample the file to detect encoding (first 10KB)
        2. Read file in CHUNK_SIZE blocks
        3. Concatenate chunks into final string

        Note: This method is useful for files close to MAX_FILE_SIZE_MB
        where memory efficiency is important.

        Args:
            file_path: Path to the large file to read

        Returns:
            Complete file content as string, or None if failed
        """
        if not file_path.exists():
            return None

        try:
            # Detect encoding from file sample (first 10KB)
            # This is faster than reading entire file for detection
            with open(file_path, 'rb') as f:
                sample = f.read(10000)  # 10KB sample for encoding detection
                result = chardet.detect(sample)
                encoding = result['encoding'] or 'utf-8'  # Default to UTF-8

            # Read file in chunks to manage memory usage
            chunks = []
            with open(file_path, 'r', encoding=encoding) as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)  # Read CHUNK_SIZE bytes at a time
                    if not chunk:
                        break  # End of file reached
                    chunks.append(chunk)

            # Combine chunks into single string
            return ''.join(chunks)

        except Exception as e:
            logger.error(f"Error reading file in chunks {file_path}: {e}")
            return None

    def write_file(self, file_path: Path, content: str, encoding: str = 'utf-8'):
        """
        Write content to file with automatic directory creation.

        This method safely writes content to a file, creating any necessary
        parent directories. Uses UTF-8 encoding by default for maximum
        compatibility.

        Args:
            file_path: Path to the output file (absolute or relative)
            content: String content to write to the file
            encoding: Character encoding for output (default: UTF-8)

        Raises:
            Exception: If file writing fails (permissions, disk space, etc.)
        """
        try:
            # Create parent directories if they don't exist
            # parents=True: Create all intermediate directories
            # exist_ok=True: Don't raise error if directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content to file
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)

            logger.debug(f"Successfully wrote file: {file_path}")

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise  # Re-raise to allow caller to handle the error

    def list_files(self, directory: Path, extensions: List[str]) -> List[Path]:
        """
        List all files with given extensions in a directory.

        Useful for batch processing - finds all files matching specified
        extensions. Handles both lowercase and uppercase extensions
        (e.g., .txt and .TXT).

        Args:
            directory: Directory path to search (non-recursive)
            extensions: List of file extensions to match (e.g., ['.txt', '.htm'])

        Returns:
            Sorted list of Path objects for matching files, or empty list if
            directory doesn't exist
        """
        files = []

        # Verify directory exists
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return files

        # Find files for each extension (both lowercase and uppercase)
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))       # Lowercase: .txt
            files.extend(directory.glob(f"*{ext.upper()}"))  # Uppercase: .TXT

        # Remove duplicates (can occur if ext is already uppercase)
        files = list(set(files))

        # Return sorted for consistent ordering
        return sorted(files)