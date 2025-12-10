# MD&A Extractor - SEC Filing Parser

## Overview

A comprehensive Python system for extracting Management's Discussion and Analysis (MD&A) sections from SEC 10-K and 10-Q filings. The extractor processes ZIP archives of SEC EDGAR filings, filters by CIK (Central Index Key), and extracts MD&A sections while preserving document structure including tables, cross-references, and formatting.

## Key Features

### Core Functionality
- **Automated MD&A Section Detection**: Uses 150+ regex patterns to identify Item 7 (MD&A) sections across various filing formats and edge cases
- **CIK-Based Filtering**: Process only filings from specified companies using CIK identifiers
- **Intelligent Section Scoring**: Multi-factor scoring system (0-100 scale) to identify the correct MD&A section and avoid false positives like Table of Contents entries
- **Table Preservation**: In-place table detection and preservation maintaining original formatting and alignment
- **Cross-Reference Resolution**: Identifies and resolves cross-references to notes, exhibits, and other document sections
- **Incorporation by Reference Handling**: Detects when MD&A is incorporated by reference to proxy statements or other documents

### Advanced Text Processing
- **Structure-Aware Normalization**: Preserves columnar layouts, tables, and formatting while cleaning text
- **Multiple Detection Methods**: Falls back to content-based detection if standard Item 7 patterns fail
- **Unicode and Encoding Handling**: Robust text normalization with support for multiple character encodings
- **SEC Marker Removal**: Strips SEC-specific formatting while maintaining document integrity

## Architecture

### Main Components

#### 1. **main.py** - Entry Point and ZIP Processing
- `ImprovedZipProcessor`: Enhanced processor for handling ZIP archives with table preservation
- Command-line interface with configurable input/output directories
- Automatic cleanup of temporary files
- Statistics tracking and comprehensive logging

Key Features:
- Processes ZIP files containing multiple SEC filings
- Extracts relevant 10-K filings based on CIK filters
- Maintains raw file copies (optional) for debugging
- Parallel processing support via multiprocessing
- Signal handling for graceful interruption

#### 2. **patterns.py** - Pattern Definitions
Contains comprehensive regex patterns organized by category:

**Item 7 Start Patterns** (150+ variations):
- Standard MD&A headers: "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"
- Abbreviated forms: "MD&A", "MDA", "M D & A"
- OCR error corrections: "MANAGMENT", "1TEM" (I→1 confusion)
- Spacing variations: indented, tab-prefixed, extra whitespace
- Punctuation variants: colons, dashes, bullets, brackets
- Mixed case: lowercase, title case, mixed
- Line break variations
- Company-specific prefixes: "OUR MANAGEMENT'S DISCUSSION"

**Additional Patterns**:
- Item 7A (Quantitative and Qualitative Disclosures)
- Item 8 (Financial Statements) boundary detection
- Item 2 (10-Q specific patterns)
- Cross-reference detection (notes, exhibits, sections)
- Table delimiters (horizontal lines, pipe-delimited, aligned columns)
- Table headers (dates, financial statement titles, audit status)
- Incorporation by reference statements
- SEC document markers for removal

#### 3. **section_parser.py** - Section Boundary Detection
`SectionParser` class handles MD&A section identification:

**10-K MD&A Detection**:
- Finds all potential Item 7 matches in document
- Applies sophisticated scoring algorithm (0-100 scale)
- Validates matches using multiple criteria
- Falls back to content-based detection if patterns fail

**Scoring System Components**:
- Position scoring: Heavily penalizes early matches (Table of Contents), favors middle sections (20-60% of document)
- TOC indicator penalties: Detects and avoids index entries
- Content validation: Checks for MD&A-specific keywords and patterns
- Length validation: Ensures sections meet minimum/maximum thresholds
- Confidence weighting: Based on pattern strength and document context

**10-Q MD&A Detection**:
- Adapted patterns for quarterly reports (Item 2)
- Shorter content thresholds
- Quarterly-specific keywords: "three months", "quarter", "interim"

**Validation**:
- Word count thresholds (100-50,000 for 10-K, 50-30,000 for 10-Q)
- Keyword presence validation
- Section boundary verification
- Subsection extraction (Overview, Results of Operations, Liquidity, etc.)

**Incorporation by Reference Detection**:
- Identifies when MD&A is incorporated from proxy statements (DEF 14A)
- Extracts document type, caption, and page references
- Captures context for potential resolution

#### 4. **table_parser.py** - Table Detection and Preservation
`TableParser` class identifies and preserves tables within MD&A text:

**Detection Methods**:
1. **Delimited Tables**:
   - Horizontal delimiters (lines of dashes, equals signs, underscores)
   - Pipe-delimited tables ("|" separated columns)
   - Minimum 2 columns, 2 rows

2. **Aligned Tables**:
   - Space-aligned columns (3+ consecutive spaces as delimiter)
   - Column boundary detection from header alignment
   - Dynamic column position tracking
   - Continuation line handling (totals, subtotals)

**Table Headers**:
- Date headers: "Year Ended December 31, 2023"
- Financial statement titles
- Unit specifications: "in thousands", "in millions"
- Key financial labels: Revenue, Income, Assets, etc.

**Preservation Strategy**:
- Maintains original formatting and spacing
- Preserves table titles (extracted from preceding lines)
- Keeps tables in their original document positions
- Stores both parsed content and original text

**Table Object Properties**:
- Content: List of rows (each row is a list of cell values)
- Position: Start/end character positions and line numbers
- Title: Extracted table caption
- Confidence: Detection confidence score (0.0-1.0)
- Table type: 'delimited', 'aligned', or 'mixed'
- Original text: Preserved formatting

#### 5. **cross_reference_parser.py** - Reference Resolution
`CrossReferenceParser` class handles document cross-references:

**Reference Types Detected**:
- Note references: "See Note 15 to Financial Statements"
- Item references: "Refer to Item 1A"
- Exhibit references: "Included as Exhibit 13"
- Section references: "Discussed in section titled 'Risk Factors'"

**Resolution Process**:
1. Pattern matching to identify reference syntax
2. Target extraction (note numbers, item identifiers, exhibit numbers)
3. Document search for referenced content
4. Text extraction with context preservation
5. Recursive resolution for nested references (max depth configurable)
6. Caching of resolved references

**Reference Object Properties**:
- Reference text: Original reference string
- Reference type: Classification (note, item, exhibit, section)
- Target ID: Identifier to resolve
- Position: Character positions in document
- Resolved: Boolean flag
- Resolution text: Extracted content (up to 2000 chars)

#### 6. **reference_resolver.py** - External Document Resolution
`ReferenceResolver` class attempts to resolve incorporated content:

**Supported Document Types**:
- DEF 14A (Proxy Statements)
- Exhibit 13 (Annual Reports to Shareholders)
- Exhibit 99 (Additional Exhibits)
- Appendices

**Resolution Strategy**:
1. Extract accession number from original filing
2. Generate search patterns for referenced document
3. Search filing directory for matching files
4. Extract content using caption or page references
5. Apply same MD&A detection logic to referenced document

**Filename Pattern Generation**:
- Handles accession numbers with/without dashes
- Multiple pattern variations per document type
- Flexible matching for different filing conventions

#### 7. **text_normalizer.py** - Text Cleaning and Normalization
`TextNormalizer` class provides structure-preserving text cleaning:

**Normalization Pipeline**:
1. SEC marker removal (page numbers, headers, TOC references)
2. Control character replacement (except tabs/newlines)
3. Unicode normalization (NFKD form)
4. Encoding issue correction (mojibake patterns)
5. Structure preservation or whitespace normalization (mode-dependent)

**Structure Preservation Mode**:
- Detects columnar content (3+ consecutive spaces)
- Identifies table delimiters
- Preserves indentation (up to 4 spaces)
- Maintains pipe-delimited structures
- Detects financial number patterns in columns

**Character Replacement Map**:
- Smart quotes: ' " " → " '
- Dashes: – — → - --
- Special characters: • · → *
- Non-breaking spaces → regular spaces
- Ellipsis: … → ...

**Additional Utilities**:
- Company name extraction from filing headers
- Filename sanitization (removes illegal characters)
- CSV-safe text cleaning (escapes quotes, removes newlines)

#### 8. **settings.py** - Configuration Management
Centralized configuration for:

**Directory Paths**:
- `INPUT_DIR`: ZIP files containing SEC filings
- `OUTPUT_DIR`: Extracted MD&A sections
- `LOG_DIR`: Error logs and processing logs
- `CIK_INPUT_DIR`: CIK filter CSV files

**File Patterns**:
- Valid extensions: .txt, .TXT
- ZIP extensions: .zip, .ZIP
- CIK CSV pattern: Extracts year from filename

**Processing Limits**:
- Max file size: 250 MB
- Max cross-reference depth: 3 levels
- Table minimums: 2 columns × 2 rows
- Chunk size: 4 MB for large file reading

**Encoding Preferences**:
- Priority order: utf-8 → latin-1 → cp1252 → ascii
- Fallback handling with error replacement

**Form Type Priority**:
- 10-K/A (amended) → 10-K → 10-Q/A → 10-Q
- Only 10-K forms processed when CIK filtering enabled

#### 9. **logger.py** - Logging Infrastructure
Comprehensive logging system:

**Log Levels**:
- DEBUG: Detailed processing information (verbose mode)
- INFO: Normal processing messages
- WARNING: Potential issues, fallback usage
- ERROR: Processing failures, exceptions
- CRITICAL: System-level errors

**Handlers**:
1. Console handler (colored output via colorlog)
2. File handler (error-only, append mode)

**Statistics Logging**:
- Total files processed
- Success/failure counts
- Table preservation statistics
- Processing time
- CIK filter matches

**Error Tracking**:
- Dedicated error log file
- Timestamp and context for each error
- File path association
- Exception stack traces (in verbose mode)

## Workflow

### Processing Pipeline

```
1. Load CIK Filter
   ├── Read CSV file with CIKs and tickers
   └── Store in CIKFilter object

2. Scan Input Directory
   ├── Find all ZIP files
   └── Queue for processing

3. Process Each ZIP File
   ├── Open ZIP archive
   ├── Enumerate text files (.txt)
   ├── For each file:
   │   ├── Quick content check (is it a 10-K?)
   │   ├── Extract CIK from header
   │   ├── Check CIK filter
   │   ├── Extract to raw directory (if match)
   │   └── Continue if filtered out
   └── Close archive

4. Extract MD&A Section
   ├── Read filing text
   ├── Detect form type (10-K vs 10-Q)
   ├── Find all potential Item 7 matches
   ├── Score each match (0-100 scale)
   │   ├── Position scoring
   │   ├── TOC detection
   │   ├── Content validation
   │   ├── Length checks
   │   └── Keyword verification
   ├── Select best match
   ├── Validate and extract section
   └── Apply fallback if needed

5. Process MD&A Content
   ├── Detect tables
   │   ├── Horizontal delimiters
   │   ├── Pipe-delimited
   │   └── Space-aligned columns
   ├── Preserve table formatting
   ├── Find cross-references
   ├── Resolve references (optional)
   ├── Normalize text
   │   ├── Remove SEC markers
   │   ├── Fix encoding issues
   │   ├── Preserve structure
   │   └── Clean whitespace
   └── Check for incorporation by reference

6. Save Results
   ├── Generate output filename (CIK-accession-year.txt)
   ├── Write to output directory
   ├── Update statistics
   └── Delete raw file (if configured)

7. Final Statistics
   ├── Total ZIPs processed
   ├── Total files examined
   ├── MD&A sections extracted
   ├── Tables preserved
   ├── Failures and errors
   └── Processing time
```

## Usage

### Command-Line Interface

```bash
python main.py \
  --input /path/to/zip/files \
  --output /path/to/output \
  --cik-csv /path/to/cik_list.csv \
  [--raw-dir /path/to/raw] \
  [--keep-raw] \
  [--preserve-tables] \
  [--verbose]
```

### Arguments

- `-i, --input`: Directory containing ZIP files with SEC filings (default: `./input`)
- `-o, --output`: Directory for extracted MD&A sections (default: `./output`)
- `-c, --cik-csv`: **Required** - CSV file with CIK and ticker columns
- `-r, --raw-dir`: Directory for raw extracted files (default: `output/raw_filings`)
- `--keep-raw`: Keep raw files after processing (default: delete)
- `--preserve-tables`: Preserve tables in original positions (default: True)
- `-v, --verbose`: Enable debug logging

### CIK CSV Format

CSV file should have at least these columns:
```csv
CIK,Ticker
0000051143,IBM
0000789019,MSFT
0001018724,AMZN
```

CIKs can be:
- Zero-padded to 10 digits: "0000051143"
- Non-padded: "51143"

The extractor will match either format.

### Example Workflow

```bash
# 1. Prepare CIK list
cat > sp500_ciks.csv << EOF
CIK,Ticker
0000051143,IBM
0000789019,MSFT
0001018724,AMZN
EOF

# 2. Download SEC EDGAR bulk filings (ZIP format)
# Place ZIP files in ./input directory

# 3. Run extraction
python main.py \
  --input ./input \
  --output ./output \
  --cik-csv sp500_ciks.csv \
  --verbose

# 4. Results will be in ./output
# Format: CIK-AccessionNumber-Year.txt
# Example: 0000051143-0000051143-20-000003-2020.txt
```

## Output Format

### Extracted MD&A Files

Files are saved as: `{CIK}-{AccessionNumber}-{Year}.txt`

Example filename: `0000051143-0000051143-20-000003-2020.txt`

Structure:
```
[MD&A Section Header]

[Introduction/Overview]

[Tables preserved in original format]
                     Year Ended December 31,
                     2023        2022        2021
Revenue              $123.4      $110.2      $98.5
Operating Income     $ 45.6      $ 40.1      $35.2
Net Income           $ 38.2      $ 33.5      $29.8

[Results of Operations section]

[More content with preserved formatting]
```

### Processing Statistics

Console output includes:
```
==================================================
PROCESSING COMPLETE
==================================================
Time elapsed: 145.3 seconds
Total ZIP files: 5
Total files in ZIPs: 1,234
Files matching CIK filter: 42
Successfully extracted MD&A: 38
Failed: 4
Tables preserved in-place within MD&A text
Raw files deleted after processing
✅ MD&A sections saved to: ./output
```

### Error Logging

Errors are logged to: `logs/mdna_extraction_errors.log`

Format:
```
2024-12-10 14:23:45 - ERROR - [file.txt] Could not find Item 7 section
2024-12-10 14:23:46 - ERROR - [file2.txt] Section validation failed: too short
```

## Technical Details

### Pattern Matching Strategy

The extractor uses a multi-tiered approach:

1. **Comprehensive Pattern Library**: 150+ regex patterns cover:
   - Standard forms
   - Typographical variations
   - OCR errors
   - Formatting differences
   - Company-specific variations

2. **Scoring Algorithm**: Each match receives a score based on:
   - Pattern confidence (inherent pattern strength)
   - Document position (penalizes TOC entries)
   - Content validation (MD&A keywords present)
   - Length checks (reasonable section size)
   - Context analysis (surrounding text)

3. **Fallback Detection**: If pattern matching fails:
   - Direct content search for MD&A indicators
   - Contextual keyword matching
   - Position-based heuristics

### Table Detection Algorithm

Three-phase approach:

**Phase 1 - Delimited Tables**:
- Scan for horizontal delimiters (---, ===, ___)
- Detect pipe-delimited tables (|)
- Extract complete table structure

**Phase 2 - Aligned Tables**:
- Identify table headers (dates, financial terms)
- Calculate column boundaries from spacing
- Track column alignment across rows
- Handle continuation lines

**Phase 3 - Preservation**:
- Maintain original character-level spacing
- Preserve all whitespace within table regions
- Keep table titles from preceding lines
- Store both parsed and original formats

### Text Normalization Strategy

**Two-Mode Operation**:

1. **Structure Preservation Mode** (default for MD&A):
   - Detects columnar content (3+ spaces = column separator)
   - Preserves table delimiters
   - Maintains indentation (limited to 4 spaces)
   - Keeps original line breaks in structured regions
   - Normalizes only regular prose text

2. **Standard Mode** (for general text):
   - Collapses multiple spaces to single space
   - Removes excess empty lines
   - Normalizes all whitespace
   - Cleans up formatting uniformly

**Character-Level Operations**:
- Control character removal (except \t \n \r)
- Unicode normalization (NFKD decomposition)
- Smart quote replacement
- Encoding correction (fixes mojibake)
- ASCII transliteration where possible

### Cross-Reference Resolution

**Detection Phase**:
1. Scan text for reference patterns
2. Classify by type (note, item, exhibit, section)
3. Extract target identifiers
4. Store positions for context

**Resolution Phase**:
1. Search full document for target
2. Extract surrounding context (up to 2000 chars)
3. Apply normalization to extracted text
4. Check for nested references (recursive, max depth 3)
5. Cache results to avoid duplicate resolution

**Supported Reference Types**:
- Financial statement notes: "Note 15", "Note (3)"
- Form items: "Item 1A", "Item 7A"
- Exhibits: "Exhibit 13", "Exhibit 99.1"
- Sections: 'Section titled "Risk Factors"'
- Page references: "pages 25-30"

## Performance Characteristics

### Processing Speed
- ~50-100 files per minute on modern hardware
- Dependent on file size and complexity
- Parallel processing supported via multiprocessing

### Memory Usage
- Processes files one at a time
- Peak memory: ~500MB for large files (100MB+ text)
- Automatic cleanup of temporary files

### Disk Space
- Raw files: Optional, can be deleted immediately
- Output: ~5-50KB per MD&A section (text)
- Logs: ~1-10MB per 1000 files processed

## Dependencies

### Required Packages
```python
# Core
re                 # Regex pattern matching
pathlib           # File system operations
argparse          # Command-line parsing
signal            # Signal handling
atexit            # Cleanup registration

# Data Processing
pandas            # CSV reading, table manipulation
unicodedata       # Unicode normalization

# Logging
logging           # Standard logging
colorlog          # Colored console output

# File Handling
zipfile           # ZIP archive processing
shutil            # File operations
```

### Python Version
- Minimum: Python 3.7
- Recommended: Python 3.9+

## Limitations and Edge Cases

### Known Limitations

1. **Incorporation by Reference**: 
   - Only detects incorporation, doesn't automatically resolve
   - Requires referenced documents in same directory
   - May miss complex incorporation patterns

2. **OCR Text**:
   - Some OCR errors may not be caught
   - Very poor quality scans may fail extraction
   - Pattern library covers common OCR errors only

3. **Non-Standard Formats**:
   - Heavily formatted or stylized MD&A sections may be missed
   - Unusual document structures may score poorly
   - Some company-specific variations may not be covered

4. **Table Detection**:
   - Complex nested tables may be partially detected
   - Very large tables may be truncated
   - Some ASCII art or diagrams may be misidentified as tables

5. **Cross-References**:
   - Resolution limited to same document
   - External file references not automatically resolved
   - Maximum recursion depth prevents infinite loops

### Edge Cases Handled

- Multiple Item 7 sections (scoring selects best)
- TOC entries matching Item 7 pattern (penalized heavily)
- Very short MD&A sections (validation and fallback)
- Missing section boundaries (fallback detection)
- Incorporated by reference (detection flagged)
- OCR errors (extensive pattern variations)
- Encoding issues (multiple encoding attempts)
- Malformed tables (multiple detection methods)
- Nested cross-references (recursive resolution)

## Error Handling

### Validation Checks
- File size limits (250MB max)
- Section length thresholds (100-50,000 words for 10-K)
- Keyword presence validation
- Pattern match scoring threshold
- Table structure validation (min 2×2)

### Fallback Mechanisms
1. Pattern matching fails → Content-based detection
2. Main pattern set fails → Fallback pattern set
3. Section too short → Strict scoring retry
4. No Item 7 found → Direct MD&A content search
5. Encoding failure → Try alternate encodings

### Error Recovery
- Continue processing on individual file failures
- Log all errors with context
- Cleanup temporary files on interruption
- Graceful signal handling (SIGINT, SIGTERM)
- Automatic retry with stricter criteria

## Best Practices

### For Optimal Results

1. **CIK Filtering**:
   - Use complete, accurate CIK lists
   - Include leading zeros (10 digits)
   - Verify CIK-Ticker mapping

2. **File Organization**:
   - Keep ZIP files from same time period together
   - Use descriptive filenames for CIK CSVs
   - Separate output by year or dataset

3. **Processing Settings**:
   - Enable verbose mode for initial runs
   - Keep raw files for debugging (first run)
   - Use table preservation for financial analysis

4. **Quality Control**:
   - Spot-check extracted sections
   - Review error logs after processing
   - Validate word counts are reasonable

5. **Performance**:
   - Process in batches of 100-500 ZIP files
   - Use external temp directory for large datasets
   - Monitor disk space for raw files

### Troubleshooting

**Problem**: No MD&A sections extracted
- Check CIK CSV format and CIK values
- Verify ZIP files contain 10-K filings
- Run with `--verbose` to see pattern matching
- Check error log for specific failures

**Problem**: Sections too short or incomplete
- Review scoring criteria in logs
- Check if incorporation by reference
- Try with stricter thresholds (code modification)
- Verify source files are complete

**Problem**: Tables not preserved correctly
- Check original file formatting
- Review table detection logs (verbose mode)
- Adjust TABLE_MIN_COLUMNS/TABLE_MIN_ROWS in settings
- Verify table delimiters are consistent

**Problem**: High memory usage
- Process fewer ZIP files at once
- Reduce MAX_FILE_SIZE_MB in settings
- Enable immediate cleanup (CLEANUP_IMMEDIATELY)
- Clear output directory periodically

## Configuration Options

### Modifiable Settings (settings.py)

```python
# Processing Limits
MAX_FILE_SIZE_MB = 250                    # Skip larger files
MAX_CROSS_REFERENCE_DEPTH = 3             # Recursion limit
TABLE_MIN_COLUMNS = 2                     # Min table width
TABLE_MIN_ROWS = 2                        # Min table height

# Text Processing
CONTROL_CHAR_REPLACEMENT = " "            # Replace control chars
ENCODING_PREFERENCES = [                  # Try in order
    "utf-8", "latin-1", "cp1252", "ascii"
]

# Performance
CHUNK_SIZE = 2048 * 2048                  # 4MB read chunks
USE_EXTERNAL_TEMP = True                  # Use output dir for temp
CLEANUP_IMMEDIATELY = True                # Delete after each file

# Logging
CONTINUE_ON_ERROR = True                  # Don't stop on failures
MAX_ERRORS_PER_FILE = 10                  # Error threshold
```

### Pattern Customization

Add company-specific patterns to `patterns.py`:

```python
# Add to ITEM_7_START_PATTERNS list
r"ITEM\s*7[\-:\s]+COMPANY_SPECIFIC_VARIANT",
```

### Scoring Adjustments (section_parser.py)

Modify `_score_item_7_match()`:

```python
# Adjust position penalties
if position_ratio < 0.05:
    score -= 40  # Make more/less aggressive

# Adjust content scoring
if keyword_count >= 5:
    score += 20  # Increase reward for keywords
```

## Future Enhancements

### Planned Features
- Automatic resolution of incorporated content
- PDF support (direct PDF parsing)
- HTML filing support (structured HTML extraction)
- Parallel ZIP processing (multiprocessing pool)
- Database storage (SQLite/PostgreSQL output)
- REST API interface
- Web-based monitoring dashboard

### Potential Improvements
- Machine learning-based section detection
- Sentiment analysis integration
- Named entity recognition (NER) for companies/people
- Financial metric extraction
- Chart and graph detection
- Multi-language support (for foreign registrants)

## License and Attribution

This extractor is designed for academic research and financial analysis. Users should:
- Comply with SEC EDGAR terms of use
- Respect rate limits when downloading filings
- Attribute extracted data appropriately
- Verify accuracy for critical applications

## Support and Contribution

### Reporting Issues
- Provide sample filing that failed
- Include verbose logs
- Describe expected vs actual behavior
- Note any error messages

### Contributing
Areas for contribution:
- Additional pattern variations
- New filing types (8-K, proxy statements)
- Performance optimizations
- Documentation improvements
- Test case development

## Version History

### Current Version: 2.0
- In-place table preservation
- Enhanced scoring algorithm
- Fallback detection mechanism
- Cross-reference resolution
- Structure-aware normalization
- Comprehensive pattern library (150+ patterns)

### Previous Versions
- 1.0: Basic Item 7 extraction
- 1.5: Added table detection
- 1.8: CIK filtering support

---

**Note**: This extractor is designed for research and analysis purposes. Always verify extracted content against original filings for critical applications. The system handles most common filing formats but may require tuning for specific datasets or time periods.