"""
Regex Patterns for MD&A Section Detection and Parsing
======================================================

This module contains all regex patterns used for parsing SEC filings.
Patterns are organized by their purpose:

Section Detection Patterns:
- ITEM_7_START_PATTERNS: Patterns to find MD&A section start in 10-K filings
- ITEM_7A_START_PATTERNS: Patterns for Item 7A (Market Risk) section
- ITEM_8_START_PATTERNS: Patterns for Item 8 (Financial Statements)
- ITEM_2_START_PATTERNS: Patterns for MD&A in 10-Q filings
- ITEM_3_START_PATTERNS: Patterns for Item 3 in 10-Q filings
- ITEM_4_START_PATTERNS: Patterns for Item 4 (Controls) in 10-Q
- PART_II_START_PATTERNS: Patterns for Part II section start

Document Structure Patterns:
- FORM_TYPE_PATTERNS: Patterns to identify 10-K vs 10-Q forms
- CROSS_REFERENCE_PATTERNS: Patterns for detecting cross-references
- TABLE_DELIMITER_PATTERNS: Patterns for identifying table structures
- TABLE_HEADER_PATTERNS: Patterns for detecting table headers
- SEC_MARKERS: Patterns for SEC-specific document markers
- INCORPORATION_BY_REFERENCE_PATTERNS: Patterns for IBR detection

Pattern Design Philosophy:
- Patterns are comprehensive to handle the wide variety of SEC filing formats
- Case-insensitive matching is used by default
- Multiple variations are included to catch OCR errors, typos, and formatting differences
- Patterns are ordered roughly by specificity (most specific first)

The compile_patterns() function compiles all patterns with appropriate flags
for optimal performance during matching.
"""

import re

# =============================================================================
# ITEM 7 (MD&A) START PATTERNS - 10-K FILINGS
# =============================================================================
# These patterns detect the beginning of the MD&A section in annual reports.
# The patterns are extensive to handle various formatting styles, OCR errors,
# and the many different ways companies format their section headers.

ITEM_7_START_PATTERNS = [
    # Standard “Management’s Discussion and Analysis”
    r"ITEM\s*7\.?\s*MANAGEMENT['’]?S\s*DISCUSSION\s*AND\s*ANALYSIS",
    r"ITEM\s*7\.?\s*MANAGEMENT['’]?S\s*DISCUSSION\s*&\s*ANALYSIS",
    # Abbreviated MD&A forms
    r"ITEM\s*7[\-:\s]+MD&A",
    r"ITEM\s*7[\-:\s]+M\s*D\s*&?\s*A",
    r"ITEM\s*7[\-:\s]+MDA",
    # Spelled‐out “Seven”
    r"ITEM\s+SEVEN\.?\s*MANAGEMENT['’]?S\s*DISCUSSION\s*AND\s*ANALYSIS",
    # Roman numeral
    r"ITEM\s*VII[\-:\s]+MD&A",
    # Part II prefix
    r"PART\s*II[-:\s]+ITEM\s*7\.?\s*MD&A",
    # Extended financial condition variants
    r"ITEM\s*7[\-:\s]+DISCUSSION\s*AND\s*ANALYSIS\s*OF\s*FINANCIAL\s*CONDITION",
    r"ITEM\s*7[\-:\s]+MANAGEMENT['’]?S\s*ANALYSIS\s*OF\s*FINANCIAL\s*CONDITION",
    # Extended results of operations variants
    r"ITEM\s*7[\-:\s]+DISCUSSION\s*AND\s*RESULTS\s*OF\s*OPERATIONS",
    r"ITEM\s*7[\-:\s]+ANALYSIS\s*OF\s*RESULTS\s*OF\s*OPERATIONS",
    r"ITEM\s*7[\-:\s]+FINANCIAL\s*CONDITION\s*AND\s*RESULTS\s*OF\s*OPERATIONS",
    # Overview & review headings
    r"ITEM\s*7[\-:\s]+OVERVIEW\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\-:\s]+REVIEW\s*OF\s*OPERATIONS",
    r"ITEM\s*7[\-:\s]+REVIEW\s*AND\s*RESULTS\s*OF\s*OPERATIONS",
    r"ITEM\s*7[\-:\s]+OPERATING\s*RESULTS\s*AND\s*DISCUSSION",
    # Outlook
    r"ITEM\s*7[\-:\s]+DISCUSSION\s*AND\s*OUTLOOK",
    # Liquidity & capital resources
    r"ITEM\s*7[\-:\s]+LIQUIDITY\s*AND\s*CAPITAL\s*RESOURCES",
    # Critical accounting
    r"ITEM\s*7[\-:\s]+CRITICAL\s*ACCOUNTING\s*POLICIES",

    # More flexible spacing and punctuation variations
    r"ITEM\s*7\s*[-–—.•·]\s*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7\b[^A-Z]*MANAGEMENT",  # Catches any separator before MANAGEMENT
    r"ITEM\s*7\s*\([A-Za-z]\)\s*MANAGEMENT",  # Item 7(a) Management's...

    # Business/Operating review variants
    r"ITEM\s*7[\-:\s]+BUSINESS\s*REVIEW\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\-:\s]+OPERATING\s*AND\s*FINANCIAL\s*REVIEW",
    r"ITEM\s*7[\-:\s]+FINANCIAL\s*REVIEW",

    # Catch cases where MD&A immediately follows without separator
    r"ITEM\s*7\s*MANAGEMENT",
    r"ITEM\s*7MD&A",
    r"ITEM7\s*[-–—.•·]?\s*MANAGEMENT",  # No space after ITEM

    # Financial performance variants
    r"ITEM\s*7[\-:\s]+ANALYSIS\s*OF\s*FINANCIAL\s*PERFORMANCE",
    r"ITEM\s*7[\-:\s]+PERFORMANCE\s*AND\s*RESULTS",

    # Common OCR errors/typos
    r"ITEM\s*7[\-:\s]+MANAGEMENTS\s*DISCUSSION",  # Missing apostrophe
    r"ITEM\s*7[\-:\s]+MANAGMENT",  # Common typo

    # Indented and space-prefixed variants (anchors removed; still match when present)
    r"\s{2,}ITEM\s*7\.?\s*MANAGEMENT['']?S\s*DISCUSSION",  # Formerly line-indented
    r"\s+ITEM\s*7\.?\s*MD&A",  # Formerly line-indented
    r"    ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # 4 spaces before ITEM
    r"\t+ITEM\s*7\.?\s*MANAGEMENT['']?S\s*DISCUSSION",  # Tab-indented

    # Missing or unusual punctuation/spacing
    r"ITEM7\s+MANAGEMENT['']?S\s*DISCUSSION",  # No space after ITEM
    r"ITEM\s+7MANAGEMENT['']?S\s*DISCUSSION",  # No space after 7
    r"ITEM\s*7\s*MANAGEMENTS\s*DISCUSSION",  # Missing apostrophe
    r"ITEM\s*7\s+MGMT\s*DISCUSSION",  # Abbreviated Management
    r"ITEM\s*7\s*MGMTS\s*DISCUSSION",  # Abbreviated with possessive

    # Parenthetical and bracketed variants
    r"ITEM\s*7\s*\(MD&A\)",  # (MD&A)
    r"ITEM\s*7\s*\[MANAGEMENT['']?S\s*DISCUSSION",  # [Management's Discussion
    r"\(ITEM\s*7\)\s*MANAGEMENT['']?S\s*DISCUSSION",  # (Item 7) Management's

    # With colons in different positions
    r"ITEM:?\s*7:?\s*MANAGEMENT",  # Item:7: Management
    r"ITEM\s*7\s*:\s*MANAGEMENT['']?S\s*DISCUSSION",  # Spaced colon

    # Line break variations
    r"ITEM\s*7\s*\n\s*MANAGEMENT['']?S\s*DISCUSSION",  # Line break between Item 7 and MD&A
    r"ITEM\s*7[\.\:\-]?\s*\n\s*MD&A",  # Line break before MD&A

    # ALL CAPS and mixed case variants
    r"item\s*7\.?\s*management['']?s\s*discussion",  # All lowercase
    r"Item\s*7\.?\s*Management['']?s\s*Discussion",  # Title case
    r"ITEM\s*7\.?\s*management['']?s\s*discussion",  # Mixed case

    # Unicode and special character variants
    r"ITEM\s*7[\.•·]\s*MANAGEMENT",  # Bullet points
    r"ITEM\s*7\s*[→➤]\s*MANAGEMENT",  # Arrow symbols
    r"ITEM\s*7\s*[|/\\]\s*MANAGEMENT",  # Pipe or slash separators

    # Financial/Business terminology variants
    r"ITEM\s*7[\.\:\-\s]*COMMENTARY\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*EXECUTIVE\s*COMMENTARY",
    r"ITEM\s*7[\.\:\-\s]*BUSINESS\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*OPERATING\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*FINANCIAL\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT\s*COMMENTARY",
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT\s*REPORT",
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT\s*REVIEW",

    # OCR and scanning errors
    r"ITEM\s*7[\.\:\-\s]*MANAGEMEN[IT]['']?S\s*DISCUSSION",  # OCR error in Management
    r"ITEM\s*7[\.\:\-\s]*MANAGE[MN]ENT['']?S\s*DISCUSS[IL]ON",  # Common OCR substitutions
    r"[IL]TEM\s*7[\.\:\-\s]*MANAGEMENT",  # I/L confusion
    r"ITEM\s*[7T][\.\:\-\s]*MANAGEMENT",  # 7/T confusion
    r"1TEM\s*7[\.\:\-\s]*MANAGEMENT",  # 1 instead of I

    # Partial matches and fragments
    r"ITEM\s*7[\.\:\-\s]*DISCUSSION\s*AND\s*ANALYSIS",  # Missing "Management's"
    r"ITEM\s*7[\.\:\-\s]*ANALYSIS\s*OF\s*FINANCIAL",  # Missing beginning
    r"ITEM\s*7[\.\:\-\s]*FINANCIAL\s*ANALYSIS",  # Shortened version

    # With company-specific prefixes
    r"ITEM\s*7[\.\:\-\s]*OUR\s*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*THE\s*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*COMPANY\s*MANAGEMENT['']?S\s*DISCUSSION",

    # Double-spaced and extra whitespace
    r"ITEM\s{2,}7[\.\:\-\s]*MANAGEMENT",  # Multiple spaces
    r"I\s*T\s*E\s*M\s*7[\.\:\-\s]*MANAGEMENT",  # Spaced out letters

    # With leading text or numbers
    r"\d+\.\s*ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # Numbered list: "2. Item 7"
    r"[A-Z]\.\s*ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # Lettered list: "B. Item 7"
    r"[-•]\s*ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # Bulleted list

    # With trailing identifiers
    r"ITEM\s*7[A-Z]?[\.\:\-\s]*\(?\s*MANAGEMENT",  # Item 7B, 7C, etc.
    r"ITEM\s*7\.0+[\.\:\-\s]*MANAGEMENT",  # Item 7.0 or 7.00

    # Wrapped in formatting
    r"\*+\s*ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # ***Item 7
    r"[=\-]{3,}\s*ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # === Item 7
    r">+\s*ITEM\s*7[\.\:\-\s]*MANAGEMENT",  # >> Item 7

    # With year or date references
    r"ITEM\s*7[\.\:\-\s]*(?:20\d{2}\s+)?MANAGEMENT['']?S\s*DISCUSSION",  # Item 7 - 2023 Management's
    r"ITEM\s*7[\.\:\-\s]*ANNUAL\s*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*FISCAL\s*YEAR\s*MANAGEMENT['']?S\s*DISCUSSION",

    # Subsection numbering variations
    r"ITEM\s*7\.1[\.\:\-\s]*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7\.01[\.\:\-\s]*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7\.[A-C][\.\:\-\s]*MANAGEMENT['']?S\s*DISCUSSION",

    # Alternative terminology for "Management's"
    r"ITEM\s*7[\.\:\-\s]*EXECUTIVE\s*DISCUSSION\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*BOARD\s*DISCUSSION\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*DIRECTOR['']?S\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*COMPANY\s*DISCUSSION\s*AND\s*ANALYSIS",

    # Different conjunction words
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT['']?S\s*DISCUSSION\s*PLUS\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT['']?S\s*DISCUSSION\s*WITH\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT['']?S\s*DISCUSSION\s*INCLUDING\s*ANALYSIS",

    # Performance-focused variants
    r"ITEM\s*7[\.\:\-\s]*PERFORMANCE\s*DISCUSSION\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*OPERATIONAL\s*PERFORMANCE\s*REVIEW",
    r"ITEM\s*7[\.\:\-\s]*BUSINESS\s*PERFORMANCE\s*ANALYSIS",

    # Strategic/Forward-looking variants
    r"ITEM\s*7[\.\:\-\s]*STRATEGIC\s*OVERVIEW\s*AND\s*ANALYSIS",
    r"ITEM\s*7[\.\:\-\s]*MANAGEMENT['']?S\s*PERSPECTIVE",
    r"ITEM\s*7[\.\:\-\s]*OPERATIONAL\s*INSIGHTS\s*AND\s*ANALYSIS",

    # With document type qualifiers
    r"ITEM\s*7[\.\:\-\s]*ANNUAL\s*MANAGEMENT['']?S\s*DISCUSSION",
    r"ITEM\s*7[\.\:\-\s]*QUARTERLY\s*MANAGEMENT['']?S\s*DISCUSSION",

    # Non-standard spacing and punctuation
    r"ITEM\s*7\s*\(\s*MD&A\s*\)",
    r"ITEM\s*7\s*\-\-\s*MANAGEMENT['']?S\s*DISCUSSION",

# --- Canonical long titles (with common punctuation/spacing/encodings) ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT(?:['’`]|&rsquo;|&#39;|â€™)?S\s*DISCUSSION\s*(?:AND|&|&amp;)\s*ANALYSIS\s*OF\s*FINANCIAL\s*CONDITION\s*(?:,?\s*(?:AND|&|&amp;)\s*)?RESULTS\s*OF\s*OPERATIONS\b(?:\s*\(MD&A\))?",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT['’`]?S\s*DISCUSSION\s*&\s*ANALYSIS\s*OF\s*FINANCIAL\s*CONDITION\s*(?:,?\s*&\s*)?RESULTS\s*OF\s*OPERATIONS",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT['’`]?S\s*DISCUSSION\s*(?:AND|&|&amp;)\s*ANALYSIS\s*OF\s*RESULTS\s*OF\s*OPERATIONS\s*(?:,?\s*(?:AND|&|&amp;)\s*)?FINANCIAL\s*CONDITION",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT['’`]?S\s*DISCUSSION\s*(?:AND|&|&amp;)\s*ANALYSIS\s*OF\s*FINANCIAL\s*CONDITION,\s*RESULTS\s*OF\s*OPERATIONS(?:\s*AND\s*CASH\s*FLOWS)?",

    # --- “Management Discussion and Analysis” without apostrophe or with spacing ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENTS?\s*DISCUSSION\s*(?:AND|&|&amp;)\s*ANALYSIS\s*OF\s*FINANCIAL\s*CONDITION.*?RESULTS\s*OF\s*OPERATIONS",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*M\s*D\s*&?\s*A(?:\s*\(MD&A\))?\b",

    # --- Compact MD&A forms following Item 7 ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*(?:[-–—]|&mdash;|&ndash;)\s*MD\s*&?\s*A\b",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*\(MD&A\)",

    # --- HTML/encoding noise between tokens (kept to a few high-value variants) ---
    r"ITEM\s*7(?!\s*[A-Za-z])(?:\s|&nbsp;|<[^>]{0,120}>)*MANAGEMENT(?:['’`]|&rsquo;|&#39;|â€™)?S(?:\s|&nbsp;|<[^>]{0,120}>)*DISCUSSION(?:\s|&nbsp;|<[^>]{0,120}>)*(?:AND|&|&amp;)(?:\s|&nbsp;|<[^>]{0,120}>)*ANALYSIS(?:\s|&nbsp;|<[^>]{0,120}>)*OF(?:\s|&nbsp;|<[^>]{0,120}>)*FINANCIAL(?:\s|&nbsp;|<[^>]{0,120}>)*CONDITION(?:\s|&nbsp;|<[^>]{0,120}>)*(?:,?\s*(?:AND|&|&amp;)\s*)?RESULTS(?:\s|&nbsp;|<[^>]{0,120}>)*OF(?:\s|&nbsp;|<[^>]{0,120}>)*OPERATIONS",
    r"ITEM\s*7(?!\s*[A-Za-z])(?:\s|&nbsp;|<[^>]{0,120}>)*(?:MD&A|MD\s*&\s*A)\b",

    # --- Amendment / restatement / revision cues for 10-K/A ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT.*?\((?:AS\s+AMENDED|AMENDED(?:\s+AND\s+RESTATED)?|RESTATED|RECAST|REVISED|UPDATED|SUPPLEMENT(?:AL)?)\)",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*\((?:AS\s+AMENDED|AMENDED(?:\s+AND\s+RESTATED)?|RESTATED|RECAST|REVISED|UPDATED|SUPPLEMENT(?:AL)?)\)\s*MANAGEMENT",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*.*?AMENDMENT\s+NO\.\s*\d+.*?MANAGEMENT.*?(?:DISCUSSION|ANALYSIS|MD&A)",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*.*?(?:AMENDED|RESTATED|RECAST|REVISED)\b.*?(?:MANAGEMENT|MD&A).*?(?:DISCUSSION|ANALYSIS)",

    # --- Roman numerals, spelled-out, and numbered variants ---
    r"PART\s*II\s*[\.\:\-\)]*\s*ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT",
    r"ITEM\s*(?:NO\.?|NUMBER)?\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT",
    r"ITEM\s*VII\s*[\.\:\-\)]*\s*MANAGEMENT(?:['’`]|&rsquo;|&#39;|â€™)?S\s*DISCUSSION",
    r"ITEM\s*SEVEN\s*[\.\:\-\)]*\s*MANAGEMENT(?:['’`]|&rsquo;|&#39;|â€™)?S\s*DISCUSSION",

    # --- Brackets/parentheses/caption wrappers ---
    r"\[?\s*ITEM\s*7(?!\s*[A-Za-z])\s*\]?\s*[\.\:\-\)]*\s*MANAGEMENT",
    r"\(ITEM\s*7(?!\s*[A-Za-z])\)\s*MANAGEMENT",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*\[\s*MANAGEMENT(?:['’`]|&rsquo;|&#39;|â€™)?S\s*DISCUSSION",

    # --- Leaders/dashes between “Item 7” and title (avoid 7A via lookahead above) ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[-–—•·]+\s*MANAGEMENT(?:['’`]|&rsquo;|&#39;|â€™)?S\s*DISCUSSION",
    r"ITEM7(?![A-Za-z])\s*[-–—•·]*\s*MANAGEMENT",

    # --- Abbreviated “Management” forms seen in some layouts ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MGMT(?:'S|S)?\s*DISCUSSION",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MGMTS?\s*DISCUSSION",

    # --- Alternate wordings sometimes used as Item 7 headers ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*OPERATING\s*(?:AND\s*)?FINANCIAL\s*REVIEW\s*(?:AND\s*PROSPECTS)?",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*FINANCIAL\s*REVIEW\s*(?:AND\s*ANALYSIS)?",

    # --- Long-title variants including “Liquidity and Capital Resources” inline ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT['’`]?S\s*DISCUSSION\s*(?:AND|&|&amp;)\s*ANALYSIS\s*[-–—]|&mdash;|&ndash;\s*LIQUIDITY\s*AND\s*CAPITAL\s*RESOURCES",

    # --- “Overview/Outlook/Executive Summary” suffixed to MD&A header ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*(?:MD&A|MANAGEMENT['’`]?S\s*DISCUSSION\s*(?:AND|&|&amp;)\s*ANALYSIS)\s*[-–—]|&mdash;|&ndash;\s*(?:OVERVIEW|OUTLOOK|PERSPECTIVE|EXECUTIVE\s*SUMMARY)",

    # --- Encoded separators between “Discussion” and “Analysis” ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT['’`]?S\s*DISCUSSION\s*[|/\\]\s*ANALYSIS",

    # --- Recast/discontinued ops qualifiers on same header line ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT['’`]?S\s*DISCUSSION.*?\b(?:RECAST|DISCONTINUED\s*OPERATIONS|AS\s*REVISED|AS\s*RESTATED)\b",

    # --- Item 7 lines that start with “Analysis of … / Discussion of …” (nonstandard but seen) ---
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*ANALYSIS\s*OF\s*(?:FINANCIAL\s*CONDITION|RESULTS\s*OF\s*OPERATIONS)",
    r"ITEM\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*DISCUSSION\s*OF\s*(?:OPERATIONS|FINANCIAL\s*CONDITION)",

    # --- Case variants (kept loose; rely on IGNORECASE) ---
    r"Item\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*Management['’`]?s\s*Discussion",
    r"item\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*management['’`]?s\s*discussion",

    # --- Weirdly spaced “I T E M 7” ---
    r"I\s*T\s*E\s*M\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT",

    # --- With number sign / “No.” / decimals on the item number ---
    r"ITEM\s*#\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT",
    r"ITEM\s*NO\.\s*7(?!\s*[A-Za-z])\s*[\.\:\-\)]*\s*MANAGEMENT",
    r"ITEM\s*7\.0+\s*[\.\:\-\)]*\s*MANAGEMENT",
]


# =============================================================================
# ITEM 7A (MARKET RISK) PATTERNS - 10-K FILINGS
# =============================================================================
# These patterns detect Item 7A which marks the END of MD&A section.
# Item 7A discusses quantitative and qualitative market risk disclosures.

ITEM_7A_START_PATTERNS = [
    # Standard quantitative/qualitative heading
    r"^\s*ITEM\s*7A\.?\s*QUANTITATIVE\s*AND\s*QUALITATIVE\s*DISCLOSURES",
    r"^\s*ITEM\s*7A\.?\s*QUANTITATIVE\s*AND\s*QUALITATIVE",
    r"^\s*ITEM\s*7A\.?\s*QUANTITATIVE\s*DISCLOSURES",
    r"^\s*ITEM\s*7A\.?\s*QUALITATIVE\s*DISCLOSURES",

    # Market risk variants
    r"^\s*ITEM\s*7A\.?\s*MARKET\s*RISK\s*DISCLOSURES",
    r"^\s*ITEM\s*7A\.?\s*DISCLOSURES\s*ABOUT\s*MARKET\s*RISK",
    r"^\s*ITEM\s*7A\.?\s*MARKET\s*RISK",
    r"^\s*ITEM\s*7A\.?\s*RISK\s*DISCLOSURES",

    # Combined Q&Q and market risk
    r"^\s*ITEM\s*7A\.?\s*QUANTITATIVE\s*AND\s*QUALITATIVE\s*DISCLOSURES\s*ABOUT\s*MARKET\s*RISK",
    r"^\s*ITEM\s*7A\.?\s*QUANTITATIVE\s*AND\s*QUALITATIVE\s*&\s*MARKET\s*RISK\s*DISCLOSURES",

    # Spelled‐out “Seven A”
    r"^\s*ITEM\s+SEVEN\s*A\.?\s*QUANTITATIVE\s*AND\s*QUALITATIVE",
    r"^\s*ITEM\s+SEVEN\s*A\.?\s*MARKET\s*RISK\s*DISCLOSURES",

    # Hyphen and colon variations
    r"^\s*ITEM\s*7A[\-:\s]+QUANTITATIVE\s*AND\s*QUALITATIVE",
    r"^\s*ITEM\s*7A[\-:\s]+MARKET\s*RISK",
    r"^\s*ITEM\s*7A[\-:\s]+QUANTITATIVE\s*DISCLOSURES",
    r"^\s*ITEM\s*7A[\-:\s]+QUALITATIVE\s*DISCLOSURES",

    # Abbreviated forms
    r"^\s*ITEM\s*7A[\-:\s]+Q\s*&\s*Q",
    r"^\s*ITEM\s*7A[\-:\s]+Q\s*&?\s*Q\s*DISCLOSURES",

    # Roman numeral seven
    r"^\s*ITEM\s*VIIA\.?\s*QUANTITATIVE\s*AND\s*QUALITATIVE",
    r"^\s*ITEM\s*VIIA\.?\s*MARKET\s*RISK\s*DISCLOSURES",
    r"^\s*ITEM\s*VIIA[\-:\s]+Q\s*&\s*Q",
]


# =============================================================================
# ITEM 8 (FINANCIAL STATEMENTS) PATTERNS - 10-K FILINGS
# =============================================================================
# Item 8 marks financial statements, another potential MD&A endpoint.

ITEM_8_START_PATTERNS = [
    # Basic Financial Statements
    r"^\s*ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s*8\.?\s*CONSOLIDATED\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",
    r"^\s*ITEM\s*8\.?\s*CONSOLIDATED\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",

    # Spelled-out Eight
    r"^\s*ITEM\s+EIGHT\.?\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s+EIGHT\.?\s*CONSOLIDATED\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s+EIGHT\.?\s*FINANCIAL\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",
    r"^\s*ITEM\s+EIGHT\.?\s*CONSOLIDATED\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",

    # Roman Numeral
    r"^\s*ITEM\s*VIII\.?\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s*VIII\.?\s*CONSOLIDATED\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s*VIII\.?\s*FINANCIAL\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",
    r"^\s*ITEM\s*VIII\.?\s*CONSOLIDATED\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",

    # Part II prefix
    r"^\s*PART\s*II\s*[-–—]\s*ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS",
    r"^\s*PART\s*II\s*[-–—]\s*ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS\s*(?:AND|&)\s*SUPPLEMENTARY\s*DATA",

    # Hyphens, colons and spaces
    r"^\s*ITEM\s*8[\-:\s]+FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s*8[\-:\s]+CONSOLIDATED\s*FINANCIAL\s*STATEMENTS",
    r"^\s*ITEM\s*8[\-:\s]+FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA",
    r"^\s*ITEM\s*8[\-:\s]+CONSOLIDATED\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA",

    # Abbreviated forms & variants
    r"^\s*ITEM\s*8[\-:\s]+FS\s*&?\s*SD",  # e.g. “FS & SD”
    r"^\s*ITEM\s*8[\-:\s]+STATEMENTS\s*AND\s*DATA",
    r"^\s*ITEM\s*8[\-:\s]+FINANCIAL\s*DATA",
]


# =============================================================================
# 10-Q FILING PATTERNS
# =============================================================================
# 10-Q (quarterly) filings use different item numbers than 10-K (annual) filings.
# Item 2 in 10-Q is equivalent to Item 7 in 10-K (both are MD&A).

ITEM_2_START_PATTERNS = [
    r"(?:^|\n)\s*ITEM\s*2[\.\:\-\s]*MANAGEMENT['’`]?[S]?\s*DISCUSSION\s*(?:AND|&)\s*ANALYSIS",
    r"(?:^|\n)\s*ITEM\s+TWO[\.\:\-\s]*MANAGEMENT['’`]?[S]?\s*DISCUSSION\s*(?:AND|&)\s*ANALYSIS",
    r"(?:^|\n)\s*ITEM\s*2[\.\:\-\s]*M\s*D\s*&?\s*A",
    r"(?:^|\n)\s*ITEM\s*2[\.\:\-\s]*DISCUSSION\s+OF\s+OPERATIONS",
    r"(?:^|\n)\s*MANAGEMENT['’`]?[S]?\s*DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS",
]


ITEM_3_START_PATTERNS = [
    r"^\s*ITEM\s*3[\.\:\-\s]*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK",
    r"^\s*ITEM\s+THREE[\.\:\-\s]*QUANTITATIVE\s+AND\s+QUALITATIVE\s+MARKET\s+RISK",
    r"^\s*ITEM\s*3[\.\:\-\s]*MARKET\s+RISK\s+DISCLOSURES",
    r"^\s*ITEM\s*3[\.\:\-\s]*QUANT\s*&?\s*QUAL\s+DISCLOSURES",
]


ITEM_4_START_PATTERNS = [
    r"^\s*ITEM\s*4[\.\:\-\s]*CONTROLS\s+AND\s+PROCEDURES",
    r"^\s*ITEM\s*4[\.\:\-\s]*DISCLOSURE\s+CONTROLS\s+AND\s+PROCEDURES",
    r"^\s*ITEM\s*4[\.\:\-\s]*EVALUATION\s+OF\s+DISCLOSURE\s+CONTROLS",
    r"^\s*ITEM\s+FOUR[\.\:\-\s]*CONTROLS\s+AND\s+PROCEDURES",
]


PART_II_START_PATTERNS = [
    r"^\s*PART\s*II[\.\:\-\s]*OTHER\s+INFORMATION",
    r"^\s*PART\s+TWO[\.\:\-\s]*OTHER\s+INFORMATION",
    r"^\s*PART\s*II[\.\:\-\s]*ITEM\s*1[\.\:\-\s]*LEGAL\s+PROCEEDINGS",
    r"^\s*PART\s*II[\.\:\-\s]*DISCLOSURE\s+ITEMS",
]


# =============================================================================
# FORM TYPE DETECTION PATTERNS
# =============================================================================
# These patterns identify whether a filing is 10-K, 10-K/A, 10-Q, or 10-Q/A.
# This is critical because different form types have different section structures.

FORM_TYPE_PATTERNS = [
    r"(?:FORM|CONFORMED\s*SUBMISSION\s*TYPE)[\s:\-]*(\d{1,2}-[KQ](?:/A|A)?)",
    r"^\s*(?:FORM\s*)?(\d{1,2}-[KQ](?:/A|A)?)\s*$",
    r"(?:QUARTERLY\s*REPORT\s*(?:ON\s*)?FORM\s*)(\d{1,2}-Q(?:/A|A)?)",
    r"(?:ANNUAL\s*REPORT\s*(?:ON\s*)?FORM\s*)(\d{1,2}-K(?:/A|A)?)",
    r"(?:Filed\s+on\s+Form\s*)(\d{1,2}-[KQ](?:/A|A)?)",
    r"Form\s+(10-[KQ])(?:/A|A)?",
]


# =============================================================================
# CROSS-REFERENCE PATTERNS
# =============================================================================
# These patterns detect references to other parts of the filing or external
# documents. Examples: "See Note 5", "Refer to Item 8", "See Exhibit 10.1"

CROSS_REFERENCE_PATTERNS = [
    # --- Note references ---
    r"(?:see|refer(?:red)?\s*to|as\s*discussed\s*in)\s*Note\s+(\d+)",  # See Note 3
    r"Note\s+(\d+)\s*(?:to|of)?\s*(?:the\s*)?(?:consolidated\s*)?financial\s*statements",  # Note 4 of FS
    r"Notes?\s+(\d+)\s*(?:through|and)\s+(\d+)",  # Notes 3 through 5

    # --- Part/Item references ---
    r"(?:see|refer(?:red)?\s*to|discussed\s*in|included\s*in)\s*Part\s*([IVX]+)[,\s]*Item\s*(\d+[A-Z]?)",  # Part II, Item 8
    r"Part\s*([IVX]+)[,\s]*Item\s*(\d+[A-Z]?)",  # Part II, Item 7A
    r"Item\s*(\d+[A-Z]?)\s*(?:of|in)?\s*Part\s*([IVX]+)",  # Item 7A of Part II
    r"(?:discussed|described)\s*(?:in|under)?\s*Item\s*(\d+[A-Z]?)",
    r"as\s+(?:set\s+forth|described)\s+in\s+Item\s*(\d+[A-Z]?)",
    r"discussed\s+in\s+(?:Item|Part\s+[IVX]+\s+Item)\s*(\d+[A-Z]?)",

    # --- Exhibit references ---
    r"(?:see|refer\s*to|contained\s*in)\s*Exhibit\s*(\d+(?:\.\d+)?)",
    r"Exhibit\s+(\d+(?:\.\d+)?)[\s\)]*(?:to|of)?\s*(?:this\s+Form\s+10-K|this\s+filing)?",

    # --- Section references (titled/quoted) ---
    r"(?:see|refer\s*to|discussed\s*in)?\s*(?:the\s*)?section\s*(?:entitled|captioned)?\s*['\"]([^'\"]+)['\"]",  # 'Liquidity and Capital Resources'
    r"(?:see|refer\s*to)?\s*(?:discussion\s*under\s*)?['\"]([^'\"]+)[\"']",  # "Results of Operations"
    r"(?:see|refer\s*to)\s*(?:the\s*)?(?:discussion\s*under\s*)?section\s*(?:called|titled)?\s*['\"]([^'\"]+)['\"]",

    # --- Generic backward/forward references ---
    r"(?:as\s+described\s+above|as\s+noted\s+below)\s+in\s+Item\s*(\d+[A-Z]?)",
    r"(?:refer\s*back\s*to|see\s*also)\s+Note\s+(\d+)",

    # --- Embedded table note reference ---
    r"see\s+accompanying\s+Notes?\s*(\d+)?\s*(?:through\s*(\d+))?",

    # --- Edge-case phrasing ---
    r"(?:see\s+also|refer\s+to)\s+(?:Note|Item|Section)\s*(\d+[A-Z]?)",

    # Specific financial statement references
    r"(?:see|refer\s*to)\s*(?:the\s*)?(?:Consolidated\s*)?Balance\s*Sheet(?:s)?\s*(?:on\s*page\s*(\d+))?",
    r"(?:see|refer\s*to)\s*(?:the\s*)?(?:Consolidated\s*)?Statement(?:s)?\s*of\s*(?:Operations|Income)",
    r"(?:see|refer\s*to)\s*(?:the\s*)?(?:Consolidated\s*)?Statement(?:s)?\s*of\s*Cash\s*Flow(?:s)?",

    # Footnote variations
    r"(?:see\s*)?footnote\s*(\d+)",
    r"(?:see\s*)?foot\s*note\s*(\d+)",
    r"\((\d+)\)\s*(?:above|below)",
    r"superscript\s*(\d+)",

    # Schedule references
    r"(?:see|refer\s*to)\s*Schedule\s*([IVX]+|\d+)",
    r"Schedule\s*([IVX]+|\d+)\s*(?:to|of)?\s*(?:this\s*filing|the\s*Form)",
    r"(?:included\s*in|contained\s*in)\s*Schedule\s*([IVX]+|\d+)",

    # Appendix variations
    r"(?:see|refer\s*to)\s*Appendix\s*([A-Z]+\d*|\d+[A-Z]*)",
    r"Appendix\s*([A-Z]+\d*|\d+[A-Z]*)\s*(?:contains|includes|sets\s*forth)",
    r"(?:included\s*as\s*)?Appendix\s*([A-Z]+\d*|\d+[A-Z]*)",

    # Annual report section references
    r"(?:see|refer\s*to)\s*(?:our\s*)?Annual\s*Report\s*(?:on\s*Form\s*10-K)?\s*(?:page\s*(\d+))?",
    r"(?:see|refer\s*to)\s*(?:the\s*)?Proxy\s*Statement\s*(?:page\s*(\d+))?",
    r"(?:see|refer\s*to)\s*(?:the\s*)?Quarterly\s*Report\s*(?:on\s*Form\s*10-Q)?",

    # Page range references
    r"(?:see\s*)?pages?\s*(\d+)\s*(?:through|to|thru|\-)\s*(\d+)",
    r"(?:on\s*)?page(?:s)?\s*(\d+)\s*et\s*seq",

    # Narrative section references
    r"(?:discussed\s*(?:in|under)\s*)?(?:the\s*)?section\s*(?:titled|entitled|captioned)\s*[\"']([^\"']{5,50})[\"']",
    r"(?:see\s*)?(?:the\s*)?(?:preceding|following)\s*(?:section|discussion|table|analysis)",
    r"(?:as\s*)?(?:detailed|outlined|described)\s*(?:in\s*the\s*)?(?:table|section|discussion)\s*(?:above|below)"

    # Direct incorporation statements
    r"(?:the\s+)?(?:required\s+)?(?:MD&A|Management['']?s\s+Discussion)\s+(?:is\s+)?incorporated\s+(?:herein\s+)?by\s+reference",
    r"(?:Item\s+7\s+)?(?:information\s+)?(?:is\s+)?incorporated\s+by\s+reference\s+(?:to|from|into)\s+(?:our\s+)?(?:Proxy\s+Statement|Annual\s+Report)",
    r"Management['']?s\s+Discussion\s+and\s+Analysis\s+(?:.*?\s+)?(?:is\s+)?incorporated\s+(?:herein\s+)?by\s+reference",
    r"(?:The\s+)?MD&A\s+(?:section\s+)?(?:is\s+)?incorporated\s+(?:herein\s+)?by\s+reference",

    # Proxy statement incorporation
    r"incorporated\s+by\s+reference\s+(?:to|from)\s+(?:our\s+)?(?:definitive\s+)?Proxy\s+Statement",
    r"(?:see\s+)?(?:our\s+)?Proxy\s+Statement\s+(?:filed\s+)?(?:on\s+Schedule\s+)?DEF\s+14A",
    r"incorporated\s+(?:herein\s+)?by\s+reference\s+(?:to|from)\s+(?:the\s+)?proxy\s+statement",
    r"(?:refer\s+to\s+)?(?:the\s+)?proxy\s+(?:filing|document)\s+(?:for\s+)?(?:Item\s+7\s+)?(?:information|discussion)",

    # Annual report incorporation
    r"incorporated\s+by\s+reference\s+(?:to|from)\s+(?:our\s+)?(?:20\d{2}\s+)?Annual\s+Report\s+(?:to\s+Stockholders|to\s+Shareholders)",
    r"(?:see\s+)?(?:our\s+)?Annual\s+Report\s+(?:to\s+(?:Stock|Share)holders\s+)?(?:for\s+)?(?:Item\s+7\s+)?(?:discussion|information)",
    r"incorporated\s+(?:herein\s+)?by\s+reference\s+(?:to|from)\s+(?:the\s+)?stockholder\s+report",
    r"(?:refer\s+to\s+)?(?:the\s+)?annual\s+stockholder\s+communication",

    # Specific document references with MD&A
    r"(?:Item\s+7\s+)?(?:MD&A\s+)?(?:is\s+)?(?:included\s+in|contained\s+in|found\s+in)\s+Exhibit\s+(\d+(?:\.\d+)?)",
    r"(?:Management['']?s\s+Discussion\s+)?(?:is\s+)?set\s+forth\s+in\s+(?:Exhibit\s+)?(\d+(?:\.\d+)?)",
    r"(?:the\s+)?(?:required\s+)?discussion\s+(?:is\s+)?(?:included\s+in|set\s+forth\s+in)\s+(?:the\s+)?(?:attached\s+)?(?:Exhibit\s+)?(\d+)",

    # Alternative phrasing for incorporation
    r"(?:Item\s+7\s+)?(?:information\s+)?(?:is\s+)?provided\s+by\s+reference\s+(?:to|from)",
    r"(?:the\s+)?(?:MD&A\s+)?(?:content\s+)?(?:is\s+)?supplied\s+by\s+reference",
    r"(?:Item\s+7\s+)?(?:disclosure\s+)?(?:is\s+)?made\s+by\s+reference\s+(?:to|from)",

    # Under caption/heading references
    r"(?:under\s+the\s+heading\s+)?[\"']([^\"']*(?:MD&A|Management|Discussion|Analysis)[^\"']*)[\"']",
    r"(?:see\s+)?(?:caption\s+)?[\"']([^\"']*(?:Financial|Results|Operations)[^\"']*)[\"']\s+(?:in\s+(?:our\s+)?(?:Proxy|Annual\s+Report))"
]


# =============================================================================
# TABLE DETECTION PATTERNS
# =============================================================================
# Patterns for identifying financial tables within the text.
# Tables may use various delimiter styles: pipes, dashes, spaces, tabs.

TABLE_DELIMITER_PATTERNS = [
    r"^\s*[-=]{3,}\s*$",                      # --- or === line
    r"^\s*\|.*\|.*\|",                        # Pipe-delimited
    r"(?:\s{2,}|\t)",                         # Multiple spaces or tabs
    r"^\s*(?:\d+\s+){2,}",                    # Rows of numeric columns
    r"^\s*[A-Za-z]+\s+(?:[-–]\s+)?\$\(?\d",   # Label followed by number (e.g., Revenue - $1,000)
    r"^\s*\(?\$?\d[\d,\.]+\)?\s+(?:\(?\$?\d[\d,\.]+\)?\s+)+$",  # Rows of numeric entries
]


TABLE_HEADER_PATTERNS = [
    r"^\s*(?:Year|Period|Quarter|Month)\s+Ended",                      # "Year Ended December 31"
    r"^\s*(?:December|June|March|September)\s+\d{1,2},?\s+20\d{2}",    # Full date
    r"^\s*\$?\s*(?:in\s+)?(?:thousands|millions|billions)",            # Units
    r"^\s*(?:Revenue|Income|Assets|Liabilities|Equity)",               # Key financial labels
    r"^\s*Statements?\s+of\s+(?:Operations|Cash\s+Flows|Income)",      # "Statement of Operations"
    r"^\s*(?:Unaudited|Audited)\s+Financial\s+Statements?",            # Audit status
    r"^\s*(?:Balance\s+Sheets?|Cash\s+Flows?|Stockholders['’]?\s+Equity)",  # More statement types
    r"^\s*(?:Total|Net|Gross|Operating)\s+(?:Income|Loss|Profit)",     # Descriptive headers
]


# =============================================================================
# SEC DOCUMENT MARKERS TO REMOVE
# =============================================================================
# These patterns identify SEC-specific formatting and metadata that should
# be removed or ignored during text normalization.

SEC_MARKERS = [
    r"<PAGE>\s*\d+",                                 # Page number
    r"Table\s*of\s*Contents",                         # TOC mention
    r"^\s*\d+\s*$",                                   # Bare page numbers
    r"</?[A-Z]+>",                                    # Fake HTML tags
    r"^\s*Form\s+10-?K/A?",                           # Form identifiers
    r"^\s*Filed\s+with\s+the\s+SEC",                  # Filing metadata
    r"^\s*Commission\s+File\s+Number",                # Header block
    r"\b(SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION)\b",# SEC letterhead
    r"\bUNITED\s+STATES\b",                           # SEC letterhead (cont.)
    r"^\s*Index\s*to\s*Financial\s*Statements",       # Indexed tables
    r"^\s*Fiscal\s+Year\s+Ended",                     # Often boilerplate
    r"^\s*\[.*\]$",                                   # Inline flags like [LOGO], [TEXT], etc.
]


# =============================================================================
# INCORPORATION BY REFERENCE PATTERNS
# =============================================================================
# These patterns detect when MD&A content is not present in the filing but
# instead references another document (proxy statement, annual report, exhibit).
# Such filings require special handling to obtain the actual MD&A content.

INCORPORATION_BY_REFERENCE_PATTERNS = [
    # Standard incorporation language
    r"(?:information\s+required\s+by\s+)?Item\s*7.*?(?:is\s+)?incorporated\s+(?:herein\s+)?by\s+reference",
    r"Management['']?s?\s+Discussion\s+and\s+Analysis.*?incorporated\s+by\s+reference",
    r"MD&A.*?incorporated\s+by\s+reference",

    # Reference to proxy statements
    r"incorporated\s+by\s+reference.*?(?:from|to).*?(?:Proxy\s+Statement|DEF\s*14A)",
    r"(?:see|refer\s+to).*?Proxy\s+Statement.*?(?:pages?\s+[\d\-A-Z]+|Appendix)",

    # Reference to exhibits
    r"incorporated\s+by\s+reference.*?Exhibit\s*(?:13|99|[\d\.]+)",
    r"(?:see|refer\s+to).*?Exhibit\s*(?:13|99|[\d\.]+).*?(?:Annual\s+Report|10-K)",

    # Reference to appendices
    r"(?:see|refer\s+to).*?Appendix\s*[A-Z]?.*?(?:pages?\s+[\d\-A-Z]+)?",
    r"incorporated.*?from.*?Appendix",

    # Caption references
    r"under\s+(?:the\s+)?caption\s+[\"']([^\"']+)[\"']",
    r"(?:section|item)\s+(?:entitled|titled)\s+[\"']([^\"']+)[\"'].*?incorporated",

    # Page references
    r"(?:on\s+)?pages?\s+([\d\-A-Z]+(?:\s+through\s+[\d\-A-Z]+)?)",

    # General incorporation phrases
    r"information.*?set\s+forth.*?incorporated\s+by\s+reference",
    r"hereby\s+incorporated\s+by\s+reference",
]

# =============================================================================
# PATTERN COMPILATION
# =============================================================================
# Patterns are compiled at module load time for optimal performance.
# Compiled patterns are stored in COMPILED_PATTERNS dictionary.

def compile_patterns():
    """
    Compile all regex patterns for better performance.

    Pre-compiling patterns with appropriate flags (IGNORECASE, MULTILINE)
    provides significant performance improvements when patterns are used
    repeatedly during processing of large documents.

    Returns:
        Dictionary mapping pattern names to lists of compiled regex objects
    """
    compiled = {
        "item_7_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ITEM_7_START_PATTERNS],
        "item_7a_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ITEM_7A_START_PATTERNS],
        "item_8_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ITEM_8_START_PATTERNS],
        "item_2_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ITEM_2_START_PATTERNS],
        "item_3_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ITEM_3_START_PATTERNS],
        "item_4_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ITEM_4_START_PATTERNS],
        "part_ii_start": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PART_II_START_PATTERNS],
        "form_type": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in FORM_TYPE_PATTERNS],
        "cross_reference": [re.compile(p, re.IGNORECASE) for p in CROSS_REFERENCE_PATTERNS],
        "table_delimiter": [re.compile(p, re.MULTILINE) for p in TABLE_DELIMITER_PATTERNS],
        "table_header": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in TABLE_HEADER_PATTERNS],
        "sec_markers": [re.compile(p, re.MULTILINE) for p in SEC_MARKERS],
        "incorporation_by_reference": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INCORPORATION_BY_REFERENCE_PATTERNS],
    }
    return compiled

COMPILED_PATTERNS = compile_patterns()