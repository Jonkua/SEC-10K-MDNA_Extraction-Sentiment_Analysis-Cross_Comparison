"""
Debug test to identify exactly where Item 7 detection is failing.
Run this as a standalone script to diagnose the issue.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.patterns import ITEM_7_START_PATTERNS, compile_patterns
from src.parsers.section_parser import SectionParser, SectionBoundary


class Item7DebugTester:
    """Debug tester for Item 7 detection issues."""

    def __init__(self):
        self.section_parser = SectionParser()
        self.patterns = compile_patterns()

    def test_item7_detection(self, file_path: str):
        """
        Comprehensive test of Item 7 detection on a specific file.
        """
        print("=" * 80)
        print(f"DEBUGGING ITEM 7 DETECTION FOR: {file_path}")
        print("=" * 80)

        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"ERROR reading file: {e}")
            return

        print(f"File size: {len(content):,} characters")
        print(f"File lines: {content.count(chr(10)):,} lines")
        print()

        # Test 1: Check if any Item 7 patterns exist in the raw content
        print("TEST 1: Raw Pattern Matching")
        print("-" * 40)
        self._test_raw_patterns(content)
        print()

        # Test 2: Show actual content around potential matches
        print("TEST 2: Content Analysis")
        print("-" * 40)
        self._analyze_content_structure(content)
        print()

        # Test 3: Test the compiled patterns specifically
        print("TEST 3: Compiled Pattern Testing")
        print("-" * 40)
        self._test_compiled_patterns(content)
        print()

        # Test 4: Test the section parser's find_all_section_matches
        print("TEST 4: Section Parser Testing")
        print("-" * 40)
        self._test_section_parser(content)
        print()

        # Test 5: Test TOC filtering
        print("TEST 5: TOC Filter Analysis")
        print("-" * 40)
        self._test_toc_filtering(content)
        print()

    def _test_raw_patterns(self, content: str):
        """Test raw regex patterns against content."""
        print("Testing individual Item 7 patterns...")

        for i, pattern_str in enumerate(ITEM_7_START_PATTERNS[:10]):  # Test first 10
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                matches = list(pattern.finditer(content))

                if matches:
                    print(f"✅ Pattern {i + 1}: Found {len(matches)} matches")
                    for j, match in enumerate(matches[:3]):  # Show first 3 matches
                        start = match.start()
                        line_num = content[:start].count('\n') + 1
                        context = self._get_context(content, start, 100)
                        print(f"   Match {j + 1}: Line {line_num}, Pos {start}")
                        print(f"   Context: {repr(context)}")
                else:
                    print(f"❌ Pattern {i + 1}: No matches")

            except Exception as e:
                print(f"❌ Pattern {i + 1}: ERROR - {e}")

        print(f"\nTested {min(10, len(ITEM_7_START_PATTERNS))} patterns")

    def _analyze_content_structure(self, content: str):
        """Analyze the structure of the content to understand format."""
        # Look for common Item 7 text variations
        item7_variations = [
            "ITEM 7",
            "Item 7",
            "ITEM SEVEN",
            "Item Seven",
            "MANAGEMENT'S DISCUSSION",
            "MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "MD&A",
            "DISCUSSION AND ANALYSIS"
        ]

        print("Searching for Item 7 related text...")
        for variation in item7_variations:
            # Case insensitive search
            pattern = re.compile(re.escape(variation), re.IGNORECASE)
            matches = list(pattern.finditer(content))

            if matches:
                print(f"✅ Found '{variation}': {len(matches)} occurrences")
                for i, match in enumerate(matches[:3]):  # Show first 3
                    start = match.start()
                    line_num = content[:start].count('\n') + 1
                    context = self._get_context(content, start, 150)
                    print(f"   Occurrence {i + 1}: Line {line_num}")
                    print(f"   Context: {repr(context)}")
            else:
                print(f"❌ '{variation}': Not found")

    def _test_compiled_patterns(self, content: str):
        """Test the compiled patterns from config/patterns.py."""
        if "item_7_start" not in self.patterns:
            print("❌ ERROR: 'item_7_start' not found in compiled patterns!")
            return

        patterns = self.patterns["item_7_start"]
        print(f"Testing {len(patterns)} compiled Item 7 patterns...")

        total_matches = 0
        for i, pattern in enumerate(patterns):
            matches = list(pattern.finditer(content))
            if matches:
                print(f"✅ Compiled Pattern {i + 1}: {len(matches)} matches")
                total_matches += len(matches)
                for j, match in enumerate(matches[:2]):  # Show first 2
                    start = match.start()
                    line_num = content[:start].count('\n') + 1
                    context = self._get_context(content, start, 100)
                    print(f"   Match {j + 1}: Line {line_num}, Pos {start}")
                    print(f"   Context: {repr(context)}")
            else:
                print(f"❌ Compiled Pattern {i + 1}: No matches")

        print(f"\nTotal matches from compiled patterns: {total_matches}")

    def _test_section_parser(self, content: str):
        """Test the section parser's methods."""
        print("Testing SectionParser._find_all_section_matches...")

        try:
            all_matches = self.section_parser._find_all_section_matches(content, "item_7_start")

            if all_matches:
                print(f"✅ _find_all_section_matches found {len(all_matches)} matches")
                for i, match in enumerate(all_matches):
                    context = self._get_context(content, match.start_pos, 100)
                    print(
                        f"   Match {i + 1}: Line {match.line_number}, Pos {match.start_pos}, Confidence {match.confidence}")
                    print(f"   Pattern: {match.pattern_matched[:100]}...")
                    print(f"   Context: {repr(context)}")
            else:
                print("❌ _find_all_section_matches found NO matches")

        except Exception as e:
            print(f"❌ ERROR in _find_all_section_matches: {e}")
            import traceback
            traceback.print_exc()

    def _test_toc_filtering(self, content: str):
        """Test the TOC filtering logic."""
        print("Testing TOC filtering...")

        try:
            # Get all matches first
            all_matches = self.section_parser._find_all_section_matches(content, "item_7_start")

            if not all_matches:
                print("❌ No matches to test TOC filtering on")
                return

            print(f"Before TOC filtering: {len(all_matches)} matches")

            # Test filtering with different KB thresholds
            for min_kb in [0, 5, 10, 15, 20]:
                try:
                    valid_match = self.section_parser._filter_toc_matches(
                        all_matches.copy(), content, min_position_kb=min_kb
                    )

                    if valid_match:
                        print(f"✅ With {min_kb}KB threshold: Found valid match at line {valid_match.line_number}")
                        context = self._get_context(content, valid_match.start_pos, 100)
                        print(f"   Context: {repr(context)}")
                    else:
                        print(f"❌ With {min_kb}KB threshold: No valid match")

                except Exception as e:
                    print(f"❌ ERROR with {min_kb}KB threshold: {e}")

        except Exception as e:
            print(f"❌ ERROR in TOC filtering test: {e}")

    def _get_context(self, content: str, position: int, context_length: int = 100) -> str:
        """Get context around a position."""
        start = max(0, position - context_length // 2)
        end = min(len(content), position + context_length // 2)
        return content[start:end].replace('\n', '\\n').replace('\t', '\\t')

    def test_sample_content(self):
        """Test with known sample content that should work."""
        print("=" * 80)
        print("TESTING WITH SAMPLE CONTENT")
        print("=" * 80)

        sample_content = """
FORM 10-K

CENTRAL INDEX KEY: 0001234567

Some preamble content here...

Table of Contents

PART I
Item 1. Business ................................................... 5
Item 7. Management's Discussion and Analysis ...................... 25
Item 8. Financial Statements ...................................... 45

PART I

ITEM 1. BUSINESS

Lots of business content here...

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

This is the actual MD&A content that we want to extract.

Revenue increased 15% compared to prior year...

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

Market risk content here...
"""

        print("Testing sample content...")
        self._test_compiled_patterns(sample_content)

        # Test full pipeline
        try:
            bounds = self.section_parser.find_mdna_section(sample_content, "10-K")
            if bounds:
                start, end = bounds
                extracted = sample_content[start:end]
                print(f"✅ Full pipeline SUCCESS: Extracted {len(extracted)} characters")
                print(f"   Extracted content preview: {repr(extracted[:200])}...")
            else:
                print("❌ Full pipeline FAILED: No MD&A section found")
        except Exception as e:
            print(f"❌ Full pipeline ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the debug test."""
    if len(sys.argv) < 2:
        print("Usage: python debug_item7.py <path_to_filing>")
        print("   or: python debug_item7.py --sample")
        return

    tester = Item7DebugTester()

    if sys.argv[1] == "--sample":
        tester.test_sample_content()
    else:
        file_path = sys.argv[1]
        if not Path(file_path).exists():
            print(f"ERROR: File does not exist: {file_path}")
            return
        tester.test_item7_detection(file_path)


if __name__ == "__main__":
    main()