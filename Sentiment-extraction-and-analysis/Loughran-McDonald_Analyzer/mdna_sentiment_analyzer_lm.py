# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports for file handling, text processing, and utilities
import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
from datetime import datetime
import multiprocessing as mp
import logging

# Natural language processing libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Data manipulation
import pandas as pd

# ============================================================================
# NLTK DATA DOWNLOADS
# ============================================================================
# Download required NLTK datasets if not already present
# These are needed for tokenization and stopword removal

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# ============================================================================
# LOUGHRAN-MCDONALD DICTIONARY LOADER
# ============================================================================

def load_lm_dictionary(csv_path: str) -> Dict[str, Set[str]]:
    """
    Load the Loughran-McDonald Master Dictionary from CSV file.

    The CSV contains columns with binary indicators (0 or 1) for each sentiment category.
    Words with a value of 1 in a category column belong to that category.

    Args:
        csv_path: Path to the LM Master Dictionary CSV file

    Returns:
        Dictionary with keys for each sentiment category, values are sets of words

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"LM Dictionary file not found: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_columns = ['Word', 'Negative', 'Positive', 'Uncertainty', 'Litigious',
                        'Strong_Modal', 'Weak_Modal', 'Constraining']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    # Convert words to lowercase for case-insensitive matching
    df['Word'] = df['Word'].str.lower()

    # Create sets for each category by filtering rows where the category column equals 1
    lexicons = {
        'negative': set(df[df['Negative'] > 0]['Word'].tolist()),
        'positive': set(df[df['Positive'] > 0]['Word'].tolist()),
        'uncertainty': set(df[df['Uncertainty'] > 0]['Word'].tolist()),
        'litigious': set(df[df['Litigious'] > 0]['Word'].tolist()),
        'strong_modal': set(df[df['Strong_Modal'] > 0]['Word'].tolist()),
        'weak_modal': set(df[df['Weak_Modal'] > 0]['Word'].tolist()),
        'constraining': set(df[df['Constraining'] > 0]['Word'].tolist())
    }

    return lexicons


# ============================================================================
# TEMPORAL INDICATOR WORD LISTS
# ============================================================================
# These are not part of the LM dictionary but are used to classify sentence timeframes

# FORWARD_LOOKING: Terms indicating future plans or projections
FORWARD_LOOKING = {
    'anticipate', 'anticipated', 'anticipates', 'believe', 'believes', 'continue',
    'continues', 'could', 'estimate', 'estimated', 'estimates', 'expect',
    'expected', 'expects', 'forecast', 'forecasts', 'forward', 'future', 'goal',
    'goals', 'guidance', 'intend', 'intended', 'intends', 'may', 'objective',
    'objectives', 'outlook', 'plan', 'planned', 'plans', 'predict', 'predicted',
    'predicts', 'project', 'projected', 'projection', 'projections', 'projects',
    'prospect', 'prospects', 'seek', 'seeking', 'should', 'target', 'targets',
    'will', 'would',
}

# HISTORICAL_INDICATORS: Terms indicating past events or completed actions
HISTORICAL_INDICATORS = {
    'achieved', 'acquired', 'announced', 'completed', 'decreased', 'delivered',
    'ended', 'experienced', 'generated', 'grew', 'had', 'has', 'have', 'implemented',
    'incurred', 'increased', 'launched', 'made', 'occurred', 'paid', 'performed',
    'produced', 'realized', 'received', 'recorded', 'reduced', 'reported', 'was',
    'were',
}


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class MDNASentimentAnalyzer:
    """
    Comprehensive sentiment and tone analyzer for 10-K MD&A sections.

    This class analyzes Management's Discussion and Analysis (MD&A) sections
    from 10-K filings using multiple sentiment analysis techniques including:
    - Lexicon-based analysis (Loughran-McDonald financial dictionary)
    - Polarity/subjectivity scoring (TextBlob)
    - Temporal classification (forward-looking vs. historical statements)
    """

    def __init__(self, lm_dict_path: str):
        """
        Initialize the analyzer with LM dictionary and stopwords.

        Args:
            lm_dict_path: Path to the Loughran-McDonald Master Dictionary CSV
        """
        self.stop_words = set(stopwords.words('english'))
        self.logger = self._setup_logger()

        # Load the full LM dictionary from CSV
        self.logger.info(f"Loading Loughran-McDonald dictionary from: {lm_dict_path}")
        self.lexicons = load_lm_dictionary(lm_dict_path)

        # Log dictionary statistics
        self.logger.info(f"Loaded LM Dictionary:")
        self.logger.info(f"  - Negative words: {len(self.lexicons['negative'])}")
        self.logger.info(f"  - Positive words: {len(self.lexicons['positive'])}")
        self.logger.info(f"  - Uncertainty words: {len(self.lexicons['uncertainty'])}")
        self.logger.info(f"  - Litigious words: {len(self.lexicons['litigious'])}")
        self.logger.info(f"  - Strong Modal words: {len(self.lexicons['strong_modal'])}")
        self.logger.info(f"  - Weak Modal words: {len(self.lexicons['weak_modal'])}")
        self.logger.info(f"  - Constraining words: {len(self.lexicons['constraining'])}")

    def _setup_logger(self):
        """
        Setup logging configuration for tracking analysis progress.

        Returns:
            logger: Configured logging object for console output
        """
        logger = logging.getLogger('MDNAAnalyzer')
        logger.setLevel(logging.INFO)

        # Only add handler if none exists (prevents duplicate handlers)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse filename to extract metadata from standardized SEC filing format.

        Expected format: (CIK)_(CO_NAME)_(FILING_DATE)_(FORM_TYPE).txt
        Example: (0000002488)_(ADVANCED MICRO DEVICES INC)_(2020-02-04)_(10-K).txt

        Args:
            filename: Name of the file to parse

        Returns:
            Dictionary containing CIK, company name, filing date, form type, and year
        """
        name = Path(filename).stem

        # Regex pattern to extract components enclosed in parentheses
        pattern = r'\(([^)]+)\)_\(([^)]+)\)_\(([^)]+)\)_\(([^)]+)\)'
        match = re.match(pattern, name)

        if match:
            cik, company_name, filing_date, form_type = match.groups()
            # Extract year from filing date (YYYY-MM-DD format)
            year = filing_date.split('-')[0] if '-' in filing_date else 'UNKNOWN'
        else:
            # Fallback values if pattern doesn't match
            cik = 'UNKNOWN'
            company_name = 'UNKNOWN'
            filing_date = 'UNKNOWN'
            form_type = 'UNKNOWN'
            year = 'UNKNOWN'

        return {
            'cik': cik,
            'company_name': company_name,
            'filing_date': filing_date,
            'form_type': form_type,
            'year': year,
            'filename': filename
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for analysis.

        - Removes excessive whitespace
        - Removes special characters while preserving sentence structure

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        # Collapse multiple whitespace characters into single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation for sentence boundaries
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        return text.strip()

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into individual words and remove stopwords.

        Args:
            text: Text to tokenize

        Returns:
            List of cleaned, alphabetic word tokens (lowercase, no stopwords)
        """
        words = word_tokenize(text.lower())
        # Filter to keep only alphabetic words that aren't stopwords
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        return words

    def count_lexicon_words(self, words: List[str], lexicon: Set[str]) -> Tuple[int, List[str]]:
        """
        Count how many words from the text appear in a specific lexicon.

        Args:
            words: List of words from the document
            lexicon: Set of words to match against

        Returns:
            Tuple of (count, list of matching words)
        """
        matches = [w for w in words if w in lexicon]
        return len(matches), matches

    def classify_sentences(self, text: str) -> Dict[str, List[str]]:
        """
        Classify sentences as forward-looking or historical based on indicator words.

        Forward-looking sentences discuss future plans, projections, or expectations.
        Historical sentences describe past events or completed actions.

        Args:
            text: Full text to analyze

        Returns:
            Dictionary with 'forward' and 'historical' lists of sentences
        """
        sentences = sent_tokenize(text)

        forward_sentences = []
        historical_sentences = []

        for sent in sentences:
            sent_lower = sent.lower()
            words = set(word_tokenize(sent_lower))

            # Count how many forward-looking vs. historical indicator words appear
            forward_score = len(words & FORWARD_LOOKING)
            historical_score = len(words & HISTORICAL_INDICATORS)

            # Classify based on which type of indicator is more prevalent
            if forward_score > historical_score and forward_score > 0:
                forward_sentences.append(sent)
            elif historical_score > 0:
                historical_sentences.append(sent)

        return {
            'forward': forward_sentences,
            'historical': historical_sentences
        }

    def analyze_sentiment_by_timeframe(self, sentences: Dict[str, List[str]]) -> Dict:
        """
        Analyze sentiment separately for forward-looking vs. historical statements.

        Uses TextBlob to calculate polarity (positive/negative) and subjectivity
        (factual/opinion) for each timeframe category.

        Args:
            sentences: Dictionary with 'forward' and 'historical' sentence lists

        Returns:
            Dictionary with sentiment metrics for each timeframe
        """
        results = {}

        for timeframe, sent_list in sentences.items():
            if not sent_list:
                # No sentences of this type found
                results[timeframe] = {
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'sentence_count': 0
                }
                continue

            # Combine all sentences of this timeframe and analyze as a whole
            combined_text = ' '.join(sent_list)
            blob = TextBlob(combined_text)

            results[timeframe] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentence_count': len(sent_list)
            }

        return results

    def calculate_sentiment_scores(self, words: List[str]) -> Dict:
        """
        Calculate comprehensive sentiment scores using Loughran-McDonald lexicons.

        Provides both raw counts and proportions of different sentiment categories:
        - Negative words (problems, risks)
        - Positive words (achievements, growth)
        - Uncertainty words (ambiguity, estimates)
        - Litigious words (legal issues)
        - Strong Modal words (strong tone)
        - Weak Modal words (weak tone)
        - Constraining words (limiting language)

        Args:
            words: List of tokenized words from the document

        Returns:
            Dictionary containing counts, proportions, and the actual matching words
        """
        total_words = len(words)

        # Count words matching each lexicon category
        neg_count, neg_words = self.count_lexicon_words(words, self.lexicons['negative'])
        pos_count, pos_words = self.count_lexicon_words(words, self.lexicons['positive'])
        unc_count, unc_words = self.count_lexicon_words(words, self.lexicons['uncertainty'])
        lit_count, lit_words = self.count_lexicon_words(words, self.lexicons['litigious'])
        strong_count, strong_words = self.count_lexicon_words(words, self.lexicons['strong_modal'])
        weak_count, weak_words = self.count_lexicon_words(words, self.lexicons['weak_modal'])
        con_count, con_words = self.count_lexicon_words(words, self.lexicons['constraining'])

        # Calculate proportions (density of sentiment words)
        neg_proportion = neg_count / total_words if total_words > 0 else 0
        pos_proportion = pos_count / total_words if total_words > 0 else 0
        unc_proportion = unc_count / total_words if total_words > 0 else 0
        lit_proportion = lit_count / total_words if total_words > 0 else 0
        strong_proportion = strong_count / total_words if total_words > 0 else 0
        weak_proportion = weak_count / total_words if total_words > 0 else 0
        con_proportion = con_count / total_words if total_words > 0 else 0

        # Calculate net sentiment (positive minus negative)
        net_sentiment = pos_count - neg_count
        # Calculate sentiment ratio (positive divided by negative)
        sentiment_ratio = pos_count / neg_count if neg_count > 0 else float('inf')

        return {
            'total_words': total_words,
            'negative_count': neg_count,
            'positive_count': pos_count,
            'uncertainty_count': unc_count,
            'litigious_count': lit_count,
            'strong_modal_count': strong_count,
            'weak_modal_count': weak_count,
            'constraining_count': con_count,
            'negative_proportion': round(neg_proportion, 4),
            'positive_proportion': round(pos_proportion, 4),
            'uncertainty_proportion': round(unc_proportion, 4),
            'litigious_proportion': round(lit_proportion, 4),
            'strong_modal_proportion': round(strong_proportion, 4),
            'weak_modal_proportion': round(weak_proportion, 4),
            'constraining_proportion': round(con_proportion, 4),
            'net_sentiment': net_sentiment,
            'sentiment_ratio': round(sentiment_ratio, 2) if sentiment_ratio != float('inf') else None,
            'negative_words': neg_words,
            'positive_words': pos_words,
            'uncertainty_words': unc_words,
            'litigious_words': lit_words,
            'strong_modal_words': strong_words,
            'weak_modal_words': weak_words,
            'constraining_words': con_words
        }

    def analyze_document(self, file_path: str) -> Dict:
        """
        Perform comprehensive sentiment analysis on a single MD&A document.

        This is the main analysis method that coordinates all the other methods
        to produce a complete analysis result for one document.

        Args:
            file_path: Path to the text file to analyze

        Returns:
            Dictionary containing all analysis results, or None if error occurs
        """
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # Extract metadata from filename
            metadata = self.parse_filename(os.path.basename(file_path))

            # Clean and prepare text
            cleaned_text = self.clean_text(text)

            # Tokenize into words
            words = self.tokenize_words(cleaned_text)

            # Calculate overall sentiment using TextBlob
            blob = TextBlob(cleaned_text)
            overall_polarity = blob.sentiment.polarity
            overall_subjectivity = blob.sentiment.subjectivity

            # Calculate lexicon-based sentiment scores
            sentiment_scores = self.calculate_sentiment_scores(words)

            # Classify sentences by temporal orientation
            sentence_classification = self.classify_sentences(cleaned_text)

            # Analyze sentiment by timeframe (forward vs. historical)
            timeframe_sentiment = self.analyze_sentiment_by_timeframe(sentence_classification)

            # Get frequency counts of most common sentiment words (top 20 for each category)
            neg_word_freq = Counter(sentiment_scores['negative_words']).most_common(20)
            pos_word_freq = Counter(sentiment_scores['positive_words']).most_common(20)
            unc_word_freq = Counter(sentiment_scores['uncertainty_words']).most_common(20)
            lit_word_freq = Counter(sentiment_scores['litigious_words']).most_common(20)
            strong_word_freq = Counter(sentiment_scores['strong_modal_words']).most_common(20)
            weak_word_freq = Counter(sentiment_scores['weak_modal_words']).most_common(20)
            con_word_freq = Counter(sentiment_scores['constraining_words']).most_common(20)

            # Compile all results into structured output
            results = {
                'metadata': metadata,
                'overall_sentiment': {
                    'polarity': round(overall_polarity, 4),
                    'subjectivity': round(overall_subjectivity, 4)
                },
                'lexicon_analysis': {
                    'total_words': sentiment_scores['total_words'],
                    'negative_count': sentiment_scores['negative_count'],
                    'positive_count': sentiment_scores['positive_count'],
                    'uncertainty_count': sentiment_scores['uncertainty_count'],
                    'litigious_count': sentiment_scores['litigious_count'],
                    'strong_modal_count': sentiment_scores['strong_modal_count'],
                    'weak_modal_count': sentiment_scores['weak_modal_count'],
                    'constraining_count': sentiment_scores['constraining_count'],
                    'negative_proportion': sentiment_scores['negative_proportion'],
                    'positive_proportion': sentiment_scores['positive_proportion'],
                    'uncertainty_proportion': sentiment_scores['uncertainty_proportion'],
                    'litigious_proportion': sentiment_scores['litigious_proportion'],
                    'strong_modal_proportion': sentiment_scores['strong_modal_proportion'],
                    'weak_modal_proportion': sentiment_scores['weak_modal_proportion'],
                    'constraining_proportion': sentiment_scores['constraining_proportion'],
                    'net_sentiment': sentiment_scores['net_sentiment'],
                    'sentiment_ratio': sentiment_scores['sentiment_ratio']
                },
                'timeframe_analysis': {
                    'forward_looking': {
                        'sentence_count': timeframe_sentiment['forward']['sentence_count'],
                        'polarity': round(timeframe_sentiment['forward']['polarity'], 4),
                        'subjectivity': round(timeframe_sentiment['forward']['subjectivity'], 4)
                    },
                    'historical': {
                        'sentence_count': timeframe_sentiment['historical']['sentence_count'],
                        'polarity': round(timeframe_sentiment['historical']['polarity'], 4),
                        'subjectivity': round(timeframe_sentiment['historical']['subjectivity'], 4)
                    }
                },
                'word_frequencies': {
                    'negative_words': dict(neg_word_freq),
                    'positive_words': dict(pos_word_freq),
                    'uncertainty_words': dict(unc_word_freq),
                    'litigious_words': dict(lit_word_freq),
                    'strong_modal_words': dict(strong_word_freq),
                    'weak_modal_words': dict(weak_word_freq),
                    'constraining_words': dict(con_word_freq)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            return None

    def process_batch(self, file_paths: List[str], num_processes: int = None) -> List[Dict]:
        """
        Process multiple files in parallel using multiprocessing for speed.

        Args:
            file_paths: List of file paths to analyze
            num_processes: Number of parallel processes (defaults to CPU count - 1)

        Returns:
            List of analysis result dictionaries
        """
        if num_processes is None:
            # Use all CPUs except one to avoid overloading the system
            num_processes = max(1, mp.cpu_count() - 1)

        self.logger.info(f"Starting batch processing of {len(file_paths)} files using {num_processes} processes")

        # Create process pool and distribute work
        with mp.Pool(processes=num_processes) as pool:
            results = []
            total = len(file_paths)

            # Process files and track progress
            for i, result in enumerate(pool.imap(self.analyze_document, file_paths), 1):
                if result is not None:
                    results.append(result)

                # Log progress every 10 files or at completion
                if i % 10 == 0 or i == total:
                    self.logger.info(f"Progress: {i}/{total} files processed ({i / total * 100:.1f}%)")

        self.logger.info(f"Batch processing complete. Successfully analyzed {len(results)} files")
        return results

    def save_results(self, results: List[Dict], output_dir: str):
        """
        Save analysis results in both CSV (summary) and JSON (detailed) formats.

        Args:
            results: List of analysis result dictionaries
            output_dir: Directory where output files should be saved
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed JSON with all analysis data
        json_path = os.path.join(output_dir, f'sentiment_analysis_detailed_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Detailed JSON saved to: {json_path}")

        # Save summary CSV with key metrics only
        csv_path = os.path.join(output_dir, f'sentiment_analysis_summary_{timestamp}.csv')

        # Define CSV columns for the summary (now including new categories)
        csv_fields = [
            'cik', 'company_name', 'filing_date', 'form_type', 'year', 'filename',
            'overall_polarity', 'overall_subjectivity',
            'total_words', 'negative_count', 'positive_count',
            'uncertainty_count', 'litigious_count',
            'strong_modal_count', 'weak_modal_count', 'constraining_count',
            'negative_proportion', 'positive_proportion',
            'uncertainty_proportion', 'litigious_proportion',
            'strong_modal_proportion', 'weak_modal_proportion', 'constraining_proportion',
            'net_sentiment', 'sentiment_ratio',
            'forward_sentence_count', 'forward_polarity', 'forward_subjectivity',
            'historical_sentence_count', 'historical_polarity', 'historical_subjectivity',
            'analysis_timestamp'
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

            # Flatten nested results structure for CSV format
            for result in results:
                row = {
                    'cik': result['metadata']['cik'],
                    'company_name': result['metadata']['company_name'],
                    'filing_date': result['metadata']['filing_date'],
                    'form_type': result['metadata']['form_type'],
                    'year': result['metadata']['year'],
                    'filename': result['metadata']['filename'],
                    'overall_polarity': result['overall_sentiment']['polarity'],
                    'overall_subjectivity': result['overall_sentiment']['subjectivity'],
                    'total_words': result['lexicon_analysis']['total_words'],
                    'negative_count': result['lexicon_analysis']['negative_count'],
                    'positive_count': result['lexicon_analysis']['positive_count'],
                    'uncertainty_count': result['lexicon_analysis']['uncertainty_count'],
                    'litigious_count': result['lexicon_analysis']['litigious_count'],
                    'strong_modal_count': result['lexicon_analysis']['strong_modal_count'],
                    'weak_modal_count': result['lexicon_analysis']['weak_modal_count'],
                    'constraining_count': result['lexicon_analysis']['constraining_count'],
                    'negative_proportion': result['lexicon_analysis']['negative_proportion'],
                    'positive_proportion': result['lexicon_analysis']['positive_proportion'],
                    'uncertainty_proportion': result['lexicon_analysis']['uncertainty_proportion'],
                    'litigious_proportion': result['lexicon_analysis']['litigious_proportion'],
                    'strong_modal_proportion': result['lexicon_analysis']['strong_modal_proportion'],
                    'weak_modal_proportion': result['lexicon_analysis']['weak_modal_proportion'],
                    'constraining_proportion': result['lexicon_analysis']['constraining_proportion'],
                    'net_sentiment': result['lexicon_analysis']['net_sentiment'],
                    'sentiment_ratio': result['lexicon_analysis']['sentiment_ratio'],
                    'forward_sentence_count': result['timeframe_analysis']['forward_looking']['sentence_count'],
                    'forward_polarity': result['timeframe_analysis']['forward_looking']['polarity'],
                    'forward_subjectivity': result['timeframe_analysis']['forward_looking']['subjectivity'],
                    'historical_sentence_count': result['timeframe_analysis']['historical']['sentence_count'],
                    'historical_polarity': result['timeframe_analysis']['historical']['polarity'],
                    'historical_subjectivity': result['timeframe_analysis']['historical']['subjectivity'],
                    'analysis_timestamp': result['analysis_timestamp']
                }
                writer.writerow(row)

        self.logger.info(f"Summary CSV saved to: {csv_path}")

    def find_all_files(self, root_dir: str) -> List[str]:
        """
        Recursively find all .txt files in the directory structure.

        Args:
            root_dir: Root directory to search

        Returns:
            List of full file paths to all .txt files found
        """
        file_paths = []
        all_files_found = []

        # Walk through directory tree
        for root, dirs, files in os.walk(root_dir):
            self.logger.info(f"Scanning directory: {root}")
            self.logger.info(f"  Subdirectories: {dirs}")
            self.logger.info(f"  Files found: {len(files)}")

            for file in files:
                all_files_found.append(file)
                # Case-insensitive matching for .txt extension
                if file.lower().endswith('.txt'):
                    file_paths.append(os.path.join(root, file))

        # Warning if no .txt files found but other files exist
        if not file_paths and all_files_found:
            self.logger.warning(f"No .txt files found, but found {len(all_files_found)} other files")
            self.logger.warning(f"Sample files: {all_files_found[:5]}")

        self.logger.info(f"Found {len(file_paths)} .txt files in {root_dir}")
        return file_paths


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function with command-line argument parsing.

    Handles:
    - Command-line argument parsing
    - Input validation
    - Orchestrating the analysis workflow
    - Error handling
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Perform comprehensive sentiment and tone analysis on 10-K MD&A documents using the full Loughran-McDonald dictionary.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python mdna_sentiment_analyzer.py -i /path/to/mdna/files -o /path/to/results -d LoughranMcDonald_MasterDictionary_1993-2024.csv
  python mdna_sentiment_analyzer.py -i ./data -o ./output -d ./LM_dict.csv -p 4
        '''
    )

    # Define command-line arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing MD&A text files (organized in year folders)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for analysis results (CSV and JSON files)'
    )

    parser.add_argument(
        '-d', '--dictionary',
        type=str,
        required=True,
        help='Path to Loughran-McDonald Master Dictionary CSV file'
    )

    parser.add_argument(
        '-p', '--processes',
        type=int,
        default=None,
        help='Number of parallel processes to use (default: CPU count - 1)'
    )

    # Parse arguments from command line
    args = parser.parse_args()

    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    LM_DICT_PATH = args.dictionary
    NUM_PROCESSES = args.processes

    # Validate input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    if not os.path.isdir(INPUT_DIR):
        print(f"Error: '{INPUT_DIR}' is not a directory.")
        return

    # Validate LM dictionary file exists
    if not os.path.exists(LM_DICT_PATH):
        print(f"Error: LM Dictionary file '{LM_DICT_PATH}' does not exist.")
        return

    # Initialize analyzer with LM dictionary
    try:
        analyzer = MDNASentimentAnalyzer(lm_dict_path=LM_DICT_PATH)
    except Exception as e:
        print(f"Error initializing analyzer: {str(e)}")
        return

    # Log configuration
    analyzer.logger.info(f"Input directory: {INPUT_DIR}")
    analyzer.logger.info(f"Output directory: {OUTPUT_DIR}")
    analyzer.logger.info(f"LM Dictionary: {LM_DICT_PATH}")
    if NUM_PROCESSES:
        analyzer.logger.info(f"Using {NUM_PROCESSES} processes")
    else:
        analyzer.logger.info(f"Using automatic process count (CPU count - 1)")

    # Find all .txt files in the input directory
    file_paths = analyzer.find_all_files(INPUT_DIR)

    if not file_paths:
        analyzer.logger.warning("No .txt files found. Please check your input directory.")
        return

    # Process all files using multiprocessing
    results = analyzer.process_batch(file_paths, num_processes=NUM_PROCESSES)

    # Save results to output directory
    if results:
        analyzer.save_results(results, OUTPUT_DIR)
        analyzer.logger.info("Analysis complete!")
    else:
        analyzer.logger.warning("No results to save.")


# Entry point - runs when script is executed directly
if __name__ == '__main__':
    main()