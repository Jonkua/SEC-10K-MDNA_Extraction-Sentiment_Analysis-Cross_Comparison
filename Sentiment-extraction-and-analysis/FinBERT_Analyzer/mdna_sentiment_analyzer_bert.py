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
from typing import Dict, List
from collections import Counter
from datetime import datetime
import multiprocessing as mp
import logging

# Natural language processing libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Data manipulation
import pandas as pd

# FinBERT for transformer-based sentiment analysis
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

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
# FINBERT ANALYZER CLASS
# ============================================================================

class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.

    Uses the ProsusAI/finbert model fine-tuned on financial phrasebank.
    Provides three-class sentiment: positive, negative, neutral.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT model and tokenizer.

        Args:
            model_name: HuggingFace model identifier (default: ProsusAI/finbert)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained FinBERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # FinBERT outputs: positive, negative, neutral
        self.labels = ['positive', 'negative', 'neutral']

    def analyze_text(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Analyze sentiment of a text string.

        Args:
            text: Input text to analyze
            max_length: Maximum token length (BERT limit is 512)

        Returns:
            Dictionary with sentiment scores and predicted label
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = softmax(outputs.logits, dim=-1)

        # Extract probabilities
        probs = predictions[0].cpu().numpy()

        return {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'sentiment': self.labels[probs.argmax()],
            'confidence': float(probs.max())
        }

    def analyze_sentences(self, sentences: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for ALL sentences in a list (NO SAMPLING).

        Args:
            sentences: List of sentence strings

        Returns:
            List of sentiment dictionaries for every sentence
        """
        # Process ALL sentences - no sampling
        return [self.analyze_text(sent) for sent in sentences]

    def analyze_sentences_batch(self, sentences: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Analyze sentiment for ALL sentences using batch processing (NO SAMPLING).
        More efficient than one-by-one processing.

        Args:
            sentences: List of ALL sentence strings
            batch_size: Number of sentences to process in each batch

        Returns:
            List of sentiment dictionaries for every sentence
        """
        results = []

        # Process ALL sentences in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions for batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = softmax(outputs.logits, dim=-1)

            # Extract results for each sentence in batch
            probs = predictions.cpu().numpy()
            for prob in probs:
                results.append({
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2]),
                    'sentiment': self.labels[prob.argmax()],
                    'confidence': float(prob.max())
                })

        return results

    def aggregate_sentiment(self, sentence_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate sentence-level sentiments into document-level metrics.

        Args:
            sentence_results: List of sentiment dictionaries from ALL sentences

        Returns:
            Aggregated sentiment metrics
        """
        if not sentence_results:
            return {
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'avg_neutral': 0.0,
                'dominant_sentiment': 'neutral',
                'total_sentences': 0,
                'positive_sentence_count': 0,
                'negative_sentence_count': 0,
                'neutral_sentence_count': 0
            }

        # Calculate averages across ALL sentences
        avg_positive = sum(r['positive'] for r in sentence_results) / len(sentence_results)
        avg_negative = sum(r['negative'] for r in sentence_results) / len(sentence_results)
        avg_neutral = sum(r['neutral'] for r in sentence_results) / len(sentence_results)

        # Determine dominant sentiment
        sentiment_counts = Counter(r['sentiment'] for r in sentence_results)
        dominant = sentiment_counts.most_common(1)[0][0]

        return {
            'avg_positive': avg_positive,
            'avg_negative': avg_negative,
            'avg_neutral': avg_neutral,
            'dominant_sentiment': dominant,
            'total_sentences': len(sentence_results),
            'positive_sentence_count': sentiment_counts.get('positive', 0),
            'negative_sentence_count': sentiment_counts.get('negative', 0),
            'neutral_sentence_count': sentiment_counts.get('neutral', 0)
        }


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
    from 10-K filings using FinBERT transformer-based sentiment analysis.
    """

    def __init__(self, use_finbert: bool = True):
        """
        Initialize the analyzer with FinBERT.

        Args:
            use_finbert: Whether to use FinBERT for sentiment analysis
        """
        self.stop_words = set(stopwords.words('english'))
        self.logger = self._setup_logger()
        self.use_finbert = use_finbert

        # Initialize FinBERT if requested
        if use_finbert:
            self.logger.info("Initializing FinBERT model...")
            self.finbert = FinBERTAnalyzer()
            device_name = 'GPU' if torch.cuda.is_available() else 'CPU'
            self.logger.info(f"FinBERT loaded successfully on {device_name}")

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

        Expected format: CIK_CompanyName_FilingDate_FormType.txt
        Example: 0000320193_APPLE INC_2023-10-28_10-K.txt

        Args:
            filename: Name of the file to parse

        Returns:
            Dictionary with keys: cik, company_name, filing_date, form_type, year
        """
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')

        metadata = {
            'cik': 'Unknown',
            'company_name': 'Unknown',
            'filing_date': 'Unknown',
            'form_type': 'Unknown',
            'year': 'Unknown'
        }

        if len(parts) >= 4:
            metadata['cik'] = parts[0]
            metadata['company_name'] = parts[1]
            metadata['filing_date'] = parts[2]
            metadata['form_type'] = parts[3]

            try:
                year = parts[2].split('-')[0]
                metadata['year'] = year
            except (IndexError, ValueError):
                pass

        return metadata

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for analysis.

        Args:
            text: Raw text string to preprocess

        Returns:
            Cleaned and normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:\-\']', ' ', text)
        return text.strip()

    def analyze_sentiment_finbert(self, text: str) -> Dict:
        """
        Perform sentiment analysis using FinBERT on the COMPLETE document.
        NO SAMPLING - analyzes ALL sentences.

        Args:
            text: Complete input text to analyze

        Returns:
            Dictionary containing FinBERT sentiment metrics for the entire document
        """
        # Split text into ALL sentences
        sentences = sent_tokenize(text)

        if not sentences:
            return {
                'overall_sentiment': {
                    'avg_positive': 0.0,
                    'avg_negative': 0.0,
                    'avg_neutral': 0.0,
                    'dominant_sentiment': 'neutral',
                    'total_sentences': 0,
                    'positive_sentence_count': 0,
                    'negative_sentence_count': 0,
                    'neutral_sentence_count': 0
                },
                'timeframe_analysis': {
                    'forward_looking': {
                        'avg_positive': 0.0,
                        'avg_negative': 0.0,
                        'avg_neutral': 0.0,
                        'dominant_sentiment': 'neutral',
                        'total_sentences': 0
                    },
                    'historical': {
                        'avg_positive': 0.0,
                        'avg_negative': 0.0,
                        'avg_neutral': 0.0,
                        'dominant_sentiment': 'neutral',
                        'total_sentences': 0
                    }
                }
            }

        self.logger.info(f"Analyzing {len(sentences)} sentences with FinBERT (processing entire document)...")

        # Analyze ALL sentences using batch processing for efficiency
        sentence_results = self.finbert.analyze_sentences_batch(sentences, batch_size=32)

        # Aggregate overall sentiment from ALL sentences
        overall_sentiment = self.finbert.aggregate_sentiment(sentence_results)

        # Perform temporal analysis on ALL sentences
        forward_sentences = []
        forward_indices = []
        historical_sentences = []
        historical_indices = []

        for idx, sentence in enumerate(sentences):
            temporal_class = self.classify_temporal_context(sentence)
            if temporal_class == 'forward':
                forward_sentences.append(sentence)
                forward_indices.append(idx)
            elif temporal_class == 'historical':
                historical_sentences.append(sentence)
                historical_indices.append(idx)

        # Analyze ALL forward-looking sentences
        if forward_sentences:
            self.logger.info(f"Analyzing {len(forward_sentences)} forward-looking sentences...")
            forward_results = self.finbert.analyze_sentences_batch(forward_sentences, batch_size=32)
            forward_sentiment = self.finbert.aggregate_sentiment(forward_results)
        else:
            forward_sentiment = {
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'avg_neutral': 0.0,
                'dominant_sentiment': 'neutral',
                'total_sentences': 0
            }

        # Analyze ALL historical sentences
        if historical_sentences:
            self.logger.info(f"Analyzing {len(historical_sentences)} historical sentences...")
            historical_results = self.finbert.analyze_sentences_batch(historical_sentences, batch_size=32)
            historical_sentiment = self.finbert.aggregate_sentiment(historical_results)
        else:
            historical_sentiment = {
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'avg_neutral': 0.0,
                'dominant_sentiment': 'neutral',
                'total_sentences': 0
            }

        return {
            'overall_sentiment': overall_sentiment,
            'timeframe_analysis': {
                'forward_looking': forward_sentiment,
                'historical': historical_sentiment
            }
        }

    def classify_temporal_context(self, sentence: str) -> str:
        """
        Classify if a sentence refers to forward-looking or historical information.

        Args:
            sentence: Text sentence to classify

        Returns:
            'forward' if forward-looking, 'historical' if historical, 'neutral' otherwise
        """
        words = set(word_tokenize(sentence.lower()))
        words = words - self.stop_words

        forward_matches = len(words.intersection(FORWARD_LOOKING))
        historical_matches = len(words.intersection(HISTORICAL_INDICATORS))

        if forward_matches > historical_matches:
            return 'forward'
        elif historical_matches > forward_matches:
            return 'historical'
        else:
            return 'neutral'

    def analyze_file(self, file_path: str) -> Dict:
        """
        Perform complete FinBERT-based sentiment analysis on a single MD&A file.
        Processes the ENTIRE document - NO SAMPLING.

        Args:
            file_path: Path to the text file to analyze

        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            # Read the complete file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            if not text.strip():
                self.logger.warning(f"Empty file: {file_path}")
                return None

            # Preprocess text
            text = self.preprocess_text(text)

            # Extract metadata from filename
            filename = os.path.basename(file_path)
            metadata = self.parse_filename(filename)
            metadata['filename'] = filename

            self.logger.info(f"Processing {filename}...")

            # Perform FinBERT analysis on COMPLETE document
            finbert_analysis = self.analyze_sentiment_finbert(text)

            # Calculate word count
            words = word_tokenize(text.lower())
            total_words = len([w for w in words if w.isalnum()])

            # Compile complete results
            result = {
                'metadata': metadata,
                'finbert_analysis': finbert_analysis,
                'total_words': total_words,
                'analysis_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Completed analysis of {filename}")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            return None

    def process_batch(self, file_paths: List[str], num_processes: int = None) -> List[Dict]:
        """
        Process multiple files in parallel using multiprocessing.
        Each file is analyzed completely - NO SAMPLING.

        Args:
            file_paths: List of file paths to analyze
            num_processes: Number of parallel processes (default: CPU count - 1)
                          Use 1 for sequential processing (safer for GPU)

        Returns:
            List of analysis results for all files
        """
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)

        self.logger.info(f"Processing {len(file_paths)} files using {num_processes} processes")
        self.logger.info("NOTE: Processing COMPLETE documents - NO SAMPLING")

        # FOR GPU OR IF MULTIPROCESSING HANGS: Use sequential processing
        if num_processes == 1:
            self.logger.info("Using sequential processing (safe for GPU)")
            results = []
            for i, file_path in enumerate(file_paths, 1):
                self.logger.info(f"Processing file {i}/{len(file_paths)}")
                result = self.analyze_file(file_path)
                if result is not None:
                    results.append(result)
            return results

        # Use multiprocessing Pool
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(self.analyze_file, file_paths)

    def save_results(self, results: List[Dict], output_dir: str):
        """
        Save analysis results to both JSON and CSV formats.

        Args:
            results: List of analysis result dictionaries
            output_dir: Directory to save output files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save complete results as JSON
        json_path = os.path.join(output_dir, f'finbert_analysis_detailed_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Detailed JSON saved to: {json_path}")

        # Save summary as CSV
        csv_path = os.path.join(output_dir, f'finbert_analysis_summary_{timestamp}.csv')

        csv_fields = [
            'cik', 'company_name', 'filing_date', 'form_type', 'year', 'filename',
            'total_words',
            # Overall FinBERT metrics
            'finbert_avg_positive', 'finbert_avg_negative', 'finbert_avg_neutral',
            'finbert_dominant_sentiment', 'finbert_total_sentences',
            'finbert_positive_count', 'finbert_negative_count', 'finbert_neutral_count',
            # FinBERT forward-looking metrics
            'forward_avg_positive', 'forward_avg_negative', 'forward_avg_neutral',
            'forward_dominant_sentiment', 'forward_sentence_count',
            # FinBERT historical metrics
            'historical_avg_positive', 'historical_avg_negative', 'historical_avg_neutral',
            'historical_dominant_sentiment', 'historical_sentence_count',
            'analysis_timestamp'
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

            for result in results:
                if 'finbert_analysis' in result:
                    fb = result['finbert_analysis']
                    row = {
                        'cik': result['metadata']['cik'],
                        'company_name': result['metadata']['company_name'],
                        'filing_date': result['metadata']['filing_date'],
                        'form_type': result['metadata']['form_type'],
                        'year': result['metadata']['year'],
                        'filename': result['metadata']['filename'],
                        'total_words': result['total_words'],
                        # Overall FinBERT metrics
                        'finbert_avg_positive': fb['overall_sentiment']['avg_positive'],
                        'finbert_avg_negative': fb['overall_sentiment']['avg_negative'],
                        'finbert_avg_neutral': fb['overall_sentiment']['avg_neutral'],
                        'finbert_dominant_sentiment': fb['overall_sentiment']['dominant_sentiment'],
                        'finbert_total_sentences': fb['overall_sentiment']['total_sentences'],
                        'finbert_positive_count': fb['overall_sentiment']['positive_sentence_count'],
                        'finbert_negative_count': fb['overall_sentiment']['negative_sentence_count'],
                        'finbert_neutral_count': fb['overall_sentiment']['neutral_sentence_count'],
                        # Forward-looking
                        'forward_avg_positive': fb['timeframe_analysis']['forward_looking']['avg_positive'],
                        'forward_avg_negative': fb['timeframe_analysis']['forward_looking']['avg_negative'],
                        'forward_avg_neutral': fb['timeframe_analysis']['forward_looking']['avg_neutral'],
                        'forward_dominant_sentiment': fb['timeframe_analysis']['forward_looking']['dominant_sentiment'],
                        'forward_sentence_count': fb['timeframe_analysis']['forward_looking']['total_sentences'],
                        # Historical
                        'historical_avg_positive': fb['timeframe_analysis']['historical']['avg_positive'],
                        'historical_avg_negative': fb['timeframe_analysis']['historical']['avg_negative'],
                        'historical_avg_neutral': fb['timeframe_analysis']['historical']['avg_neutral'],
                        'historical_dominant_sentiment': fb['timeframe_analysis']['historical']['dominant_sentiment'],
                        'historical_sentence_count': fb['timeframe_analysis']['historical']['total_sentences'],
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
        description='Perform sentiment analysis on 10-K MD&A using FinBERT.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single directory:
  python mdna_sentiment_analyzer.py -i /path/to/mdna/files -o /path/to/results --finbert

  # Multiple directories (multiple years):
  python mdna_sentiment_analyzer.py -i 10ks_cleaned_2018 10ks_cleaned_2019 10ks_cleaned_2020 -o results --finbert

  # With custom number of processes:
  python mdna_sentiment_analyzer.py -i folder1 folder2 folder3 -o /path/to/results --finbert -p 4

NOTE: This program analyzes COMPLETE documents with NO SAMPLING.
      All sentences in each document are processed.
        '''
    )

    parser.add_argument('-i', '--input', type=str, nargs='+', required=True,
                        help='Input directory(s) containing MD&A text files. Can specify multiple directories separated by spaces.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--finbert', action='store_true',
                        help='Use FinBERT for sentiment analysis (analyzes complete documents)')
    parser.add_argument('-p', '--processes', type=int, default=1,
                        help='Number of parallel processes (use 1 for GPU, more for CPU)')

    args = parser.parse_args()

    # Validate that FinBERT is specified
    if not args.finbert:
        print("Error: Must specify --finbert flag to use FinBERT analysis")
        print("Example: python mdna_sentiment_analyzer.py -i input_dir -o output_dir --finbert")
        return

    INPUT_DIRS = args.input  # Now a list of directories
    OUTPUT_DIR = args.output
    NUM_PROCESSES = args.processes

    # Validate all input directories exist
    invalid_dirs = []
    for input_dir in INPUT_DIRS:
        if not os.path.exists(input_dir):
            invalid_dirs.append(f"'{input_dir}' does not exist")
        elif not os.path.isdir(input_dir):
            invalid_dirs.append(f"'{input_dir}' is not a directory")

    if invalid_dirs:
        print("Error: The following input directories are invalid:")
        for error in invalid_dirs:
            print(f"  - {error}")
        return

    # Initialize analyzer with FinBERT
    try:
        analyzer = MDNASentimentAnalyzer(use_finbert=True)
    except Exception as e:
        print(f"Error initializing analyzer: {str(e)}")
        return

    # Log configuration
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("MDNA SENTIMENT ANALYZER - FinBERT MODE")
    analyzer.logger.info("=" * 60)
    analyzer.logger.info(f"Input directory: {INPUT_DIRS}")
    analyzer.logger.info(f"Output directory: {OUTPUT_DIR}")
    analyzer.logger.info("Processing mode: COMPLETE DOCUMENTS (NO SAMPLING)")
    if NUM_PROCESSES:
        analyzer.logger.info(f"Using {NUM_PROCESSES} processes")
    else:
        analyzer.logger.info(f"Using automatic process count (CPU count - 1)")
    analyzer.logger.info("=" * 60)

    # Find all .txt files in the input directory
    file_paths = []
    for input_dir in INPUT_DIRS:
        analyzer.logger.info(f"\nSearching for files in: {input_dir}")
        dir_files = analyzer.find_all_files(input_dir)  # ✅ Passing one directory at a time
        file_paths.extend(dir_files)
        analyzer.logger.info(f"Found {len(dir_files)} files in {input_dir}")

    analyzer.logger.info(f"\nTotal files found across all directories: {len(file_paths)}")

    if not file_paths:
        analyzer.logger.warning("No .txt files found. Please check your input directory.")
        return

    # Process all files using multiprocessing
    results = analyzer.process_batch(file_paths, num_processes=NUM_PROCESSES)

    # Save results to output directory
    if results:
        analyzer.save_results(results, OUTPUT_DIR)
        analyzer.logger.info("=" * 60)
        analyzer.logger.info("Analysis complete!")
        analyzer.logger.info(f"Results saved to: {OUTPUT_DIR}")
        analyzer.logger.info("=" * 60)
    else:
        analyzer.logger.warning("No results to save.")


# Entry point - runs when script is executed directly
if __name__ == '__main__':
    main()