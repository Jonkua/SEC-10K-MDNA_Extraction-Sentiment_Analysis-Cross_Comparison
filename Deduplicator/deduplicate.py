import re
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import hashlib


class TextDeduplicator:
    """Intelligent text deduplication tool for MDNA sections - optimized for speed."""

    def __init__(self, min_section_length: int = 200, similarity_threshold: float = 0.85):
        """
        Initialize the deduplicator.

        Args:
            min_section_length: Minimum character length to consider as a section
            similarity_threshold: Similarity threshold (0-1) for duplicate detection
        """
        self.min_section_length = min_section_length
        self.similarity_threshold = similarity_threshold

    def get_text_hash(self, text: str) -> str:
        """Get a quick hash of text for fast exact duplicate detection."""
        return hashlib.md5(text.lower().encode()).hexdigest()

    def normalize_text(self, text: str) -> str:
        """Normalize text for better comparison."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\r\n]+', '\n', text)
        return text.strip()

    def split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into character-based chunks for faster processing."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def calculate_similarity_fast(self, text1: str, text2: str) -> float:
        """Fast similarity calculation using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union else 0.0

    def find_duplicates_fast(self, text: str) -> List[Dict]:
        """
        Fast duplicate detection using hash-based exact matching
        and selective similarity checking.
        """
        # Split into chunks
        chunks = self.split_into_chunks(text, chunk_size=self.min_section_length)

        if len(chunks) < 2:
            return []

        duplicates = []
        seen_hashes = {}
        marked = set()

        # First pass: exact duplicates using hashing (very fast)
        for i, chunk in enumerate(chunks):
            if i in marked or len(chunk) < self.min_section_length:
                continue

            chunk_hash = self.get_text_hash(chunk)

            if chunk_hash in seen_hashes:
                original_idx = seen_hashes[chunk_hash]
                duplicates.append({
                    'original': chunks[original_idx],
                    'duplicate': chunk,
                    'similarity': 1.0,
                    'original_start': original_idx,
                    'duplicate_start': i,
                })
                marked.add(i)
            else:
                seen_hashes[chunk_hash] = i

        # Second pass: near-duplicates (only for unmarked chunks)
        unmarked_indices = [i for i in range(len(chunks)) if
                            i not in marked and len(chunks[i]) >= self.min_section_length]

        for i in range(len(unmarked_indices)):
            idx1 = unmarked_indices[i]
            chunk1 = chunks[idx1]

            for j in range(i + 1, len(unmarked_indices)):
                idx2 = unmarked_indices[j]

                if idx2 in marked:
                    continue

                chunk2 = chunks[idx2]

                # Quick length check to avoid unnecessary comparisons
                len_ratio = min(len(chunk1), len(chunk2)) / max(len(chunk1), len(chunk2))
                if len_ratio < 0.5:  # Skip if lengths are too different
                    continue

                similarity = self.calculate_similarity_fast(chunk1, chunk2)

                if similarity >= self.similarity_threshold:
                    duplicates.append({
                        'original': chunk1,
                        'duplicate': chunk2,
                        'similarity': similarity,
                        'original_start': idx1,
                        'duplicate_start': idx2,
                    })
                    marked.add(idx2)
                    break

        return duplicates

    def remove_duplicates(self, text: str) -> Tuple[str, int, List[Dict]]:
        """
        Remove duplicate sections from text.

        Returns:
            Tuple of (cleaned_text, removed_count, duplicate_info)
        """
        duplicates = self.find_duplicates_fast(text)

        if not duplicates:
            return text, 0, []

        # Sort duplicates by position (remove from end to start)
        duplicates.sort(key=lambda x: x['duplicate_start'], reverse=True)

        cleaned_text = text
        removed_sections = []
        actually_removed = 0

        for dup in duplicates:
            duplicate_text = dup['duplicate']

            # Find exact match in text
            idx = cleaned_text.find(duplicate_text)

            if idx != -1:
                # Find if there's another occurrence
                second_idx = cleaned_text.find(duplicate_text, idx + 1)

                if second_idx != -1:
                    # Remove the second occurrence
                    cleaned_text = cleaned_text[:second_idx] + cleaned_text[second_idx + len(duplicate_text):]

                    # Clean up excess whitespace
                    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
                    cleaned_text = re.sub(r'  +', ' ', cleaned_text)

                    removed_sections.append({
                        'text': duplicate_text[:150] + '...' if len(duplicate_text) > 150 else duplicate_text,
                        'similarity': round(dup['similarity'] * 100, 1),
                        'length': len(duplicate_text)
                    })
                    actually_removed += 1

        return cleaned_text.strip(), actually_removed, removed_sections

    def process_file(self, input_path: str, output_path: str = None) -> Dict:
        """
        Process a single file and remove duplicates.

        Args:
            input_path: Path to input file
            output_path: Path to output file (if None, will append '_(cleaned)' to filename)

        Returns:
            Dictionary with processing results
        """
        # Read input file
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_text = f.read()

        # Process text
        cleaned_text, removed_count, duplicates = self.remove_duplicates(original_text)

        # Determine output path
        if output_path is None:
            path_obj = Path(input_path)
            output_path = path_obj.parent / f"{path_obj.stem}_(cleaned){path_obj.suffix}"

        # Write cleaned file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        # Calculate statistics
        original_size = len(original_text)
        cleaned_size = len(cleaned_text)
        reduction_pct = ((original_size - cleaned_size) / original_size * 100) if original_size > 0 else 0

        return {
            'input_file': input_path,
            'output_file': str(output_path),
            'original_size': original_size,
            'cleaned_size': cleaned_size,
            'reduction_pct': reduction_pct,
            'removed_count': removed_count,
            'duplicates': duplicates
        }


def process_single_file_wrapper(args):
    """Wrapper function for multiprocessing."""
    file_path, output_path, min_length, similarity = args
    deduplicator = TextDeduplicator(min_section_length=min_length, similarity_threshold=similarity)
    return deduplicator.process_file(str(file_path), str(output_path))


def process_directory_parallel(input_dir: str, output_dir: str = None,
                               pattern: str = "*.txt",
                               min_length: int = 200,
                               similarity: float = 0.85,
                               workers: int = None) -> List[Dict]:
    """
    Process all files in a directory using parallel processing.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        pattern: File pattern to match
        min_length: Minimum section length
        similarity: Similarity threshold
        workers: Number of parallel workers (default: CPU count)

    Returns:
        List of result dictionaries
    """
    input_path = Path(input_dir)

    if output_dir is None:
        output_path = input_path / "cleaned"
    else:
        output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    files = list(input_path.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return []

    print(f"Found {len(files)} file(s) to process...")

    # Prepare arguments for parallel processing
    args_list = [
        (file_path, output_path / file_path.name, min_length, similarity)
        for file_path in files
    ]

    # Determine number of workers
    if workers is None:
        workers = min(cpu_count(), len(files))

    print(f"Using {workers} parallel workers\n")

    # Process files in parallel
    results = []
    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap(process_single_file_wrapper, args_list), 1):
            filename = Path(result['input_file']).name
            print(
                f"[{i}/{len(files)}] {filename}: Removed {result['removed_count']} duplicates, saved {result['reduction_pct']:.1f}%")
            results.append(result)

    return results


def print_summary(results: List[Dict]):
    """Print a summary of processing results."""
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80 + "\n")

    total_files = len(results)
    total_duplicates = sum(r['removed_count'] for r in results)
    total_original = sum(r['original_size'] for r in results)
    total_cleaned = sum(r['cleaned_size'] for r in results)
    total_saved = total_original - total_cleaned

    print(f"Files processed: {total_files}")
    print(f"Total duplicates removed: {total_duplicates}")
    print(f"Total original size: {total_original:,} characters")
    print(f"Total cleaned size: {total_cleaned:,} characters")
    print(
        f"Total space saved: {total_saved:,} characters ({(total_saved / total_original * 100):.1f}%)" if total_original > 0 else "Total space saved: 0")

    # Show top 5 files with most duplicates
    if results:
        print("\n" + "-" * 80)
        print("TOP FILES BY DUPLICATES REMOVED")
        print("-" * 80 + "\n")

        sorted_results = sorted(results, key=lambda x: x['removed_count'], reverse=True)[:5]
        for i, result in enumerate(sorted_results, 1):
            if result['removed_count'] > 0:
                filename = Path(result['input_file']).name
                print(f"{i}. {filename}: {result['removed_count']} duplicates ({result['reduction_pct']:.1f}% saved)")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Fast text deduplication tool for MDNA sections (parallel processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory with parallel processing
  python deduplicate.py 10ks --output 10ks_cleaned

  # Use specific number of workers
  python deduplicate.py 10ks --output 10ks_cleaned --workers 8

  # Adjust similarity threshold
  python deduplicate.py 10ks --output 10ks_cleaned --similarity 0.9

  # Process single file
  python deduplicate.py file.txt --output cleaned_file.txt
        """
    )

    parser.add_argument(
        'input',
        help='Input file or directory path'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file or directory path'
    )

    parser.add_argument(
        '--similarity', '-s',
        type=float,
        default=0.85,
        help='Similarity threshold (0-1, default: 0.85)'
    )

    parser.add_argument(
        '--min-length', '-m',
        type=int,
        default=200,
        help='Minimum section length in characters (default: 200)'
    )

    parser.add_argument(
        '--pattern', '-p',
        default='*.txt',
        help='File pattern for directory processing (default: *.txt)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: {cpu_count()})'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FAST TEXT DEDUPLICATION TOOL")
    print("=" * 80)
    print(f"Settings: Similarity={args.similarity}, Min Length={args.min_length} chars")
    print(f"CPU Cores Available: {cpu_count()}\n")

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        print(f"Processing single file: {input_path.name}\n")
        deduplicator = TextDeduplicator(
            min_section_length=args.min_length,
            similarity_threshold=args.similarity
        )
        result = deduplicator.process_file(str(input_path), args.output)
        results = [result]
        print(f"✓ Removed {result['removed_count']} duplicates, saved {result['reduction_pct']:.1f}%")
    elif input_path.is_dir():
        # Process directory with parallel processing
        print(f"Processing directory: {input_path}\n")
        results = process_directory_parallel(
            str(input_path),
            args.output,
            args.pattern,
            args.min_length,
            args.similarity,
            args.workers
        )
    else:
        print(f"Error: '{args.input}' is not a valid file or directory")
        return

    # Print summary
    if results:
        print_summary(results)
        print("\n✅ Processing complete! Cleaned files saved.")
    else:
        print("\n⚠️  No files were processed.")


if __name__ == "__main__":
    main()