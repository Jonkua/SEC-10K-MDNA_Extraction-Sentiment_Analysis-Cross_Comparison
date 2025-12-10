#!/usr/bin/env python3
"""
Split sentiment-scored CSV file by year.

This script reads a CSV file containing sentiment scores (from FinBERT) for financial
documents and splits it into separate CSV files by year.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def split_csv_by_year(input_file, output_dir=None):
    """
    Split a CSV file by year column and save each year to a separate file.

    Args:
        input_file (str): Path to the input CSV file
        output_dir (str, optional): Directory to save output files. If None, uses input file's directory.

    Returns:
        dict: Dictionary mapping years to output file paths
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)

        # Display basic info about the dataframe
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")

        # Try to identify the year column (case-insensitive)
        year_column = None
        for col in df.columns:
            if 'year' in col.lower():
                year_column = col
                break

        if year_column is None:
            print("\nError: Could not find a 'year' column in the CSV file.")
            print(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)

        print(f"Using year column: '{year_column}'")

        # Get unique years
        years = sorted(df[year_column].unique())
        print(f"Found years: {', '.join(map(str, years))}")

        # Determine output directory
        if output_dir is None:
            output_dir = Path(input_file).parent
        else:
            output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split and save by year
        output_files = {}
        for year in years:
            # Filter data for this year
            year_df = df[df[year_column] == year]

            # Create output filename
            output_file = output_dir / f"sentiment_summary_bert_{year}.csv"

            # Save to CSV
            year_df.to_csv(output_file, index=False)
            output_files[year] = output_file

            print(f"Saved {len(year_df)} rows for year {year} to: {output_file}")

        print(f"\nSuccessfully split data into {len(output_files)} files!")
        print(f"Original file preserved at: {input_file}")

        return output_files

    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_file} is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


def main():
    """Main function to handle command-line arguments and execute the split."""
    parser = argparse.ArgumentParser(
        description='Split sentiment-scored CSV file by year',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/sentiment_data.csv
  %(prog)s /path/to/sentiment_data.csv -o /path/to/output_directory
        """
    )

    parser.add_argument(
        'input_file',
        help='Path to the input CSV file containing sentiment scores'
    )

    parser.add_argument(
        '-o', '--output-dir',
        help='Directory to save output files (default: same as input file)',
        default=None
    )

    args = parser.parse_args()

    # Execute the split
    split_csv_by_year(args.input_file, args.output_dir)


if __name__ == '__main__':
    main()