"""
Text preprocessing and structured field extraction.
Converts raw product catalogs into cleaned, structured Parquet files.
"""

import re
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import unicodedata
import pandas as pd
import numpy as np
from utils import safe_read_csv, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Common units pattern
UNIT_PATTERN = r'\b(ounce|oz|pound|lb|gram|g|kg|kilogram|liter|litre|l|ml|milliliter|' \
               r'fl oz|fluid ounce|gallon|gal|pint|pt|quart|qt|count|ct|piece|pc|' \
               r'pack|box|case|each|item|unit)\b'


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â€"', '-')
    
    # Normalize whitespace but keep sentence structure
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def extract_value_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract value and unit from text.
    Prioritizes explicit 'Value:' and 'Unit:' labels.
    
    Args:
        text: Product catalog text
        
    Returns:
        Tuple of (value, unit)
    """
    if not isinstance(text, str):
        return None, None
    
    # Try explicit Value: and Unit: labels first
    value = None
    unit = None
    
    # Extract Value
    value_match = re.search(r'Value:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if value_match:
        try:
            value = float(value_match.group(1))
        except (ValueError, AttributeError):
            pass
    
    # Extract Unit
    unit_match = re.search(r'Unit:\s*([A-Za-z\s]+?)(?:\n|$)', text, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1).strip().lower()
    
    # Fallback: look for number + unit pattern
    if value is None or unit is None:
        pattern = r'(\d+(?:\.\d+)?)\s*(' + UNIT_PATTERN + r')'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            # Take the first or most prominent match
            try:
                fallback_value = float(matches[0][0])
                fallback_unit = matches[0][1].lower()
                
                if value is None:
                    value = fallback_value
                if unit is None:
                    unit = fallback_unit
            except (ValueError, IndexError):
                pass
    
    # Normalize unit
    if unit:
        unit = unit.lower().strip()
        # Standardize common variations
        unit_map = {
            'ounce': 'oz',
            'fluid ounce': 'fl oz',
            'pound': 'lb',
            'gram': 'g',
            'kilogram': 'kg',
            'liter': 'l',
            'litre': 'l',
            'milliliter': 'ml',
            'gallon': 'gal',
            'pint': 'pt',
            'quart': 'qt',
            'count': 'ct',
            'piece': 'pc',
        }
        unit = unit_map.get(unit, unit)
    
    return value, unit


def extract_pack_size(text: str) -> Optional[int]:
    """
    Extract pack size from text (e.g., 'Pack of 6', '12 per case').
    
    Args:
        text: Product catalog text
        
    Returns:
        Pack size as integer or None
    """
    if not isinstance(text, str):
        return None
    
    # Pattern: "Pack of N", "N per case", "N count", "N-pack", "(Pack of N)"
    patterns = [
        r'pack\s+of\s+(\d+)',
        r'\(pack\s+of\s+(\d+)\)',
        r'(\d+)\s*-?\s*pack',
        r'(\d+)\s+per\s+case',
        r'(\d+)\s*x\s*ct',
        r'(\d+)\s*count',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, AttributeError):
                pass
    
    return None


def extract_item_name(text: str) -> Optional[str]:
    """
    Extract item name from catalog content.
    
    Args:
        text: Product catalog text
        
    Returns:
        Item name or None
    """
    if not isinstance(text, str):
        return None
    
    # Look for "Item Name:" label
    match = re.search(r'Item Name:\s*([^\n]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: take first line if it looks like a product name
    lines = text.split('\n')
    if lines:
        first_line = lines[0].strip()
        if len(first_line) > 10 and len(first_line) < 200:
            return first_line
    
    return None


def extract_brand(item_name: str) -> Optional[str]:
    """
    Extract brand from item name (typically first 1-2 words).
    
    Args:
        item_name: Product item name
        
    Returns:
        Brand name or None
    """
    if not isinstance(item_name, str):
        return None
    
    # Take first word or first two words if short
    words = item_name.split()
    if not words:
        return None
    
    if len(words) == 1:
        return words[0]
    
    # Return first word if it's capitalized and reasonable length
    if words[0][0].isupper() and 2 <= len(words[0]) <= 20:
        return words[0]
    
    # Otherwise return first two words if total length reasonable
    brand = ' '.join(words[:2])
    if len(brand) <= 30:
        return brand
    
    return words[0]


def compute_text_stats(text: str) -> Dict[str, int]:
    """
    Compute basic text statistics.
    
    Args:
        text: Text content
        
    Returns:
        Dictionary with text_len, word_count, num_sentences
    """
    if not isinstance(text, str):
        return {'text_len': 0, 'word_count': 0, 'num_sentences': 0}
    
    text_len = len(text)
    word_count = len(text.split())
    
    # Count sentences (rough estimate)
    sentence_endings = r'[.!?]+'
    num_sentences = len(re.findall(sentence_endings, text))
    if num_sentences == 0 and text_len > 0:
        num_sentences = 1  # At least one sentence if there's text
    
    return {
        'text_len': text_len,
        'word_count': word_count,
        'num_sentences': num_sentences
    }


def preprocess_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess a DataFrame with product data.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (has price column)
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Preprocessing {'training' if is_train else 'test'} data with {len(df)} samples")
    
    # Create a copy
    df = df.copy()
    
    # Clean catalog_content
    df['catalog_content_clean'] = df['catalog_content'].apply(clean_text)
    
    # Extract structured fields
    logger.info("Extracting value and unit...")
    value_unit = df['catalog_content'].apply(extract_value_unit)
    df['value_num'] = value_unit.apply(lambda x: x[0])
    df['unit'] = value_unit.apply(lambda x: x[1])
    
    logger.info("Extracting pack size...")
    df['pack_size'] = df['catalog_content'].apply(extract_pack_size)
    
    logger.info("Extracting item names and brands...")
    df['item_name'] = df['catalog_content'].apply(extract_item_name)
    df['brand'] = df['item_name'].apply(extract_brand)
    
    # Compute text statistics
    logger.info("Computing text statistics...")
    text_stats = df['catalog_content_clean'].apply(compute_text_stats)
    df['text_len'] = text_stats.apply(lambda x: x['text_len'])
    df['word_count'] = text_stats.apply(lambda x: x['word_count'])
    df['num_sentences'] = text_stats.apply(lambda x: x['num_sentences'])
    
    # Image availability flag
    df['has_image'] = df['image_link'].notna() & (df['image_link'].str.strip() != '')
    
    # Price processing (training only)
    if is_train and 'price' in df.columns:
        logger.info("Processing price column...")
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price_log'] = np.log(df['price'] + 1e-6)
    
    # Handle missing values
    df['value_num'] = df['value_num'].fillna(-1)
    df['pack_size'] = df['pack_size'].fillna(1)
    df['unit'] = df['unit'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')
    
    logger.info(f"Preprocessing complete. Output shape: {df.shape}")
    
    return df


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess product catalog data')
    parser.add_argument('--train_path', type=str, default='data/train.csv',
                       help='Path to training CSV')
    parser.add_argument('--test_path', type=str, default='data/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--out_dir', type=str, default='data/',
                       help='Output directory')
    parser.add_argument('--dry_run', action='store_true',
                       help='Process only first 100 rows for debugging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Process training data
    logger.info(f"Loading training data from {args.train_path}")
    train_df = safe_read_csv(args.train_path)
    
    if args.dry_run:
        logger.warning("DRY RUN MODE: Processing only first 100 rows")
        train_df = train_df.head(100)
    
    train_processed = preprocess_dataframe(train_df, is_train=True)
    
    train_out_path = Path(args.out_dir) / 'processed_train.parquet'
    train_processed.to_parquet(train_out_path, index=False)
    logger.info(f"Saved processed training data to {train_out_path}")
    
    # Process test data
    logger.info(f"Loading test data from {args.test_path}")
    test_df = safe_read_csv(args.test_path)
    
    if args.dry_run:
        test_df = test_df.head(100)
    
    test_processed = preprocess_dataframe(test_df, is_train=False)
    
    test_out_path = Path(args.out_dir) / 'processed_test.parquet'
    test_processed.to_parquet(test_out_path, index=False)
    logger.info(f"Saved processed test data to {test_out_path}")
    
    # Summary statistics
    logger.info("\n=== Preprocessing Summary ===")
    logger.info(f"Training samples: {len(train_processed)}")
    logger.info(f"Test samples: {len(test_processed)}")
    
    if 'price' in train_processed.columns:
        logger.info(f"Price range: ${train_processed['price'].min():.2f} - ${train_processed['price'].max():.2f}")
        logger.info(f"Price mean: ${train_processed['price'].mean():.2f}")
        logger.info(f"Price median: ${train_processed['price'].median():.2f}")
    
    logger.info(f"Samples with images: {train_processed['has_image'].sum()} / {len(train_processed)}")
    logger.info(f"Unique units: {train_processed['unit'].nunique()}")
    logger.info(f"Unique brands: {train_processed['brand'].nunique()}")
    
    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()

