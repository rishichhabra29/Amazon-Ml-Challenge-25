"""
Utility functions for the Smart Product Pricing Challenge.
Includes image download, file safety guards, and seed setting.
"""

import os
import sys
import random
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set seed {seed} for Python, NumPy, and PyTorch")
    except ImportError:
        logger.info(f"Set seed {seed} for Python and NumPy (PyTorch not available)")


def ensure_local_only() -> None:
    """
    Enforce that scripts only access local data.
    Raises an exception if called from a context that might access external data.
    This is a placeholder guard - actual enforcement is in safe_read_csv.
    """
    # This function serves as a reminder and documentation
    logger.debug("Local-only data access policy enforced")


def safe_read_csv(path: str, allowed_dirs: Optional[list] = None) -> pd.DataFrame:
    """
    Read CSV file with safety checks to ensure only local data access.
    
    Args:
        path: Path to CSV file
        allowed_dirs: List of allowed directory prefixes (default: ['data'])
    
    Returns:
        DataFrame
        
    Raises:
        ValueError: If path is outside allowed directories or attempts network access
    """
    if allowed_dirs is None:
        allowed_dirs = ['data', 'dataset']
    
    # Normalize path
    abs_path = Path(path).resolve()
    
    # Check for URL schemes
    if any(path.startswith(scheme) for scheme in ['http://', 'https://', 'ftp://', 's3://']):
        raise ValueError(
            f"EXTERNAL DATA ACCESS BLOCKED: Attempting to read from network: {path}. "
            "Only local files are allowed per challenge rules."
        )
    
    # Check if path is under allowed directories
    allowed = False
    for allowed_dir in allowed_dirs:
        try:
            allowed_path = Path(allowed_dir).resolve()
            if abs_path.is_relative_to(allowed_path) or abs_path == allowed_path:
                allowed = True
                break
        except (ValueError, AttributeError):
            # Python < 3.9 doesn't have is_relative_to, use alternative
            try:
                abs_path.relative_to(allowed_path)
                allowed = True
                break
            except ValueError:
                continue
    
    if not allowed:
        raise ValueError(
            f"EXTERNAL DATA ACCESS BLOCKED: Path {path} is outside allowed directories {allowed_dirs}. "
            "Only local data directory access is permitted per challenge rules."
        )
    
    logger.info(f"Reading CSV from safe path: {path}")
    return pd.read_csv(path)


def download_images(
    df: pd.DataFrame,
    image_col: str = 'image_link',
    id_col: str = 'sample_id',
    out_dir: str = 'data/images',
    max_retries: int = 3,
    timeout: int = 10,
    delay: float = 0.1,
    min_size: int = 50,
    max_size_mb: int = 10
) -> Dict[str, Optional[str]]:
    """
    Download product images with validation and retry logic.
    
    Args:
        df: DataFrame containing image URLs
        image_col: Column name with image URLs
        id_col: Column name with sample IDs
        out_dir: Output directory for images
        max_retries: Maximum number of download attempts per image
        timeout: Request timeout in seconds
        delay: Base delay between retries (exponential backoff)
        min_size: Minimum image dimension in pixels
        max_size_mb: Maximum file size in MB
        
    Returns:
        Dictionary mapping sample_id to local path (or None if failed)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    failures = []
    
    logger.info(f"Starting download of {len(df)} images to {out_dir}")
    
    for idx, row in df.iterrows():
        sample_id = row[id_col]
        image_url = row[image_col]
        
        # Skip if URL is missing or invalid
        if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
            results[sample_id] = None
            failures.append({
                'sample_id': sample_id,
                'image_link': image_url,
                'reason': 'Missing or invalid URL'
            })
            continue
        
        # Determine file extension
        parsed = urlparse(image_url)
        ext = os.path.splitext(parsed.path)[1]
        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            ext = '.jpg'
        
        output_path = os.path.join(out_dir, f"{sample_id}{ext}")
        
        # Skip if already downloaded
        if os.path.exists(output_path):
            try:
                # Validate existing file
                img = Image.open(output_path)
                if img.width >= min_size and img.height >= min_size:
                    results[sample_id] = output_path
                    continue
            except Exception:
                # Invalid file, will re-download
                pass
        
        # Download with retries
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Request with timeout
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(image_url, timeout=timeout, headers=headers, stream=True)
                response.raise_for_status()
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                    raise ValueError(f"Image too large: {content_length} bytes")
                
                # Read content
                image_data = response.content
                
                # Validate with PIL
                img = Image.open(BytesIO(image_data))
                img.verify()  # Verify it's a valid image
                
                # Re-open for size check (verify() closes the file)
                img = Image.open(BytesIO(image_data))
                
                if img.width < min_size or img.height < min_size:
                    raise ValueError(f"Image too small: {img.width}x{img.height}")
                
                # Save to disk
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                results[sample_id] = output_path
                success = True
                break
                
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
        
        if not success:
            results[sample_id] = None
            failures.append({
                'sample_id': sample_id,
                'image_link': image_url,
                'reason': last_error or 'Unknown error'
            })
        
        # Progress logging
        if (idx + 1) % 100 == 0:
            success_count = sum(1 for v in results.values() if v is not None)
            logger.info(f"Progress: {idx + 1}/{len(df)} images, "
                       f"{success_count} successful, {len(failures)} failed")
    
    # Save failure log
    if failures:
        failure_path = os.path.join('data', 'image_failures.csv')
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        logger.warning(f"Saved {len(failures)} failed downloads to {failure_path}")
    
    success_count = sum(1 for v in results.values() if v is not None)
    logger.info(f"Download complete: {success_count}/{len(df)} images successful")
    
    return results


def get_file_extension(url: str, default: str = '.jpg') -> str:
    """
    Extract file extension from URL.
    
    Args:
        url: Image URL
        default: Default extension if not found
        
    Returns:
        File extension (including dot)
    """
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']
    return ext if ext in valid_extensions else default


def save_json(data: dict, path: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output path
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> dict:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Input path
        
    Returns:
        Dictionary
    """
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {path}")
    return data
