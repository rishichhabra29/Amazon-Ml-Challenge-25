"""
Extract SigLIP embeddings for text and images.
SigLIP: Improved CLIP with sigmoid loss.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from utils import set_seed, save_json, load_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_siglip_model(model_name: str = "google/siglip-base-patch16-224", device: str = 'cuda'):
    """
    Load SigLIP model and processor.
    
    Args:
        model_name: HuggingFace model name
        device: Device to use
    
    Returns:
        model, processor
    """
    logger.info(f"Loading SigLIP model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    logger.info(f"SigLIP model loaded on {device}")
    return model, processor


def extract_siglip_features(
    sample_id: str,
    text: str,
    image_path: Path,
    model,
    processor,
    device: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract SigLIP text and image embeddings, plus similarity.
    
    Args:
        sample_id: Sample ID
        text: Text content
        image_path: Path to image
        model: SigLIP model
        processor: SigLIP processor
        device: Device
    
    Returns:
        text_embedding, image_embedding, similarity
    """
    try:
        # Process text (limit to 512 tokens)
        text_inputs = processor(
            text=text[:512],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Get text features
        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            text_emb = text_outputs.cpu().numpy().astype(np.float32).flatten()
        
        # Process image if available
        if image_path.exists():
            try:
                image = Image.open(image_path).convert('RGB')
                image_inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    image_outputs = model.get_image_features(**image_inputs)
                    image_emb = image_outputs.cpu().numpy().astype(np.float32).flatten()
                
                # Compute similarity (cosine similarity)
                text_norm = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
                image_norm = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
                similarity = (text_norm * image_norm).sum().item()
                
            except Exception as e:
                logger.warning(f"Failed to process image for {sample_id}: {e}")
                # Use zero embedding for missing image
                image_emb = np.zeros_like(text_emb)
                similarity = 0.0
        else:
            # Image doesn't exist
            image_emb = np.zeros_like(text_emb)
            similarity = 0.0
        
        return text_emb, image_emb, similarity
    
    except Exception as e:
        logger.error(f"Failed to extract features for {sample_id}: {e}")
        # Return zero embeddings
        embedding_dim = 768  # SigLIP base dimension
        return np.zeros(embedding_dim, dtype=np.float32), np.zeros(embedding_dim, dtype=np.float32), 0.0


def extract_and_save_siglip_embeddings(
    df: pd.DataFrame,
    split: str,
    image_dir: Path,
    model,
    processor,
    device: str,
    output_dir: Path
):
    """
    Extract SigLIP embeddings for all samples and save.
    
    Args:
        df: DataFrame with sample_id, catalog_content_clean
        split: 'train' or 'test'
        image_dir: Directory containing images
        model: SigLIP model
        processor: SigLIP processor
        device: Device
        output_dir: Output directory
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Extracting SigLIP features for {split} set ({len(df)} samples)")
    logger.info(f"{'='*70}")
    
    # Initialize arrays
    n_samples = len(df)
    embedding_dim = 768  # SigLIP base dimension
    
    text_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
    image_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
    similarities = np.zeros(n_samples, dtype=np.float32)
    
    # Create ID to index mapping
    id_to_index = {str(row['sample_id']): idx for idx, row in df.iterrows()}
    
    # Extract features
    failed_extractions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
        sample_id = str(row['sample_id'])
        text = row['catalog_content_clean']
        image_path = image_dir / f"{sample_id}.jpg"
        
        text_emb, image_emb, similarity = extract_siglip_features(
            sample_id, text, image_path, model, processor, device
        )
        
        text_embeddings[idx] = text_emb
        image_embeddings[idx] = image_emb
        similarities[idx] = similarity
        
        # Track failed extractions (zero similarity)
        if similarity == 0.0:
            failed_extractions.append(sample_id)
    
    # Save embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    
    text_path = output_dir / f'siglip_text_embeddings_{split}.npy'
    image_path = output_dir / f'siglip_image_embeddings_{split}.npy'
    similarity_path = output_dir / f'siglip_similarity_{split}.npy'
    index_path = output_dir / f'siglip_id_to_index_{split}.json'
    
    np.save(text_path, text_embeddings)
    np.save(image_path, image_embeddings)
    np.save(similarity_path, similarities)
    save_json(id_to_index, index_path)
    
    logger.info(f"\n✅ Saved {split} embeddings:")
    logger.info(f"   Text:       {text_path} (shape: {text_embeddings.shape})")
    logger.info(f"   Image:      {image_path} (shape: {image_embeddings.shape})")
    logger.info(f"   Similarity: {similarity_path} (shape: {similarities.shape})")
    logger.info(f"   ID mapping: {index_path}")
    
    # Statistics
    non_zero_similarities = (similarities > 0).sum()
    logger.info(f"\n📊 Extraction Statistics:")
    logger.info(f"   Samples with valid image: {non_zero_similarities}/{n_samples} ({non_zero_similarities/n_samples*100:.2f}%)")
    logger.info(f"   Failed extractions: {len(failed_extractions)}")
    
    if failed_extractions:
        logger.info(f"   Failed sample IDs: {failed_extractions[:10]}{'...' if len(failed_extractions) > 10 else ''}")
    
    # Similarity statistics
    valid_similarities = similarities[similarities > 0]
    if len(valid_similarities) > 0:
        logger.info(f"\n   Similarity stats (valid samples):")
        logger.info(f"      Mean: {valid_similarities.mean():.4f}")
        logger.info(f"      Std:  {valid_similarities.std():.4f}")
        logger.info(f"      Min:  {valid_similarities.min():.4f}")
        logger.info(f"      Max:  {valid_similarities.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract SigLIP embeddings for multimodal data.")
    parser.add_argument('--train_data', type=str, default='data/processed_train.parquet',
                       help='Path to training data parquet file')
    parser.add_argument('--test_data', type=str, default='data/processed_test.parquet',
                       help='Path to test data parquet file')
    parser.add_argument('--image_dir', type=str, default='data/images',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for embeddings')
    parser.add_argument('--model_name', type=str, default='google/siglip-base-patch16-224',
                       help='SigLIP model name from HuggingFace')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load SigLIP model
    model, processor = load_siglip_model(args.model_name, device)
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    # Extract train embeddings
    extract_and_save_siglip_embeddings(
        train_df, 'train', image_dir, model, processor, device, output_dir
    )
    
    # Extract test embeddings
    extract_and_save_siglip_embeddings(
        test_df, 'test', image_dir, model, processor, device, output_dir
    )
    
    logger.info("\n" + "="*70)
    logger.info("✅ SigLIP feature extraction complete!")
    logger.info("="*70)
    logger.info("\n📁 Output files:")
    logger.info(f"   {output_dir}/siglip_text_embeddings_train.npy")
    logger.info(f"   {output_dir}/siglip_image_embeddings_train.npy")
    logger.info(f"   {output_dir}/siglip_similarity_train.npy")
    logger.info(f"   {output_dir}/siglip_id_to_index_train.json")
    logger.info(f"   {output_dir}/siglip_text_embeddings_test.npy")
    logger.info(f"   {output_dir}/siglip_image_embeddings_test.npy")
    logger.info(f"   {output_dir}/siglip_similarity_test.npy")
    logger.info(f"   {output_dir}/siglip_id_to_index_test.json")


if __name__ == '__main__':
    main()

