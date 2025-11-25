"""
Extract CLIP (Contrastive Language-Image Pre-training) features for multimodal learning.

CLIP provides unified vision-language representations where text and images are
projected into the SAME embedding space through contrastive learning.

Model: openai/clip-vit-base-patch32
- Vision encoder: ViT-B/32
- Text encoder: Transformer
- Output: 512-dim embeddings for BOTH text and image (aligned!)
- Trained on 400M image-text pairs

Advantages:
- Same embedding space (naturally aligned)
- Better for e-commerce (trained on web data)
- Fast and efficient
- State-of-the-art cross-modal understanding
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from utils import set_seed, save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = 'cuda'):
    """Load CLIP model and processor."""
    logger.info(f"Loading CLIP model ({model_name})...")
    
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    logger.info(f"CLIP model loaded on {device}")
    logger.info(f"Embedding dimension: {model.config.projection_dim}")
    return model, processor


def extract_clip_features(
    df: pd.DataFrame,
    image_dir: Path,
    model,
    processor,
    device: str,
    batch_size: int = 32
) -> Tuple:
    """
    Extract CLIP features for text and images.
    
    Returns:
        text_embeddings: [N, 512] array
        image_embeddings: [N, 512] array
        id_to_index: Dict mapping sample_id to index
        similarity_scores: [N] array of text-image cosine similarity
    """
    
    text_embeddings = []
    image_embeddings = []
    similarity_scores = []
    id_to_index = {}
    
    embed_dim = model.config.projection_dim
    logger.info(f"Processing {len(df)} samples with CLIP...")
    logger.info(f"Embedding dimension: {embed_dim}")
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLIP features"):
            sample_id = row['sample_id']
            text = row.get('catalog_content_clean', row.get('catalog_content', ''))
            
            # Load image
            image_path = image_dir / f"{sample_id}.jpg"
            if not image_path.exists():
                # Try alternative extensions
                for ext in ['.png', '.jpeg', '.webp']:
                    alt_path = image_dir / f"{sample_id}{ext}"
                    if alt_path.exists():
                        image_path = alt_path
                        break
            
            # Default embeddings for missing data
            text_emb = np.zeros(embed_dim, dtype=np.float32)
            image_emb = np.zeros(embed_dim, dtype=np.float32)
            similarity = 0.0
            
            try:
                has_text = text and len(text.strip()) > 0
                has_image = image_path.exists()
                
                # Process text
                if has_text:
                    text_inputs = processor(
                        text=[text[:512]],  # Truncate long text
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)
                    
                    # Get text features (already projected to common space)
                    text_features = model.get_text_features(**text_inputs)
                    text_emb = text_features.cpu().numpy().astype(np.float32).flatten()
                
                # Process image
                if has_image:
                    image = Image.open(image_path).convert('RGB')
                    
                    image_inputs = processor(
                        images=image,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Get image features (already projected to common space)
                    image_features = model.get_image_features(**image_inputs)
                    image_emb = image_features.cpu().numpy().astype(np.float32).flatten()
                
                # Compute cosine similarity (both in same space!)
                if has_text and has_image:
                    # Normalize features
                    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Cosine similarity
                    similarity = (text_norm * image_norm).sum().item()
                
            except Exception as e:
                logger.warning(f"Error processing sample {sample_id}: {e}")
            
            text_embeddings.append(text_emb)
            image_embeddings.append(image_emb)
            similarity_scores.append(similarity)
            id_to_index[str(sample_id)] = len(id_to_index)
    
    text_embeddings = np.vstack(text_embeddings)
    image_embeddings = np.vstack(image_embeddings)
    similarity_scores = np.array(similarity_scores, dtype=np.float32)
    
    logger.info(f"Extracted features for {len(df)} samples")
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")
    logger.info(f"Image embeddings shape: {image_embeddings.shape}")
    logger.info(f"Similarity scores shape: {similarity_scores.shape}")
    logger.info(f"Similarity range: [{similarity_scores.min():.3f}, {similarity_scores.max():.3f}]")
    logger.info(f"Mean similarity: {similarity_scores.mean():.3f}")
    
    return text_embeddings, image_embeddings, similarity_scores, id_to_index


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP features')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input parquet file')
    parser.add_argument('--image_dir', type=str, default='data/images',
                       help='Directory containing images')
    parser.add_argument('--out_dir', type=str, default='data/',
                       help='Output directory')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Setup paths
    input_path = Path(args.input)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine split name
    split_name = 'train' if 'train' in input_path.name else 'test'
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Load CLIP model
    model, processor = load_clip_model(args.model_name, device)
    
    # Extract features
    text_emb, image_emb, similarity, id_to_idx = extract_clip_features(
        df, image_dir, model, processor, device, args.batch_size
    )
    
    # Save features
    text_emb_path = out_dir / f'clip_text_embeddings_{split_name}.npy'
    image_emb_path = out_dir / f'clip_image_embeddings_{split_name}.npy'
    similarity_path = out_dir / f'clip_similarity_{split_name}.npy'
    index_path = out_dir / f'clip_id_to_index_{split_name}.json'
    
    np.save(text_emb_path, text_emb)
    np.save(image_emb_path, image_emb)
    np.save(similarity_path, similarity)
    save_json(id_to_idx, str(index_path))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CLIP Feature Extraction Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Split: {split_name}")
    logger.info(f"Samples processed: {len(df)}")
    logger.info(f"Text embeddings: {text_emb.shape} -> {text_emb_path}")
    logger.info(f"Image embeddings: {image_emb.shape} -> {image_emb_path}")
    logger.info(f"Similarity scores: {similarity.shape} -> {similarity_path}")
    logger.info(f"Index mapping: {len(id_to_idx)} entries -> {index_path}")
    logger.info(f"Total size: {(text_emb.nbytes + image_emb.nbytes + similarity.nbytes) / 1024 / 1024:.2f} MB")
    logger.info(f"{'='*60}")
    logger.info("✅ CLIP extraction complete!")


if __name__ == '__main__':
    main()

