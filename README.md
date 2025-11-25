📦 Smart Product Pricing — Amazon ML Challenge 2025

Multimodal Deep Learning Pipeline (CLIP + SigLIP + Blended MLP)

This repository contains the full multimodal solution developed for the Amazon ML Challenge 2025 – Smart Product Pricing Task.
The goal is to predict the optimal price of 75,000+ e-commerce products using:

Product images

Product catalog descriptions

Structured fields (value, unit, brand, pack size)

Robust outlier handling

CLIP + SigLIP embeddings

A blended MLP regressor with EMA stabilization

This approach secured a Top-15 Public Leaderboard Rank (#14, SMAPE: 40.858%).

🚀 Project Structure
├── preprocess.py                    # Text cleanup + unit extraction + parquet generator
├── extract_clip_features.py         # CLIP (ViT-B/32) embeddings
├── extract_siglip_features.py       # SigLIP embeddings
├── train_clip_siglip_blend_v2.py    # Final blended model training (MLP + outlier handling)
├── utils.py                         # Seed, CSV safety, image downloader, helpers
├── data/                            # Raw + processed data + embeddings
└── models/                          # Saved models + predictions

1️⃣ Data Preprocessing (preprocess.py)

preprocess

✔ Cleans & normalizes product text

Unicode normalization

Fixes encoding artifacts

Removes HTML tags

Compresses whitespace

✔ Extracts structured fields

Value + unit extraction (oz, lb, ml, g, etc.)

Pack size extraction (Pack of 6, 12 per case, …)

Brand extraction

Item name extraction

✔ Text statistics

length, word count, sentence count

has_image flag

✔ Price transformation

log(price) stored as price_log

✔ Output

Creates:

data/processed_train.parquet
data/processed_test.parquet

▶ Run:
python preprocess.py --train_path data/train.csv --test_path data/test.csv --out_dir data/

2️⃣ CLIP Feature Extraction (extract_clip_features.py)

extract_clip_features

Extracts 512-dim text + 512-dim image embeddings using:
openai/clip-vit-base-patch32

Features generated:

clip_text_embeddings_{train|test}.npy

clip_image_embeddings_{train|test}.npy

clip_similarity_{train|test}.npy (cosine similarity)

ID → index mapping JSON

▶ Run:
python extract_clip_features.py \
    --input data/processed_train.parquet \
    --image_dir data/images \
    --out_dir data/

3️⃣ SigLIP Feature Extraction (extract_siglip_features.py)

extract_siglip_features

Uses google/siglip-base-patch16-224 for improved cross-modal embeddings.

Outputs:

siglip_text_embeddings_{train|test}.npy

siglip_image_embeddings_*.npy

siglip_similarity_*.npy

ID mapping JSON

Includes:

Safe image loading

Zero-vector fallback

Text truncation

▶ Run:
python extract_siglip_features.py \
    --train_data data/processed_train.parquet \
    --test_data data/processed_test.parquet \
    --image_dir data/images \
    --output_dir data/

4️⃣ Multimodal Feature Engineering (train_clip_siglip_blend_v2.py)

train_clip_siglip_blend_v2

This script blends four embedding vectors:

CLIP image

CLIP text

SigLIP image

SigLIP text

Plus:

✔ Cross-modal similarity

Cosine similarity

Mean/Std/Max/Min difference

Elementwise products

✔ Statistical comparisons

Norms, means, std-devs of each embedding
Image vs text statistical deltas
Image vs text statistical ratios

✔ SigLIP similarity scores included

Total dimensionality ≈ ~2600+

5️⃣ Advanced Outlier Handling

Integrated inside training script.

✔ Multi-level outlier processing:

Price clipping (1st–99th percentile)

Winsorization on 2600-dim features

Isolation Forest (optional anomaly detection)

Quantile Transformation for heavy-tailed distributions

RobustScaler normalization

This dramatically stabilizes training and improves SMAPE.

6️⃣ Model Architecture
🧠 Residual MLP Regressor

Backbone: 1024 → 512 → 256

Two residual blocks

GELU activations

Dropout=0.20

Weight Decay + AdamW

Gradient clipping

EMA (Exponential Moving Average) for stable predictions

Training:

5-fold K-Fold

Huber Loss + L1

ReduceLROnPlateau

Automatic Mixed Precision (AMP)

7️⃣ Training the Model
▶ Run full Blend-V2 Training:
python train_clip_siglip_blend_v2.py \
    --train_data data/processed_train.parquet \
    --test_data data/processed_test.parquet \
    --out_dir models/clip_siglip_blend_v2 \
    --device cuda:0

Output files:

fold_*_best.pt (best checkpoints)

preprocessing.pkl

config.json

oof.csv

test_predictions.csv

📊 Performance
Metric	Score
SMAPE (public leaderboard)	40.858%
Rank	#14

Reasons for strong performance:

CLIP + SigLIP cross-modal synergy

Rich feature engineering

Aggressive outlier handling

EMA-stabilized MLP

🖼 Example: Complete Pipeline
# 1. Preprocess
python preprocess.py

# 2. Download images
python utils.py --download_images

# 3. Extract CLIP + SigLIP
python extract_clip_features.py
python extract_siglip_features.py

# 4. Train final blended model
python train_clip_siglip_blend_v2.py

📁 Directory Expectations
data/
  train.csv
  test.csv
  images/
  processed_train.parquet
  processed_test.parquet
  clip_text_embeddings_train.npy
  ...
models/
  clip_siglip_blend_v2/
      fold_1_best.pt
      test_predictions.csv

🙌 Credits

This work was developed as part of the Amazon ML Challenge 2025.
The pipeline is fully reproducible and uses only local, license-compliant models.
