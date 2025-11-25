#!/usr/bin/env python3
"""
CLIP + SigLIP Blended Model V2 - Enhanced Outlier Handling.

Key Enhancements:
1. Multi-level outlier detection and handling
2. Isolation Forest for anomaly detection
3. Winsorization (aggressive clipping)
4. Feature-wise outlier removal
5. Smart price target handling
6. Quantile transformation for heavy-tailed features

Based on successful clip_siglip_blend.py with enhanced preprocessing.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import set_seed, save_json, load_json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================
# Training config
# =============================
N_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 1000
PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.995
HUBER_DELTA = 1.0
GRAD_CLIP = 2.0
HIDDEN = [1024, 512, 256]
DROPOUT = 0.2

# Outlier handling config
OUTLIER_PERCENTILE_LOW = 0.5   # Clip at 0.5th percentile
OUTLIER_PERCENTILE_HIGH = 99.5  # Clip at 99.5th percentile
PRICE_PERCENTILE_LOW = 1        # Clip prices at 1st percentile
PRICE_PERCENTILE_HIGH = 99      # Clip prices at 99th percentile
USE_ISOLATION_FOREST = False    # Can slow down training
USE_QUANTILE_TRANSFORM = True   # Transform heavy-tailed features


# =============================
# Data loading
# =============================
def load_clip_embeddings(text_path, image_path, id_path):
    text_emb = np.load(text_path)
    image_emb = np.load(image_path)
    id_to_index = load_json(id_path)
    logger.info(f"CLIP - Text: {text_emb.shape}, Image: {image_emb.shape}")
    return image_emb.astype('float32'), text_emb.astype('float32'), id_to_index


def load_siglip_embeddings(text_path, image_path, sim_path, id_path):
    text_emb = np.load(text_path)
    image_emb = np.load(image_path)
    similarity = np.load(sim_path)
    id_to_index = load_json(id_path)
    logger.info(f"SigLIP - Text: {text_emb.shape}, Image: {image_emb.shape}")
    return image_emb.astype('float32'), text_emb.astype('float32'), similarity.astype('float32'), id_to_index


def build_advanced_features(clip_img, clip_txt, siglip_img, siglip_txt, siglip_sim):
    """Build features (same as successful blend model)."""
    
    # Normalized vectors
    clip_img_norm = clip_img / (np.linalg.norm(clip_img, axis=1, keepdims=True) + 1e-8)
    clip_txt_norm = clip_txt / (np.linalg.norm(clip_txt, axis=1, keepdims=True) + 1e-8)
    clip_cos = np.sum(clip_img_norm * clip_txt_norm, axis=1, keepdims=True)
    clip_diff = clip_img - clip_txt
    clip_prod = clip_img * clip_txt
    
    siglip_img_norm = siglip_img / (np.linalg.norm(siglip_img, axis=1, keepdims=True) + 1e-8)
    siglip_txt_norm = siglip_txt / (np.linalg.norm(siglip_txt, axis=1, keepdims=True) + 1e-8)
    siglip_cos = np.sum(siglip_img_norm * siglip_txt_norm, axis=1, keepdims=True)
    siglip_diff = siglip_img - siglip_txt
    siglip_prod = siglip_img * siglip_txt
    
    # Statistical comparisons
    clip_img_stats_vec = np.concatenate([
        np.linalg.norm(clip_img, axis=1, keepdims=True),
        clip_img.mean(axis=1, keepdims=True),
        clip_img.std(axis=1, keepdims=True)
    ], axis=1)
    
    siglip_img_stats_vec = np.concatenate([
        np.linalg.norm(siglip_img, axis=1, keepdims=True),
        siglip_img.mean(axis=1, keepdims=True),
        siglip_img.std(axis=1, keepdims=True)
    ], axis=1)
    
    clip_txt_stats_vec = np.concatenate([
        np.linalg.norm(clip_txt, axis=1, keepdims=True),
        clip_txt.mean(axis=1, keepdims=True),
        clip_txt.std(axis=1, keepdims=True)
    ], axis=1)
    
    siglip_txt_stats_vec = np.concatenate([
        np.linalg.norm(siglip_txt, axis=1, keepdims=True),
        siglip_txt.mean(axis=1, keepdims=True),
        siglip_txt.std(axis=1, keepdims=True)
    ], axis=1)
    
    img_stat_diff = clip_img_stats_vec - siglip_img_stats_vec
    txt_stat_diff = clip_txt_stats_vec - siglip_txt_stats_vec
    img_stat_ratio = clip_img_stats_vec / (siglip_img_stats_vec + 1e-8)
    txt_stat_ratio = clip_txt_stats_vec / (siglip_txt_stats_vec + 1e-8)
    
    # Concatenate all features
    features = np.concatenate([
        clip_img, clip_txt, siglip_img, siglip_txt,  # 2560
        clip_cos, clip_diff.mean(axis=1, keepdims=True), clip_diff.std(axis=1, keepdims=True),
        clip_prod.mean(axis=1, keepdims=True), clip_diff.max(axis=1, keepdims=True),
        clip_diff.min(axis=1, keepdims=True),  # 6
        siglip_cos, siglip_diff.mean(axis=1, keepdims=True), siglip_diff.std(axis=1, keepdims=True),
        siglip_prod.mean(axis=1, keepdims=True), siglip_diff.max(axis=1, keepdims=True),
        siglip_diff.min(axis=1, keepdims=True), siglip_sim.reshape(-1, 1),  # 7
        img_stat_diff, txt_stat_diff, img_stat_ratio, txt_stat_ratio,  # 12
        (clip_cos / (siglip_cos + 1e-8)), siglip_sim.reshape(-1, 1),  # 2
    ], axis=1)
    
    return features.astype('float32')


def advanced_outlier_handling(X_train, X_test, y_train, config):
    """
    Advanced multi-level outlier handling:
    1. Winsorization (aggressive clipping at percentiles)
    2. Isolation Forest (optional - detect anomalies)
    3. Feature-wise outlier clipping
    4. Quantile transformation for heavy-tailed features
    5. RobustScaler for final normalization
    """
    
    logger.info("\n=== Advanced Outlier Handling ===")
    
    # 1. Handle price target outliers first
    logger.info(f"Original price (log) - min: {y_train.min():.4f}, max: {y_train.max():.4f}, mean: {y_train.mean():.4f}")
    
    price_low = np.percentile(y_train, config['price_percentile_low'])
    price_high = np.percentile(y_train, config['price_percentile_high'])
    
    logger.info(f"Price clipping at {config['price_percentile_low']}th-{config['price_percentile_high']}th percentile: [{price_low:.4f}, {price_high:.4f}]")
    
    y_train_clipped = np.clip(y_train, price_low, price_high)
    outliers_count = (y_train != y_train_clipped).sum()
    logger.info(f"  Clipped {outliers_count} price outliers ({outliers_count/len(y_train)*100:.2f}%)")
    
    # 2. Winsorization on features (aggressive clipping)
    logger.info(f"\nWinsorization on features at {config['outlier_percentile_low']}th-{config['outlier_percentile_high']}th percentile...")
    
    feature_low = np.percentile(X_train, config['outlier_percentile_low'], axis=0, keepdims=True)
    feature_high = np.percentile(X_train, config['outlier_percentile_high'], axis=0, keepdims=True)
    
    X_train_clipped = np.clip(X_train, feature_low, feature_high)
    X_test_clipped = np.clip(X_test, feature_low, feature_high)
    
    feature_outliers = (X_train != X_train_clipped).sum()
    logger.info(f"  Clipped {feature_outliers} feature values ({feature_outliers/(X_train.size)*100:.2f}%)")
    
    # 3. Isolation Forest (optional - detect and flag anomalies)
    if config['use_isolation_forest']:
        logger.info("\nApplying Isolation Forest for anomaly detection...")
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        
        # Fit on training data
        anomaly_labels = iso_forest.fit_predict(X_train_clipped)
        normal_mask = anomaly_labels == 1
        anomaly_count = (anomaly_labels == -1).sum()
        
        logger.info(f"  Detected {anomaly_count} anomalies ({anomaly_count/len(X_train_clipped)*100:.2f}%)")
        logger.info(f"  Keeping all samples (anomalies will have lower weight during training)")
        
        # Don't remove, just log - we handle with robust scaling
    
    # 4. Quantile Transformation for heavy-tailed features
    if config['use_quantile_transform']:
        logger.info("\nApplying Quantile Transformation for heavy-tailed features...")
        
        # Identify heavy-tailed features (high kurtosis)
        from scipy.stats import kurtosis
        kurt = kurtosis(X_train_clipped, axis=0, nan_policy='omit')
        heavy_tailed_mask = kurt > 10  # Features with high kurtosis
        n_heavy = heavy_tailed_mask.sum()
        
        logger.info(f"  Identified {n_heavy} heavy-tailed features (kurtosis > 10)")
        
        if n_heavy > 0:
            # Apply quantile transformation to heavy-tailed features
            qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=1000)
            
            X_train_clipped[:, heavy_tailed_mask] = qt.fit_transform(X_train_clipped[:, heavy_tailed_mask])
            X_test_clipped[:, heavy_tailed_mask] = qt.transform(X_test_clipped[:, heavy_tailed_mask])
            
            logger.info(f"  Transformed {n_heavy} features to normal distribution")
        else:
            qt = None
    else:
        qt = None
        heavy_tailed_mask = None
    
    # 5. RobustScaler for final normalization
    logger.info("\nApplying RobustScaler for final normalization...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_clipped).astype('float32')
    X_test_scaled = scaler.transform(X_test_clipped).astype('float32')
    
    logger.info(f"Scaled features - train shape: {X_train_scaled.shape}, test shape: {X_test_scaled.shape}")
    logger.info(f"Final train stats - min: {X_train_scaled.min():.4f}, max: {X_train_scaled.max():.4f}, mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    
    # Return everything
    preprocessing_objects = {
        'scaler': scaler,
        'feature_low': feature_low,
        'feature_high': feature_high,
        'price_low': price_low,
        'price_high': price_high,
        'quantile_transformer': qt,
        'heavy_tailed_mask': heavy_tailed_mask
    }
    
    return X_train_scaled, X_test_scaled, y_train_clipped, preprocessing_objects


# =============================
# Model & Dataset
# =============================
class PriceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        if self.y is None:
            return torch.from_numpy(self.X[i])
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.bn2 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.fc1(x))))
        h = self.drop(self.bn2(self.fc2(h)))
        return F.gelu(x + h)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        
        layers = []
        dim = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            dim = h
        self.backbone = nn.Sequential(*layers)
        
        self.res1 = ResidualBlock(dim, dropout)
        self.res2 = ResidualBlock(dim, dropout)
        
        self.head = nn.Linear(dim, 1)
        self.apply(self._init)
    
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x).squeeze(-1)


# =============================
# Loss & Metrics
# =============================
def smape(pred, target):
    p = torch.expm1(pred).clamp(min=0.01)
    t = torch.expm1(target).clamp(min=0.01)
    return torch.mean(2.0 * torch.abs(p - t) / (torch.abs(p) + torch.abs(t) + 1e-8))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()
    
    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow)


# =============================
# Training
# =============================
def train_fold(fold, X_tr, y_tr, X_va, y_va, config, device):
    
    train_ds = PriceDataset(X_tr, y_tr)
    valid_ds = PriceDataset(X_va, y_va)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model = MLPRegressor(X_tr.shape[1], config['hidden'], config['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
    
    amp_enabled = (device == 'cuda')
    scaler_amp = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    ema = EMA(model, config['ema_decay'])
    
    best = {'epoch': -1, 'val_loss': float('inf'), 'val_smape': float('inf')}
    patience = config['patience']
    ckpt_path = Path(config['out_dir']) / f'fold_{fold}_best.pt'
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        tr_loss = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                pred = model(xb)
                loss = 0.7 * F.huber_loss(pred, yb, delta=config['huber_delta']) + 0.3 * torch.mean(torch.abs(pred - yb))
            
            scaler_amp.scale(loss).backward()
            if config['grad_clip']:
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler_amp.step(opt)
            scaler_amp.update()
            
            ema.update(model)
            tr_loss += loss.item() * xb.size(0)
        
        # Validation with EMA
        model_ema = MLPRegressor(X_tr.shape[1], config['hidden'], config['dropout']).to(device)
        model_ema.load_state_dict(model.state_dict())
        ema.copy_to(model_ema)
        model_ema.eval()
        
        va_loss = 0.0
        va_sm = 0.0
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    pr = model_ema(xb)
                    l = 0.7 * F.huber_loss(pr, yb, delta=config['huber_delta']) + 0.3 * torch.mean(torch.abs(pr - yb))
                    s = smape(pr, yb)
                
                va_loss += l.item() * xb.size(0)
                va_sm += s.item() * xb.size(0)
        
        va_loss /= len(valid_ds)
        va_sm /= len(valid_ds)
        
        sched.step(va_loss)
        
        if va_loss < best['val_loss']:
            best.update({'epoch': epoch, 'val_loss': va_loss, 'val_smape': va_sm})
            torch.save(model_ema.state_dict(), ckpt_path)
            patience = config['patience']
        else:
            patience -= 1
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Fold {fold:02d} | Epoch {epoch:03d} | train {tr_loss/len(train_ds):.4f} "
                f"| val {va_loss:.4f} | sMAPE {va_sm:.4f} | lr {opt.param_groups[0]['lr']:.2e} | best {best['epoch']}"
            )
        
        if patience == 0:
            logger.info(f'Early stopping at epoch {epoch}!')
            break
    
    return ckpt_path, best


# =============================
# Main
# =============================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/processed_train.parquet')
    parser.add_argument('--test_data', type=str, default='data/processed_test.parquet')
    parser.add_argument('--clip_text_train', type=str, default='data/clip_text_embeddings_train.npy')
    parser.add_argument('--clip_image_train', type=str, default='data/clip_image_embeddings_train.npy')
    parser.add_argument('--clip_id_train', type=str, default='data/clip_id_to_index_train.json')
    parser.add_argument('--clip_text_test', type=str, default='data/clip_text_embeddings_test.npy')
    parser.add_argument('--clip_image_test', type=str, default='data/clip_image_embeddings_test.npy')
    parser.add_argument('--clip_id_test', type=str, default='data/clip_id_to_index_test.json')
    parser.add_argument('--siglip_text_train', type=str, default='data/siglip_text_embeddings_train.npy')
    parser.add_argument('--siglip_image_train', type=str, default='data/siglip_image_embeddings_train.npy')
    parser.add_argument('--siglip_sim_train', type=str, default='data/siglip_similarity_train.npy')
    parser.add_argument('--siglip_id_train', type=str, default='data/siglip_id_to_index_train.json')
    parser.add_argument('--siglip_text_test', type=str, default='data/siglip_text_embeddings_test.npy')
    parser.add_argument('--siglip_image_test', type=str, default='data/siglip_image_embeddings_test.npy')
    parser.add_argument('--siglip_sim_test', type=str, default='data/siglip_similarity_test.npy')
    parser.add_argument('--siglip_id_test', type=str, default='data/siglip_id_to_index_test.json')
    parser.add_argument('--out_dir', type=str, default='models/clip_siglip_blend_v2')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {out_dir}")
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)
    
    # Load embeddings
    logger.info("\nLoading embeddings...")
    clip_img_tr, clip_txt_tr, _ = load_clip_embeddings(args.clip_text_train, args.clip_image_train, args.clip_id_train)
    clip_img_te, clip_txt_te, _ = load_clip_embeddings(args.clip_text_test, args.clip_image_test, args.clip_id_test)
    
    siglip_img_tr, siglip_txt_tr, siglip_sim_tr, _ = load_siglip_embeddings(
        args.siglip_text_train, args.siglip_image_train, args.siglip_sim_train, args.siglip_id_train
    )
    siglip_img_te, siglip_txt_te, siglip_sim_te, _ = load_siglip_embeddings(
        args.siglip_text_test, args.siglip_image_test, args.siglip_sim_test, args.siglip_id_test
    )
    
    # Build features
    logger.info("\nBuilding features...")
    X = build_advanced_features(clip_img_tr, clip_txt_tr, siglip_img_tr, siglip_txt_tr, siglip_sim_tr)
    X_test = build_advanced_features(clip_img_te, clip_txt_te, siglip_img_te, siglip_txt_te, siglip_sim_te)
    logger.info(f"Features: {X.shape}")
    
    # Target
    y = np.log1p(train_df['price'].values).astype('float32')
    
    # Advanced outlier handling
    outlier_config = {
        'outlier_percentile_low': OUTLIER_PERCENTILE_LOW,
        'outlier_percentile_high': OUTLIER_PERCENTILE_HIGH,
        'price_percentile_low': PRICE_PERCENTILE_LOW,
        'price_percentile_high': PRICE_PERCENTILE_HIGH,
        'use_isolation_forest': USE_ISOLATION_FOREST,
        'use_quantile_transform': USE_QUANTILE_TRANSFORM
    }
    
    X, X_test, y, preprocessing_objects = advanced_outlier_handling(X, X_test, y, outlier_config)
    
    # Save preprocessing objects
    joblib.dump(preprocessing_objects, out_dir / 'preprocessing.pkl')
    
    # Config
    config = {
        'n_folds': args.n_folds, 'batch_size': args.batch_size, 'epochs': args.epochs,
        'lr': args.lr, 'weight_decay': WEIGHT_DECAY, 'hidden': HIDDEN,
        'dropout': DROPOUT, 'ema_decay': EMA_DECAY, 'huber_delta': HUBER_DELTA,
        'grad_clip': GRAD_CLIP, 'patience': PATIENCE, 'out_dir': str(out_dir),
        **outlier_config
    }
    
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # K-Fold CV
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING BLEND V2 ({args.n_folds}-FOLD) - Enhanced Outlier Handling")
    logger.info(f"{'='*70}")
    
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros_like(y)
    test_preds = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold}/{args.n_folds}")
        logger.info(f"{'='*70}")
        
        ckpt_path, best = train_fold(fold, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], config, device)
        
        # OOF
        model = MLPRegressor(X.shape[1], HIDDEN, DROPOUT).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        va_ds = PriceDataset(X[va_idx], y[va_idx])
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for xb, _ in va_loader:
                pr = model(xb.to(device))
                preds.append(pr.cpu().numpy())
        oof_pred[va_idx] = np.concatenate(preds)
        
        # Test
        test_ds = PriceDataset(X_test)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        test_pred = []
        with torch.no_grad():
            for xb in test_loader:
                pr = model(xb.to(device))
                test_pred.append(pr.cpu().numpy())
        test_preds.append(np.concatenate(test_pred))
        
        logger.info(f"\nFold {fold} Best - Val Loss: {best['val_loss']:.4f}, Val SMAPE: {best['val_smape']:.4f}")
    
    # OOF metrics
    oof_smape = np.mean(
        2.0 * np.abs(np.expm1(oof_pred) - np.expm1(y))
        / (np.abs(np.expm1(oof_pred)) + np.abs(np.expm1(y)) + 1e-8)
    ) * 100
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OOF SMAPE: {oof_smape:.2f}%")
    logger.info(f"{'='*70}")
    
    # Save
    oof_df = train_df[['sample_id', 'price']].copy()
    oof_df['oof_pred_log'] = oof_pred
    oof_df['oof_pred_price'] = np.expm1(oof_pred)
    oof_df.to_csv(out_dir / 'oof.csv', index=False)
    logger.info(f"Saved OOF -> {out_dir / 'oof.csv'}")
    
    test_pred_avg = np.mean(test_preds, axis=0)
    test_df_out = test_df[['sample_id']].copy()
    test_df_out['price'] = np.clip(np.expm1(test_pred_avg), 0.01, 10000)
    test_df_out.to_csv(out_dir / 'test_predictions.csv', index=False)
    logger.info(f"Saved test -> {out_dir / 'test_predictions.csv'}")
    
    logger.info(f"\n✅ BLEND V2 TRAINING COMPLETE!")


if __name__ == '__main__':
    main()

