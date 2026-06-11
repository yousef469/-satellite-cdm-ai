#!/usr/bin/env python3
"""
Threshold calibration sweep for production deployment.

Sweeps risk thresholds and finds the optimal operating point
balancing maneuver savings vs missed real risks.

Usage:
    python calibrate.py
"""
import json
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_PATH, DEVICE, BATCH_SIZE, ESA_DATA_PATH,
    ALGORITHM_AMBER_THRESHOLD, ALGORITHM_RED_THRESHOLD, ALGORITHM_CRITICAL_THRESHOLD,
    TRAIN_SPLIT, VAL_SPLIT,
)
from model import CDMRiskModel
from train import CDMDataset, pad_collate_fn
from esa_loader import ESADataset, load_esa_kelvins, events_to_tensors
from algorithm import ConstellationDecisionEngine, AlgorithmConfig, AlertThresholds, SatelliteState

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('calibrate')


def load_model_and_data():
    logger.info(f"Loading model from {CHECKPOINT_PATH}...")
    model = CDMRiskModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"Loaded checkpoint epoch {ckpt['epoch']}")

    logger.info(f"Loading data from {ESA_DATA_PATH}...")
    events, features = load_esa_kelvins(ESA_DATA_PATH)
    sequences, labels = events_to_tensors(events, features)

    n = len(sequences)
    n_val = int(n * VAL_SPLIT)
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]

    val_seqs = [sequences[i] for i in val_idx]
    val_lbls = labels[val_idx]
    return model, val_seqs, val_lbls, features


@torch.no_grad()
def get_risk_scores(model, sequences):
    scores = []
    model.eval()
    for seq in sequences:
        seq_t = seq.unsqueeze(0).to(DEVICE)
        time_t = seq_t[:, :, 0].to(DEVICE)
        mask = None
        proto, risk, d = model(seq_t, time_t, mask)
        prob = torch.sigmoid(risk).item()
        scores.append(prob)
    return np.array(scores)


def sweep_thresholds(risk_scores, labels):
    results = []
    for th in np.arange(0.05, 0.96, 0.05):
        preds = (risk_scores > th).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        fpr = fp / max(fp + tn, 1)
        savings = 100 * (1 - (tp + fp) / max(len(labels), 1))
        misses = int(fn)

        results.append({
            'threshold': round(th, 2),
            'recall': round(recall, 3),
            'precision': round(precision, 3),
            'f1': round(f1, 3),
            'fpr': round(fpr, 4),
            'savings_pct': round(savings, 1),
            'maneuvers': int(tp + fp),
            'missed_risks': misses,
        })

    return results


def print_results(results):
    print(f"\n{'='*80}")
    print(f"{'TH':<6} {'RECALL':<8} {'PREC':<8} {'F1':<8} {'FPR':<8} {'SAVINGS':<10} {'MANEUVERS':<12} {'MISSES':<8}")
    print(f"{'-'*80}")
    best_f1 = max(results, key=lambda r: r['f1'])
    best_f1_idx = results.index(best_f1)
    for i, r in enumerate(results):
        marker = " ◀ BEST F1" if i == best_f1_idx else ""
        print(f"{r['threshold']:<6} {r['recall']:<8} {r['precision']:<8} {r['f1']:<8} {r['fpr']:<8} {r['savings_pct']:<10} {r['maneuvers']:<12} {r['missed_risks']:<8}{marker}")
    print(f"{'='*80}\n")
    return results


def main():
    model, val_seqs, val_lbls, features = load_model_and_data()

    logger.info("Computing risk scores on validation set...")
    risk_scores = get_risk_scores(model, val_seqs)
    labels = val_lbls.numpy()

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    logger.info(f"Val set: {len(labels)} events ({n_pos:.0f} pos, {n_neg:.0f} neg)")

    results = sweep_thresholds(risk_scores, labels)
    print_results(results)

    best_f1 = max(results, key=lambda r: r['f1'])
    best_f1_idx = results.index(best_f1)

    print(f"\nRecommended operating point (best F1={best_f1['f1']}):")
    print(f"  Threshold:    {best_f1['threshold']}")
    print(f"  Recall:       {best_f1['recall']}")
    print(f"  Precision:    {best_f1['precision']}")
    print(f"  Maneuvers:    {best_f1['maneuvers']} ({best_f1['savings_pct']}% savings)")
    print(f"  Missed risks: {best_f1['missed_risks']}")
    print(f"\nCalibrated alert thresholds:")
    print(f"  AMBER:    {max(0.10, best_f1['threshold'] - 0.15):.2f}")
    print(f"  RED:      {best_f1['threshold']:.2f}")
    print(f"  CRITICAL: {min(0.95, best_f1['threshold'] + 0.25):.2f}")


if __name__ == '__main__':
    main()
