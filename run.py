#!/usr/bin/env python3
"""
Production training pipeline for satellite collision avoidance.

Usage:
    # Download real data (~9GB ESA Kelvins)
    python download_data.py

    # Train on ESA data (production)
    python run.py --mode train --data esa --epochs 100

    # Train on synthetic data (quick test)
    python run.py --mode train --data synthetic

    # Evaluate best checkpoint
    python run.py --mode eval

    # Run full constellation demo
    python run.py --mode demo

Training strategy (two-phase):
    Phase 1 — Head warmup (5 epochs):
        Freeze backbone, train only risk_head (43 params) with BCE.
        LR: 1e-2, linear warmup over 1 epoch, cosine to 0.
        This learns the basic Pc → risk mapping.

    Phase 2 — Full fine-tune (50-200 epochs):
        Unfreeze backbone, train all 40K params.
        Backbone LR: 5e-4, Head LR: 1e-4 (prevent forgetting).
        Linear warmup over 3 epochs, cosine to 0.
        Contrastive loss ramps from 0 → 0.05 over first 10 epochs.
        Checkpoint by P@10 on validation set.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from config import (
    CHECKPOINT_DIR, CHECKPOINT_PATH, DEVICE, BATCH_SIZE, NUM_WORKERS,
    PHASE1_EPOCHS, PHASE2_EPOCHS, PHASE1_LR, PHASE2_WEIGHT_DECAY,
    PHASE2_BACKBONE_LR, PHASE2_HEAD_LR, PHASE2_WARMUP_EPOCHS,
    BCE_POS_WEIGHT_CAP, CONTRASTIVE_WEIGHT, CONTRASTIVE_RAMP_EPOCHS,
    ORTHOGONALITY_WEIGHT, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    NUM_SYNTHETIC_EVENTS, RISK_RATIO, ESA_DATA_PATH,
    ALGORITHM_AMBER_THRESHOLD, ALGORITHM_RED_THRESHOLD, ALGORITHM_CRITICAL_THRESHOLD,
    ALGORITHM_TEMPORAL_WINDOW, ALGORITHM_TEMPORAL_MIN_POSITIVE,
    ALGORITHM_FUEL_RESERVE_FRACTION, ALGORITHM_MIN_LEAD_HOURS, ALGORITHM_MAX_LEAD_HOURS,
)
from cdm_generator import generate_dataset, generate_cdm_sequence
from model import CDMRiskModel, ConjunctionContrastiveLoss, count_params
from train import CDMDataset, pad_collate_fn
from evaluate import precision_recall_curve, compute_maneuver_savings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('cdm_ai')


# ── Data Loading ────────────────────────────────────────────────────────────

def load_data(source: str, max_samples: int = 0, augment_methods: str = '',
              aug_copies: int = 5, aug_synth: int = 10000):
    """Load dataset. Returns (train_loader, val_loader, test_loader, pos_weight)."""
    if source == 'esa':
        if augment_methods:
            return _load_esa_data_augmented(max_samples, augment_methods, aug_copies, aug_synth)
        return _load_esa_data(max_samples)
    else:
        return _load_synthetic_data(max_samples)


def _load_synthetic_data(max_samples: int):
    n = max_samples if max_samples > 0 else NUM_SYNTHETIC_EVENTS
    logger.info(f"Generating {n} synthetic events (risk_ratio={RISK_RATIO})...")
    sequences, labels = generate_dataset(n, RISK_RATIO)

    n_total = len(sequences)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)

    train_ds = CDMDataset(sequences[:n_train], labels[:n_train])
    val_ds = CDMDataset(sequences[n_train:n_train + n_val], labels[n_train:n_train + n_val])
    test_ds = CDMDataset(sequences[n_train + n_val:], labels[n_train + n_val:])

    pos_count = labels[:n_train].sum().item()
    neg_count = n_train - pos_count
    logger.info(f"  Train: {len(train_ds)} ({pos_count} pos, {neg_count} neg)")
    logger.info(f"  Val:   {len(val_ds)} ({labels[n_train:n_train+n_val].sum().item()} pos)")
    logger.info(f"  Test:  {len(test_ds)} ({labels[n_train+n_val:].sum().item()} pos)")

    return _make_loaders(train_ds, val_ds, test_ds), neg_count / max(pos_count, 1)


def _load_esa_data(max_samples: int):
    logger.info(f"Loading ESA Kelvins data from {ESA_DATA_PATH}...")
    try:
        from esa_loader import ESADataset, load_esa_kelvins, events_to_tensors
        events, features = load_esa_kelvins(ESA_DATA_PATH, max_events=max_samples if max_samples > 0 else None)
        sequences, labels = events_to_tensors(events, features)
        logger.info(f"  Loaded {len(sequences)} sequences ({labels.sum().item():.0f} positive)")
    except FileNotFoundError:
        logger.error(f"ESA data not found at {ESA_DATA_PATH}.")
        logger.error("Run: python download_data.py")
        sys.exit(1)

    n = len(sequences)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = CDMDataset([sequences[i] for i in train_idx], labels[train_idx])
    val_ds = CDMDataset([sequences[i] for i in val_idx], labels[val_idx])
    test_ds = CDMDataset([sequences[i] for i in test_idx], labels[test_idx])

    pos_count = labels[train_idx].sum().item()
    neg_count = len(train_idx) - pos_count
    logger.info(f"  Train: {len(train_ds)} ({pos_count} pos, {neg_count} neg)")
    logger.info(f"  Val:   {len(val_ds)} ({labels[val_idx].sum().item()} pos)")
    logger.info(f"  Test:  {len(test_ds)} ({labels[test_idx].sum().item()} pos)")

    return _make_loaders(train_ds, val_ds, test_ds), neg_count / max(pos_count, 1)


def _load_esa_data_augmented(max_samples: int, augment_methods: str, aug_copies: int, aug_synth: int):
    logger.info(f"Loading ESA Kelvins data from {ESA_DATA_PATH}...")
    try:
        from esa_loader import load_esa_kelvins, events_to_tensors
        events, features = load_esa_kelvins(ESA_DATA_PATH, max_events=max_samples if max_samples > 0 else None)
    except FileNotFoundError:
        logger.error(f"ESA data not found at {ESA_DATA_PATH}.")
        sys.exit(1)

    # Split event IDs, then augment only training events
    event_ids = list(events.keys())
    n = len(event_ids)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    np.random.shuffle(event_ids)
    train_ids = event_ids[:n_train]
    val_ids = event_ids[n_train:n_train + n_val]
    test_ids = event_ids[n_train + n_val:]

    train_events = {eid: events[eid] for eid in train_ids}
    val_events = {eid: events[eid] for eid in val_ids}
    test_events = {eid: events[eid] for eid in test_ids}

    methods = [m.strip() for m in augment_methods.split(',')]
    for m in methods:
        if m == 'mixup':
            from data_augment import augment_mixup
            train_events = augment_mixup(train_events, features, num_synthetic=aug_synth)
        elif m == 'jitter':
            from data_augment import augment_jitter
            train_events = augment_jitter(train_events, features, num_copies=aug_copies)

    train_seqs, train_labels = events_to_tensors(train_events, features)
    val_seqs, val_lbls = events_to_tensors(val_events, features)
    test_seqs, test_lbls = events_to_tensors(test_events, features)

    train_ds = CDMDataset(train_seqs, train_labels)
    val_ds = CDMDataset(val_seqs, val_lbls)
    test_ds = CDMDataset(test_seqs, test_lbls)

    logger.info(f"  Train (augmented): {len(train_ds)} ({train_labels.sum().item():.0f} pos)")
    logger.info(f"  Val:   {len(val_ds)} ({val_lbls.sum().item():.0f} pos)")
    logger.info(f"  Test:  {len(test_ds)} ({test_lbls.sum().item():.0f} pos)")

    pos_count = train_labels.sum().item()
    neg_count = len(train_labels) - pos_count
    return _make_loaders(train_ds, val_ds, test_ds), neg_count / max(pos_count, 1)


def _make_loaders(train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=pad_collate_fn, num_workers=NUM_WORKERS,
                               pin_memory=DEVICE == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=pad_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=pad_collate_fn, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader


# ── LR Scheduler with Warmup ───────────────────────────────────────────────

class WarmupReduceLR:
    """Per-batch warmup (linear) then per-epoch ReduceLROnPlateau."""
    def __init__(self, optimizer, warmup_epochs: int,
                 batches_per_epoch: int, factor=0.5, patience=3, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * batches_per_epoch
        self.current_step = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience,
            min_lr=min_lr, threshold=1e-3, verbose=False)
        self._warmup_done = False

    def step_batch(self):
        """Call per batch during warmup."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            factor = self.current_step / max(self.warmup_steps, 1)
            for g in self.optimizer.param_groups:
                g['lr'] = g.get('base_lr', 1e-3) * max(factor, 0.01)
        else:
            self._warmup_done = True

    def step_epoch(self, metric):
        """Call once per epoch after warmup with validation metric."""
        if self._warmup_done and metric is not None:
            self.plateau.step(metric)

    def in_warmup(self):
        return self.current_step <= self.warmup_steps


# ── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, pos_weight: float):
    model.eval()
    all_probs, all_labels = [], []

    for cdm, lbl, mask in loader:
        cdm = cdm.to(DEVICE)
        lbl = lbl.to(DEVICE)
        mask = mask.to(DEVICE)
        _, risk, _ = model(cdm, cdm[:, :, 0], mask)
        all_probs.append(torch.sigmoid(risk).cpu())
        all_labels.append(lbl.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    n_pos = labels.sum().item()

    # Sort by risk score descending
    sorted_idx = torch.argsort(probs, descending=True)
    sorted_labels = labels[sorted_idx]

    # P@K
    p10 = sorted_labels[:10].float().mean().item() if len(sorted_labels) >= 10 else 0.0
    p100 = sorted_labels[:100].float().mean().item() if len(sorted_labels) >= 100 else 0.0

    # Accuracy / Recall at 0.5 threshold
    acc = ((probs > 0.5) == labels).float().mean().item()
    tp = ((probs > 0.5) & (labels == 1)).sum().item()
    fn = ((probs <= 0.5) & (labels == 1)).sum().item()
    recall = tp / max(tp + fn, 1)

    # Best F1 across thresholds
    f1_scores = []
    for th in np.linspace(0.05, 0.95, 19):
        preds = (probs > th).float()
        _tp = ((preds == 1) & (labels == 1)).sum().item()
        _fp = ((preds == 1) & (labels == 0)).sum().item()
        _fn = ((preds == 0) & (labels == 1)).sum().item()
        prec = _tp / max(_tp + _fp, 1)
        rec = _tp / max(_tp + _fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1_scores.append((th, f1, prec, rec))

    best_f1 = max(f1_scores, key=lambda x: x[1])

    return {
        'p10': p10,
        'p100': p100,
        'accuracy': acc,
        'recall': recall,
        'best_f1_threshold': best_f1[0],
        'best_f1': best_f1[1],
        'best_f1_precision': best_f1[2],
        'best_f1_recall': best_f1[3],
        'n_pos': n_pos,
        'n_total': len(labels),
        'probs': probs,
        'labels': labels,
    }


# ── Training ────────────────────────────────────────────────────────────────

def train_phase(model, loader, val_loader, opt, scheduler, bce_loss, cont_loss,
                 phase_name: str, epochs: int, epoch_offset: int,
                 target_contrastive_weight: float, orth_weight: float,
                 contrastive_ramp_epochs: int = 0):
    """Train for N epochs. Contrastive weight ramps from 0 to target over ramp_epochs."""
    best_p10 = 0.0
    metrics_history = []

    for ep in range(epochs):
        model.train()
        total_loss = total_bce = total_cont = total_orth = 0
        n_batches = 0
        t0 = time.perf_counter()

        # Ramp contrastive weight
        if contrastive_ramp_epochs > 0:
            cw = target_contrastive_weight * min(1.0, (ep + 1) / contrastive_ramp_epochs)
        else:
            cw = target_contrastive_weight

        for cdm, lbl, mask in loader:
            cdm = cdm.to(DEVICE)
            lbl = lbl.to(DEVICE)
            mask = mask.to(DEVICE)

            proto, risk, d = model(cdm, cdm[:, :, 0], mask)
            loss_bce = bce_loss(risk, lbl)
            loss_cont = cont_loss(d, lbl)
            loss_orth = model.get_aux_losses()
            loss = loss_bce + cw * loss_cont + orth_weight * loss_orth

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step_batch()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_cont += loss_cont.item()
            total_orth += loss_orth.item()
            n_batches += 1

        epoch = ep + epoch_offset + 1
        train_time = time.perf_counter() - t0

        metrics = evaluate(model, val_loader, 0)
        p10 = metrics['p10']
        scheduler.step_epoch(metric=p10)

        current_lrs = [g['lr'] for g in opt.param_groups]
        logger.info(
            f"E{epoch:3d}/{epoch_offset + epochs} | "
            f"L={total_loss / n_batches:.4f} (bce={total_bce / n_batches:.4f} "
            f"cont={total_cont / n_batches:.4f} orth={total_orth / n_batches:.4f}) | "
            f"cw={cw:.3f} | "
            f"Val: acc={100 * metrics['accuracy']:.1f}% recall={100 * metrics['recall']:.0f}% "
            f"P@10={p10:.2f} F1={metrics['best_f1']:.3f} | "
            f"LR=[{','.join(f'{lr:.2e}' for lr in current_lrs[:2])}] "
            f"({train_time:.1f}s)"
        )
        metrics_history.append({**metrics, 'epoch': epoch, 'loss': total_loss / n_batches})

        if p10 > best_p10:
            best_p10 = p10
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'metrics': metrics,
                'phase': phase_name,
            }
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(ckpt, CHECKPOINT_PATH)
            logger.info(f"  *** New best P@10: {p10:.3f} ***")

    return best_p10, epoch_offset + epochs, metrics_history


def train(args):
    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING — Satellite Collision Avoidance")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    # Data
    n_samples = args.samples if args.samples > 0 else 0
    (train_loader, val_loader, test_loader), pos_weight = load_data(
        args.data, n_samples, args.augment, args.aug_copies, args.aug_synth)
    pos_weight = min(pos_weight, BCE_POS_WEIGHT_CAP)
    logger.info(f"BCE pos_weight: {pos_weight:.2f}")

    # Model
    model = CDMRiskModel().to(DEVICE)
    head_params = count_params(model.risk_head)
    logger.info(f"Model: {count_params(model):,} params "
                 f"(risk_head: {head_params}, backbone: {count_params(model) - head_params:,})")

    # Losses
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    cont_loss = ConjunctionContrastiveLoss()

    # ── Phase 1: Train risk_head only ──
    logger.info("\n" + "-" * 60)
    logger.info("Phase 1: Head warmup (backbone frozen)")
    logger.info("-" * 60)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.risk_head.parameters():
        p.requires_grad = True

    opt1 = torch.optim.AdamW(model.risk_head.parameters(), lr=PHASE1_LR, weight_decay=0.0)
    sched1 = WarmupReduceLR(opt1, warmup_epochs=1, batches_per_epoch=len(train_loader))
    for g in opt1.param_groups:
        g['base_lr'] = PHASE1_LR

    p10_1, epoch_count, _ = train_phase(
        model, train_loader, val_loader, opt1, sched1,
        bce_loss, cont_loss, 'phase1', PHASE1_EPOCHS, 0,
        target_contrastive_weight=0.0, orth_weight=0.0,
    )
    logger.info(f"Phase 1 complete. Best P@10: {p10_1:.3f}")

    # ── Phase 2: Full fine-tune ──
    logger.info("\n" + "-" * 60)
    logger.info("Phase 2: Full fine-tune (backbone unfrozen)")
    logger.info("-" * 60)
    for p in model.parameters():
        p.requires_grad = True

    opt2 = torch.optim.AdamW([
        {'params': model.risk_head.parameters(), 'lr': PHASE2_HEAD_LR,
         'weight_decay': PHASE2_WEIGHT_DECAY, 'base_lr': PHASE2_HEAD_LR},
        {'params': [p for n, p in model.named_parameters() if 'risk_head' not in n],
         'lr': PHASE2_BACKBONE_LR, 'weight_decay': PHASE2_WEIGHT_DECAY,
         'base_lr': PHASE2_BACKBONE_LR},
    ])
    sched2 = WarmupReduceLR(opt2, warmup_epochs=PHASE2_WARMUP_EPOCHS,
                             batches_per_epoch=len(train_loader))

    p10_2, _, _ = train_phase(
        model, train_loader, val_loader, opt2, sched2,
        bce_loss, cont_loss, 'phase2', args.epochs, PHASE1_EPOCHS,
        target_contrastive_weight=CONTRASTIVE_WEIGHT,
        orth_weight=ORTHOGONALITY_WEIGHT,
        contrastive_ramp_epochs=CONTRASTIVE_RAMP_EPOCHS,
    )
    logger.info(f"Phase 2 complete. Best P@10: {p10_2:.3f}")

    # Load best checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"\nLoaded best checkpoint from epoch {ckpt['epoch']}")

    # Final test evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Test Evaluation")
    logger.info("=" * 60)
    test_metrics = evaluate(model, test_loader, pos_weight)
    logger.info(f"  Test P@10:        {test_metrics['p10']:.3f}")
    logger.info(f"  Test P@100:       {test_metrics['p100']:.3f}")
    logger.info(f"  Test Accuracy:    {100 * test_metrics['accuracy']:.1f}%")
    logger.info(f"  Test Recall @0.5: {100 * test_metrics['recall']:.0f}%")
    logger.info(f"  Best F1:          {test_metrics['best_f1']:.3f} "
                 f"(at threshold {test_metrics['best_f1_threshold']:.2f})")
    logger.info(f"  Best F1 Prec:     {test_metrics['best_f1_precision']:.3f}")
    logger.info(f"  Best F1 Recall:   {test_metrics['best_f1_recall']:.3f}")

    return model


# ── CLI ─────────────────────────────────────────────────────────────────────

def evaluate_model(args):
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"No checkpoint found at {CHECKPOINT_PATH}. Run --mode train first.")
        return

    logger.info("Loading best model...")
    model = CDMRiskModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    logger.info(f"Loaded from epoch {ckpt.get('epoch', '?')} ({count_params(model):,} params)")

    logger.info("Loading validation data...")
    (_, val_loader, _), pos_weight = load_data(args.data)

    logger.info("\nPrecision / Recall:")
    results = precision_recall_curve(model, val_loader)
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    logger.info("\nEstimated maneuver savings (threshold = 0.50):")
    savings = compute_maneuver_savings(model, val_loader)
    for k, v in savings.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.1f}")
        else:
            logger.info(f"  {k}: {v}")


def demo(args):
    model = CDMRiskModel().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        logger.info("Loaded trained model.")
    else:
        logger.info("No checkpoint found. Using untrained model.")
    model.eval()

    from algorithm import (ConstellationDecisionEngine, AlgorithmConfig,
                           AlertThresholds, generate_decision_report)

    logger.info("=" * 60)
    logger.info("FULL PIPELINE: AI Model + Decision Algorithm")
    logger.info("=" * 60)

    # Step 1: Generate 300-sat constellation pass
    n_sats = 300
    logger.info(f"\n1. Simulating {n_sats}-satellite constellation pass...")
    conjunctions = []
    np.random.seed(42)
    for sid in range(n_sats):
        n_cdms = np.random.randint(0, 4)
        for cid in range(n_cdms):
            is_risk = np.random.random() < RISK_RATIO
            seq, label = generate_cdm_sequence(real_risk=is_risk, seed=sid * 100 + cid)
            with torch.no_grad():
                _, risk_logit, _ = model(seq.unsqueeze(0).to(DEVICE),
                                          seq[:, 0].unsqueeze(0).to(DEVICE))
                prob = torch.sigmoid(risk_logit).item()
            conjunctions.append({
                'satellite_id': f'STARLINK-{30000 + sid}',
                'conjunction_id': f'CDM-{sid}-{cid}',
                'risk_score': prob,
                'time_to_tca_hours': float(seq[-1, 0].item()),
                'miss_distance_km': float(seq[-1, 1].item()),
                'probability_of_collision': float(seq[-1, 2].item()),
                'true_label': int(label),
            })
    logger.info(f"   Total CDMs: {len(conjunctions)}")
    logger.info(f"   Real risks: {sum(c['true_label'] for c in conjunctions)}")

    # Step 2: Satellite states with random fuel
    from algorithm import SatelliteState
    satellites = {}
    for c in conjunctions:
        sid = c['satellite_id']
        if sid not in satellites:
            fuel = np.random.uniform(0.5, 5.0)
            satellites[sid] = SatelliteState(sid, fuel, 0.15, np.random.uniform(0.6, 1.0),
                                              maneuvers_remaining=int(fuel / 0.15))

    # Step 3: Decision algorithm
    logger.info("\n2. Running decision algorithm...")
    cfg = AlgorithmConfig(
        thresholds=AlertThresholds(
            amber=ALGORITHM_AMBER_THRESHOLD,
            red=ALGORITHM_RED_THRESHOLD,
            critical=ALGORITHM_CRITICAL_THRESHOLD,
        ),
        temporal_window=ALGORITHM_TEMPORAL_WINDOW,
        temporal_min_positive=ALGORITHM_TEMPORAL_MIN_POSITIVE,
        fuel_reserve_fraction=ALGORITHM_FUEL_RESERVE_FRACTION,
        min_lead_time_hours=ALGORITHM_MIN_LEAD_HOURS,
        max_lead_time_hours=ALGORITHM_MAX_LEAD_HOURS,
    )
    engine = ConstellationDecisionEngine(cfg)
    decisions = engine.batch_process(satellites, conjunctions)
    report = generate_decision_report(decisions)

    logger.info(f"\n3. Decision Report:")
    for k, v in report.items():
        logger.info(f"   {k}: {v}")

    # Step 4: Top-5 urgent
    logger.info(f"\n4. Top-5 Most Urgent:")
    logger.info(f"   {'Satellite':<20s} {'Risk':<8s} {'Level':<10s} {'TCA(h)':<8s} {'Maneuver':<10s}")
    logger.info(f"   {'-'*66}")
    for d in decisions[:5]:
        mv = "YES" if d.maneuver else "no"
        logger.info(f"   {d.satellite_id:<20s} {d.risk_score:<8.3f} {d.alert_level:<10s} "
                     f"{d.time_to_tca_hours:<8.1f} {mv:<10s}")

    # Step 5: Performance
    conj_by_id = {c['conjunction_id']: c for c in conjunctions}
    decisions_by_risk = sorted(decisions, key=lambda d: d.risk_score, reverse=True)
    p10 = sum(1 for d in decisions_by_risk[:10] if conj_by_id.get(d.conjunction_id, {}).get('true_label'))
    p10 /= 10.0
    maneuvers_ordered = sum(1 for d in decisions if d.maneuver)
    risks_total = sum(1 for c in conjunctions if c['true_label'])
    risks_caught = sum(1 for d in decisions if d.maneuver and conj_by_id.get(d.conjunction_id, {}).get('true_label'))
    savings = 100 * (1 - maneuvers_ordered / max(len(conjunctions), 1))

    logger.info(f"\n5. Performance vs Baseline:")
    logger.info(f"   P@10: {p10:.2f}")
    logger.info(f"   Risks detected: {risks_caught}/{risks_total} ({100 * risks_caught / max(risks_total, 1):.0f}%)")
    logger.info(f"   Maneuver savings: {savings:.0f}% ({len(conjunctions)} → {maneuvers_ordered})")
    logger.info(f"   Algorithm latency: {engine.metrics['avg_latency_ms']:.2f}ms per conjunction")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Satellite CDM Risk Model — Production Pipeline")
    parser.add_argument('--mode', choices=['train', 'eval', 'demo'], default='demo',
                        help='train | eval | demo')
    parser.add_argument('--data', choices=['synthetic', 'esa'], default='synthetic',
                        help='Training data source')
    parser.add_argument('--epochs', type=int, default=PHASE2_EPOCHS,
                        help=f'Number of Phase-2 epochs (default: {PHASE2_EPOCHS})')
    parser.add_argument('--samples', type=int, default=0,
                        help='Max training samples (0 = all available)')
    parser.add_argument('--lr', type=float, default=0.0,
                        help='Override backbone LR (0 = use config default)')
    parser.add_argument('--augment', type=str, default='',
                        help='Data augmentation: mixup, jitter, or mixup+jitter (comma-separated)')
    parser.add_argument('--aug-copies', type=int, default=5,
                        help='Number of jitter copies per real event (default: 5)')
    parser.add_argument('--aug-synth', type=int, default=10000,
                        help='Number of synthetic mixup events (default: 10000)')
    args = parser.parse_args()

    if args.lr > 0:
        # Override learning rates
        import config as cfg_mod
        cfg_mod.PHASE2_BACKBONE_LR = args.lr

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate_model(args)
    else:
        demo(args)
