#!/usr/bin/env python3
"""
Satellite CDM Collision Avoidance Risk Model
PDM + DepthwiseTCN hybrid architecture for conjunction risk prediction.

Usage:
  python run.py --mode train           # Train on synthetic data
  python run.py --mode eval            # Evaluate best checkpoint
  python run.py --mode demo            # Run a single inference demo
"""
import argparse
import torch
import sys
import os
import numpy as np
from torch.utils.data import DataLoader

from config import (
    NUM_SYNTHETIC_EVENTS, RISK_RATIO, BATCH_SIZE, EPOCHS,
    CHECKPOINT_PATH,
)
from cdm_generator import generate_dataset, generate_cdm_sequence
from model import CDMRiskModel, count_params
from train import CDMDataset, pad_collate_fn
from evaluate import precision_recall_curve, compute_maneuver_savings


def train(args):
    print("=" * 60)
    print("CDM Risk Model Training")
    print("=" * 60)

    sequences, labels = generate_dataset(NUM_SYNTHETIC_EVENTS, RISK_RATIO)
    n_train = int(0.8 * len(sequences))

    train_ds = CDMDataset(sequences[:n_train], labels[:n_train])
    val_ds = CDMDataset(sequences[n_train:], labels[n_train:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    model = CDMRiskModel()
    print(f"Model: {count_params(model):,} params")

    pos_count = labels[:n_train].sum().item()
    neg_count = n_train - pos_count
    pos_weight = min(neg_count / max(pos_count, 1), 3.0)
    print(f"pos_weight = {pos_weight:.2f}")
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    from model import ConjunctionContrastiveLoss
    cont = ConjunctionContrastiveLoss()
    opt = torch.optim.AdamW([
        {'params': model.risk_head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
        {'params': [p for n, p in model.named_parameters() if 'risk_head' not in n], 'lr': 1e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_p10 = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        for cdm, lbl, mask in train_loader:
            proto, risk, d = model(cdm, cdm[:,:,0], mask)
            loss = bce(risk, lbl) + 0.05 * cont(d, lbl) + 0.01 * model.get_aux_losses()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            all_risk, all_lbl = [], []
            for cdm, lbl, mask in val_loader:
                _, risk, _ = model(cdm, cdm[:,:,0], mask)
                all_risk.append(torch.sigmoid(risk))
                all_lbl.append(lbl)
            probs = torch.cat(all_risk)
            labels = torch.cat(all_lbl)
            sv, si = torch.sort(probs, descending=True)
            p10 = labels[si][:10].float().mean().item()
            acc = ((probs > 0.5) == labels).float().mean().item()
            tp = ((probs > 0.5) & (labels == 1)).sum().item()
            fn = ((probs <= 0.5) & (labels == 1)).sum().item()
            recall = tp / max(tp + fn, 1)
        print(f"  Val: acc={100*acc:.1f}% recall={100*recall:.0f}% P@10={p10:.2f}")

        if p10 > best_p10:
            best_p10 = p10
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  New best P@10: {p10:.2f}")

    print(f"\nDone. Best P@10: {best_p10:.2f}")
    return model


def evaluate(args):
    print("Loading best model...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Run --mode train first.")
        return

    model = CDMRiskModel()
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded model ({count_params(model):,} params)")

    print("Generating validation data...")
    sequences, labels = generate_dataset(2000, RISK_RATIO)
    n_val = int(0.2 * len(sequences))
    val_ds = CDMDataset(sequences[-n_val:], labels[-n_val:])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    print(f"Val samples: {n_val} ({labels[-n_val:].sum().item():.0f} positive)")

    print("\nPrecision / Recall:")
    results = precision_recall_curve(model, val_loader)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    print("\nEstimated maneuver savings (threshold = 0.50):")
    savings = compute_maneuver_savings(model, val_loader)
    for k, v in savings.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")


def demo(args):
    model = CDMRiskModel()
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("Loaded trained model.")
    else:
        print("No checkpoint found. Using untrained model.")

    model.eval()

    print("\n" + "=" * 60)
    print("FULL PIPELINE: AI Model + Decision Algorithm")
    print("=" * 60)

    # ── Step 1: Generate events for a simulated constellation ──
    n_sats = 300
    print(f"\n1. Simulating {n_sats}-satellite constellation pass...")
    conjunctions = []
    np.random.seed(42)
    for sid in range(n_sats):
        # Each satellite has 0-3 CDMs in this batch
        n_cdms = np.random.randint(0, 4)
        for cid in range(n_cdms):
            is_risk = np.random.random() < 0.04  # ~4% real risk
            seq, label = generate_cdm_sequence(real_risk=is_risk, seed=sid * 100 + cid)
            with torch.no_grad():
                _, risk_logit, _ = model(seq.unsqueeze(0), seq[:, 0].unsqueeze(0))
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
    print(f"   Total CDMs: {len(conjunctions)}")
    print(f"   Real risks: {sum(c['true_label'] for c in conjunctions)}")

    # ── Step 2: Initialize satellite states ──
    from algorithm import (SatelliteState, ConstellationDecisionEngine,
                           AlertThresholds, generate_decision_report)
    satellites = {}
    for c in conjunctions:
        sid = c['satellite_id']
        if sid not in satellites:
            fuel = np.random.uniform(0.5, 5.0)  # kg
            satellites[sid] = SatelliteState(
                satellite_id=sid,
                fuel_remaining_kg=fuel,
                fuel_per_maneuver_kg=0.15,
                battery_soc=np.random.uniform(0.6, 1.0),
                maneuvers_remaining=int(fuel / 0.15),
            )

    # ── Step 3: Run the decision algorithm ──
    print("\n2. Running constellation-scale decision algorithm...")
    engine = ConstellationDecisionEngine(
        thresholds=AlertThresholds(amber=0.20, red=0.50, critical=0.85),
    )
    decisions = engine.batch_process(satellites, conjunctions)
    report = generate_decision_report(decisions)

    print(f"\n3. Decision Report:")
    print(f"   Total conjunctions:  {report['total_conjunctions']}")
    print(f"   Maneuvers ordered:   {report['maneuvers_recommended']}")
    print(f"   Critical alerts:     {report['critical_alerts']}")
    print(f"   Red alerts:          {report['red_alerts']}")
    print(f"   Amber alerts:        {report['amber_alerts']}")
    print(f"   Safe (green):        {report['green']}")
    print(f"   Satellites affected: {report['satellites_affected']}")

    # ── Step 4: Show top-5 most urgent decisions ──
    print(f"\n4. Top-5 Most Urgent Decisions:")
    print(f"   {'Satellite':<20s} {'Risk':<8s} {'Level':<10s} {'TCA(h)':<8s} {'Miss(km)':<10s} {'Maneuver':<10s}")
    print(f"   {'-'*66}")
    for d in decisions[:5]:
        mv = "YES" if d.maneuver else "no"
        print(f"   {d.satellite_id:<20s} {d.risk_score:<8.3f} {d.alert_level:<10s} {d.time_to_tca_hours:<8.1f} {d.miss_distance_km:<10.3f} {mv:<10s}")

    # ── Step 5: Accuracy vs baseline ──
    conj_by_id = {c['conjunction_id']: c for c in conjunctions}
    decisions_by_risk = sorted(decisions, key=lambda d: d.risk_score, reverse=True)
    p10 = sum(1 for d in decisions_by_risk[:10] if conj_by_id[d.conjunction_id]['true_label'])
    p10 /= 10.0
    maneuvers_ordered = sum(1 for d in decisions if d.maneuver)
    risks_total = sum(1 for c in conjunctions if c['true_label'])
    risks_caught = sum(1 for d in decisions if d.maneuver and conj_by_id[d.conjunction_id]['true_label'])
    print(f"\n5. Performance:")
    print(f"   P@10: {p10:.2f}")
    print(f"   Risks detected: {risks_caught}/{risks_total} ({100*risks_caught/max(risks_total,1):.0f}%)")
    baseline = len(conjunctions)
    savings = 100 * (1 - maneuvers_ordered / max(baseline, 1))
    print(f"   Maneuver savings: {savings:.0f}% ({baseline} → {maneuvers_ordered})")
    print(f"   Model: {count_params(model):,} params")
    print(f"   Algorithm: rules-based, no training required")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'demo'], default='demo')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    else:
        demo(args)
