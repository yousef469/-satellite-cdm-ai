"""
CDM sequence augmentation for data-limited satellite collision avoidance.

Strategy — MixUp on CDM sequences:
  1. Sample two real events from the same risk class (high/low)
  2. Linearly interpolate CDM sequences + final risk label
  3. Produces new events with novel feature combinations
  This preserves temporal structure while generating diverse training data.

Strategy — Feature jittering:
  1. Take a real event, add Gaussian noise to each CDM feature
  2. Noise std = noise_scale * feature_std (per feature)
  3. Keeps final risk label unchanged
  Generates density around real data points.

Both run in minutes, output in the same format as load_esa_kelvins.
"""
import numpy as np
import random
from collections import defaultdict


def augment_mixup(events, features, num_synthetic, alpha=0.4, risk_threshold=-7.0):
    """
    Generate synthetic events via MixUp between same-class event pairs.

    Args:
        events: dict from load_esa_kelvins()
        features: list of feature names
        num_synthetic: how many synthetic events to create
        alpha: Beta distribution parameter for mixup coefficient
               0.4 → strong mixing (weights ~0.3-0.7), 2.0 → weak mixing

    Returns:
        events dict with original + synthetic events
    """
    event_list = list(events.items())
    high_risk = [(eid, ev) for eid, ev in event_list
                 if ev.get('final_risk', -30) > risk_threshold]
    low_risk = [(eid, ev) for eid, ev in event_list
                if ev.get('final_risk', -30) <= risk_threshold]

    if len(high_risk) < 2:
        print(f"  Warning: only {len(high_risk)} high-risk events, MixUp may be limited")

    synthetic = {}
    next_id = max(int(k) if isinstance(k, (int, str)) and str(k).isdigit() else 0 for k in events) + 1

    for i in range(num_synthetic):
        lam = np.random.beta(alpha, alpha)
        pool = random.choice([high_risk, low_risk])
        if len(pool) < 2:
            pool = event_list
        e1 = random.choice(pool)
        e2 = random.choice(pool)
        while e2[0] == e1[0] and len(pool) > 1:
            e2 = random.choice(pool)

        rows1 = sorted(e1[1]['rows'], key=lambda r: r['time_to_tca'])
        rows2 = sorted(e2[1]['rows'], key=lambda r: r['time_to_tca'])

        n = min(len(rows1), len(rows2))
        new_rows = []
        for j in range(n):
            new_row = {}
            for feat in features:
                v1 = rows1[j][feat]
                v2 = rows2[j][feat]
                new_row[feat] = lam * v1 + (1 - lam) * v2
            # Copy non-feature columns from first parent
            for col in rows1[j].index if hasattr(rows1[0], 'index') else rows1[0].keys():
                if col not in features:
                    new_row[col] = rows1[j][col]
            new_rows.append(new_row)

        new_final = lam * e1[1]['final_risk'] + (1 - lam) * e2[1]['final_risk']

        synthetic[next_id] = {'rows': new_rows, 'final_risk': new_final}
        next_id += 1

    events.update(synthetic)
    n_new = len(synthetic)
    n_high = sum(1 for e in events.values()
                 if e.get('final_risk', -30) > risk_threshold)
    print(f"  MixUp: generated {n_new} events (total high-risk: {n_high})")
    return events


def augment_jitter(events, features, num_copies=3, noise_scale=0.15):
    """
    Generate synthetic events by adding Gaussian noise to CDM features.

    Noise std = noise_scale * feature_std (computed from real data).
    Final risk label is preserved.

    Args:
        events: dict from load_esa_kelvins()
        features: list of feature names
        num_copies: number of jittered copies per real event
        noise_scale: fraction of feature std to use as noise magnitude
    """
    from esa_loader import _NORM_STD

    event_list = list(events.items())
    synthetic = {}
    next_id = max(int(k) if isinstance(k, (int, str)) and str(k).isdigit() else 0 for k in events) + 1

    for eid, event in event_list:
        rows = event['rows']
        for c in range(num_copies):
            new_rows = []
            for row in rows:
                new_row = dict(row)
                for feat in features:
                    if feat in row:
                        sigma = _NORM_STD.get(feat, 1.0)
                        noise = np.random.normal(0, noise_scale * sigma)
                        new_row[feat] = row[feat] + noise
                new_rows.append(new_row)

            synthetic[next_id] = {
                'rows': new_rows,
                'final_risk': event['final_risk'],
            }
            next_id += 1

    orig_count = len(events)
    events.update(synthetic)
    n_new = len(synthetic)
    risk_threshold = -7.0
    n_high = sum(1 for e in events.values()
                 if e.get('final_risk', -30) > risk_threshold)
    print(f"  Jitter: generated {n_new} events from {orig_count} originals "
          f"(total high-risk: {n_high})")
    return events
