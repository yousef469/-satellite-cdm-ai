"""
ESA Kelvins Collision Avoidance Challenge data loader.

Dataset: https://kelvins.esa.int/collision-avoidance-challenge/data/
Format: CSV with ~160k rows, each row is one CDM for an event.
Events are identified by event_id. Each event has ~12 CDMs over time.
Target: binary (maneuver required) or probability of collision.

To use: download train_data.zip from ESA Kelvins, extract CSV, set DATA_PATH.
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import NUM_CDM_FEATURES, MAX_CDMS_PER_EVENT


DATA_PATH = 'data/esa_kelvins_train.csv'

CDM_FEATURES = [
    'time_to_tca',
    'miss_distance',
    'collision_probability',
    'cov_rr', 'cov_tt', 'cov_nn',
    'cov_rt', 'cov_rn', 'cov_tn',
    'cov_vr', 'cov_vt', 'cov_vn',
    'relative_speed',
    'object_1_rcs',
    'object_1_mass',
    'is_debris', 'is_payload', 'is_rocket_body',
    'f107', 'ap',
    'altitude', 'inclination',
    'hard_body_radius',
    'object_1_sigma_r', 'object_1_sigma_t', 'object_1_sigma_n',
    'object_2_sigma_r', 'object_2_sigma_t', 'object_2_sigma_n',
]

TARGET_COL = 'final_collision_probability'


def load_esa_kelvins(path=DATA_PATH, max_events=None):
    print(f"Loading ESA Kelvins data from {path}...")
    df = pd.read_csv(path)

    print(f"  Raw rows: {len(df)}, columns: {list(df.columns[:20])}")

    events = {}
    event_id_col = 'event_id' if 'event_id' in df.columns else None
    if event_id_col is None:
        for col in df.columns:
            if 'event' in col.lower() or 'id' in col.lower():
                event_id_col = col
                break
    if event_id_col is None:
        raise ValueError("No event_id column found")

    for _, row in df.iterrows():
        eid = row[event_id_col]
        if eid not in events:
            events[eid] = {'rows': [], 'final_pc': None}
        events[eid]['rows'].append(row)

    if TARGET_COL in df.columns:
        final_pcs = df.groupby(event_id_col)[TARGET_COL].last()
        for eid in events:
            if eid in final_pcs.index:
                events[eid]['final_pc'] = final_pcs[eid]

    available_features = [c for c in CDM_FEATURES if c in df.columns]
    print(f"  Available features: {len(available_features)}/{len(CDM_FEATURES)}")
    print(f"  Events: {len(events)}")
    print(f"  Has target: {TARGET_COL in df.columns}")

    if max_events:
        keys = list(events.keys())[:max_events]
        events = {k: events[k] for k in keys}
        print(f"  Using {max_events} events")

    return events, available_features


def events_to_tensors(events, features):
    sequences = []
    targets = []

    for eid, event in events.items():
        rows = sorted(event['rows'], key=lambda r: r['time_to_tca'] if 'time_to_tca' in r else 0)

        if len(rows) < 2:
            continue

        seq = np.zeros((len(rows), NUM_CDM_FEATURES))
        for i, row in enumerate(rows):
            for j, feat in enumerate(features):
                if j < NUM_CDM_FEATURES:
                    seq[i, j] = row[feat]

        sequences.append(torch.from_numpy(seq).float())

        if event['final_pc'] is not None:
            target = 1.0 if float(event['final_pc']) > 1e-4 else 0.0
        else:
            target = 0.0
        targets.append(target)

    return sequences, torch.tensor(targets, dtype=torch.float32)


class ESADataset(Dataset):
    def __init__(self, path=DATA_PATH, max_events=None):
        events, features = load_esa_kelvins(path, max_events)
        self.sequences, self.labels = events_to_tensors(events, features)
        print(f"  Loaded {len(self.sequences)} valid sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
