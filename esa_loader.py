import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import NUM_CDM_FEATURES, MAX_CDMS_PER_EVENT

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(_DATA_DIR, "esa_kelvins_full.csv")

CDM_FEATURES = [
    'time_to_tca',
    'miss_distance',
    'relative_speed',
    'relative_position_r', 'relative_position_t', 'relative_position_n',
    'relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n',
    't_rcs_estimate', 'c_rcs_estimate',
    't_cd_area_over_mass', 'c_cd_area_over_mass',
    't_j2k_sma', 't_j2k_ecc', 't_j2k_inc',
    'c_j2k_sma', 'c_j2k_ecc', 'c_j2k_inc',
    't_sigma_r', 't_sigma_t', 't_sigma_n',
    'c_sigma_r', 'c_sigma_t', 'c_sigma_n',
    't_sigma_rdot', 't_sigma_tdot', 't_sigma_ndot',
    'c_sigma_rdot', 'c_sigma_tdot', 'c_sigma_ndot',
    'geocentric_latitude', 'azimuth', 'elevation',
    'mahalanobis_distance',
    't_position_covariance_det', 'c_position_covariance_det',
    'F10', 'F3M', 'SSN', 'AP',
    't_h_apo', 't_h_per',
    'c_h_apo', 'c_h_per',
    't_weighted_rms', 'c_weighted_rms',
    't_sedr', 'c_sedr',
]

OBJ_TYPE_CATEGORIES = ['DEBRIS', 'PAYLOAD', 'ROCKET BODY', 'UNKNOWN']

RISK_THRESHOLD = -7.0

# Populated by load_esa_kelvins for normalization
_NORM_MEAN = None
_NORM_STD = None


def load_esa_kelvins(path=DATA_PATH, max_events=None):
    global _NORM_MEAN, _NORM_STD
    print(f"Loading ESA Kelvins data from {path}...")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df)}, columns: {len(df.columns)}")

    event_id_col = 'event_id'

    available_features = [c for c in CDM_FEATURES if c in df.columns]
    print(f"  Available features: {len(available_features)}/{len(CDM_FEATURES)}")

    obj_dummies = pd.get_dummies(df['c_object_type'], prefix='c_object_type').astype(float)
    obj_dummies = obj_dummies.reindex(columns=[f'c_object_type_{c}' for c in OBJ_TYPE_CATEGORIES], fill_value=0.0)
    df = pd.concat([df, obj_dummies], axis=1)
    available_features.extend(c for c in obj_dummies.columns if c not in available_features)

    # Impute NaN with median per feature
    feat_df = df[available_features]
    nan_count = feat_df.isna().sum().sum()
    if nan_count > 0:
        print(f"  Imputing {nan_count} NaN values with feature medians...")
        medians = feat_df.median(numeric_only=True)
        feat_df = feat_df.fillna(medians)
        df[available_features] = feat_df

    # Compute normalization stats (z-score: mean/std per feature)
    _NORM_MEAN = feat_df.mean(numeric_only=True).to_dict()
    _NORM_STD = feat_df.std(numeric_only=True).to_dict()
    # Replace zero std with 1 to avoid division by zero
    for k in _NORM_STD:
        if _NORM_STD[k] == 0.0:
            _NORM_STD[k] = 1.0

    # Group by event
    events = {}
    for _, row in df.iterrows():
        eid = row[event_id_col]
        if eid not in events:
            events[eid] = {'rows': []}
        events[eid]['rows'].append(row)

    # Sort rows by time_to_tca and take LAST risk as final
    for eid in events:
        events[eid]['rows'].sort(key=lambda r: r['time_to_tca'])
        final_risk = events[eid]['rows'][-1]['risk']
        events[eid]['final_risk'] = final_risk

    print(f"  Events: {len(events)}")
    high_risk = sum(1 for e in events.values() if e.get('final_risk', -30) > RISK_THRESHOLD)
    print(f"  High-risk events (risk > 1e-7): {high_risk}/{len(events)} ({100*high_risk/len(events):.1f}%)")

    if max_events:
        keys = list(events.keys())[:max_events]
        events = {k: events[k] for k in keys}
        print(f"  Using {max_events} events")

    return events, available_features


def normalize_feat(val: float, feat: str) -> float:
    if _NORM_MEAN is None or _NORM_STD is None:
        return val
    mu = _NORM_MEAN.get(feat, 0.0)
    sigma = _NORM_STD.get(feat, 1.0)
    return (val - mu) / sigma


def events_to_tensors(events, features):
    sequences = []
    targets = []

    for eid, event in events.items():
        rows = sorted(event['rows'], key=lambda r: r['time_to_tca'])

        if len(rows) < 2:
            continue

        n_feats = min(len(features), NUM_CDM_FEATURES)
        seq = np.zeros((len(rows), n_feats))
        for i, row in enumerate(rows):
            for j, feat in enumerate(features):
                if j < n_feats:
                    seq[i, j] = normalize_feat(row[feat], feat)

        sequences.append(torch.from_numpy(seq).float())

        final_risk = event.get('final_risk', -30.0)
        target = 1.0 if final_risk > RISK_THRESHOLD else 0.0
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
