import torch
import numpy as np
import random
from config import NUM_CDM_FEATURES, MAX_CDMS_PER_EVENT


CDM_FEATURE_NAMES = [
    'time_to_tca',
    'miss_distance_km',
    'probability_of_collision',
    'cov_xx', 'cov_yy', 'cov_zz',
    'cov_xy', 'cov_xz', 'cov_yz',
    'cov_vx', 'cov_vy', 'cov_vz',
    'relative_speed_kms',
    'object_1_rcs',
    'object_1_mass_est',
    'object_1_is_debris',
    'object_1_is_payload',
    'object_1_is_rocket_body',
    'solar_flux_f107',
    'geomagnetic_ap',
    'altitude_km',
    'inclination_deg',
    'hard_body_radius_m',
    'uncertainty_calibration',
    'cov_trace_growth_rate',
    'miss_distance_trend',
    'pc_trend',
    'time_since_last_cdm_hours',
    'conjunction_arc_length_deg',
    'encounter_type_radial',
    'encounter_type_along',
    'encounter_type_cross',
    'object_2_is_maneuverable',
    'object_2_tle_age_hours',
    'drag_regime_stable',
    'drag_regime_chaotic',
    'solar_activity_quiet',
    'solar_activity_storm',
    'collision_risk_inferred_calibrated',
    'relative_velocity_angle_deg',
    'object_1_catalog_age_days',
    'num_previous_conjunctions',
]


def generate_cdm_sequence(real_risk=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Random number of CDMs (6-15)
    num_cdms = np.random.randint(6, MAX_CDMS_PER_EVENT + 1)
    seq = np.zeros((num_cdms, NUM_CDM_FEATURES))

    # Time to TCA: starts at 72-168h, decreases to 0-2h
    tca_max = np.random.uniform(60, 168)
    time_to_tca = np.linspace(tca_max, np.random.uniform(0, 2), num_cdms)
    seq[:, 0] = time_to_tca

    # Solar activity background
    f107 = np.random.normal(100, 30)
    ap = np.random.exponential(10)
    is_storm = np.random.random() < 0.05
    if is_storm:
        f107 += np.random.uniform(50, 150)
        ap += np.random.uniform(20, 80)
    seq[:, 18] = f107
    seq[:, 19] = ap
    seq[:, 35] = 1.0 if not is_storm else 0.0
    seq[:, 36] = 0.0 if not is_storm else 1.0

    # Altitude (LEO: 300-1200 km)
    altitude = np.random.uniform(300, 1200)
    seq[:, 20] = altitude

    # Inclination
    inclination = np.random.uniform(0, 98)
    seq[:, 21] = inclination

    # Object type
    obj_is_debris = np.random.random() < 0.6
    obj_is_payload = (not obj_is_debris) and (np.random.random() < 0.6)
    seq[:, 15] = 1.0 if obj_is_debris else 0.0
    seq[:, 16] = 1.0 if obj_is_payload else 0.0
    seq[:, 17] = 1.0 if not (obj_is_debris or obj_is_payload) else 0.0

    # Object 2 is maneuverable?
    obj2_maneuverable = np.random.random() < 0.15
    seq[:, 31] = 1.0 if obj2_maneuverable else 0.0

    # RCS and mass
    if obj_is_debris:
        rcs = np.random.exponential(0.5)
        mass = np.random.exponential(50)
    elif obj_is_payload:
        rcs = np.random.lognormal(0, 1)
        mass = np.random.lognormal(5, 1)
    else:
        rcs = np.random.exponential(2)
        mass = np.random.exponential(200)
    seq[:, 13] = min(rcs, 50)
    seq[:, 14] = min(mass, 5000)

    # Relative speed (LEO typical: 1-15 km/s)
    rel_speed = np.random.uniform(1, 15)
    seq[:, 12] = rel_speed

    # Covariance diagonal elements: grow as we go back in time from TCA
    # Base covariance at TCA
    base_cov_xx = np.random.lognormal(-2, 1)
    base_cov_yy = np.random.lognormal(-2, 1)
    base_cov_zz = np.random.lognormal(-2, 1)

    cov_growth_rate = np.random.uniform(0.5, 3.0)
    seq[:, 23] = cov_growth_rate

    for i in range(num_cdms):
        t = time_to_tca[i] / 24.0  # days
        growth = 1.0 + cov_growth_rate * t
        if is_storm:
            growth *= (1.0 + np.random.uniform(0.5, 2.0))
        seq[i, 3] = base_cov_xx * growth
        seq[i, 4] = base_cov_yy * growth
        seq[i, 5] = base_cov_zz * growth
        seq[i, 6] = np.random.normal(0, 0.01) * growth  # xy
        seq[i, 7] = np.random.normal(0, 0.01) * growth  # xz
        seq[i, 8] = np.random.normal(0, 0.01) * growth  # yz

    # Velocity covariance
    for i in range(num_cdms):
        t = time_to_tca[i] / 24.0
        growth = 1.0 + 0.3 * cov_growth_rate * t
        seq[i, 9] = np.random.lognormal(-4, 0.5) * growth
        seq[i, 10] = np.random.lognormal(-4, 0.5) * growth
        seq[i, 11] = np.random.lognormal(-4, 0.5) * growth

    # Uncertainty calibration factor (how reliable is the CDM)
    calibration = np.random.beta(5, 1)
    seq[:, 23] = calibration

    # ---------- RISK MODELING ----------
    # Miss distance: starts uncertain, converges
    if real_risk:
        # High-risk conjunctions
        miss_final = np.random.uniform(0.01, 0.5)
        miss_initial = miss_final + np.random.uniform(0.1, 5.0)
        miss_decay = np.random.uniform(0.3, 1.5)

        # Pc rises as TCA approaches
        pc_final = np.random.uniform(1e-4, 0.5)
        pc_initial = max(1e-8, pc_final * np.random.uniform(0.001, 0.1))

        # Hard body radius (small for debris, larger for sats)
        hbr = np.random.uniform(0.5, 5.0)
        seq[:, 22] = hbr

        # Encounter type favors radial for high risk
        seq[:, 28] = np.random.uniform(0.3, 0.8)
        seq[:, 29] = np.random.uniform(-0.3, 0.3)
        seq[:, 30] = np.sqrt(1.0 - seq[:, 28]**2 - seq[:, 29]**2)
    else:
        # Safe conjunctions
        miss_initial = np.random.uniform(1, 20)
        miss_final = miss_initial * np.random.uniform(0.5, 2.0)
        miss_decay = np.random.uniform(-0.5, 0.5)

        pc_final = 0.0
        pc_initial = 0.0

        hbr = np.random.uniform(0.5, 5.0)
        seq[:, 22] = hbr

        seq[:, 28] = np.random.uniform(-0.5, 0.5)
        seq[:, 29] = np.random.uniform(-0.5, 0.5)
        seq[:, 30] = np.sqrt(np.maximum(0, 1.0 - seq[:, 28]**2 - seq[:, 29]**2))

    # Generate miss distance and Pc sequences
    for i in range(num_cdms):
        t_norm = time_to_tca[i] / tca_max
        if real_risk:
            seq[i, 1] = miss_final + (miss_initial - miss_final) * np.exp(-miss_decay * time_to_tca[i] / 10.0)
            seq[i, 1] = max(0.001, seq[i, 1])
            seq[i, 2] = pc_final * (1.0 - np.exp(-3.0 * (1.0 - t_norm)))
            seq[i, 2] = max(1e-10, seq[i, 2])
        else:
            noise = np.random.normal(0, 0.5)
            seq[i, 1] = max(0.001, miss_final + np.random.normal(0, 1.0))
            seq[i, 2] = 0.0

        seq[i, 24] = seq[-1, 1] - seq[max(0, i-1), 1] if i > 0 else 0.0
        seq[i, 25] = seq[min(num_cdms-1, i+1), 2] - seq[i, 2] if i < num_cdms - 1 else 0.0

    # Time since last CDM (irregular intervals)
    intervals = np.random.exponential(8, num_cdms)
    intervals = np.clip(intervals, 0.5, 24)
    seq[:, 27] = intervals

    # Encounter type normalization
    enc_norm = np.sqrt(seq[:, 28]**2 + seq[:, 29]**2 + seq[:, 30]**2)
    seq[:, 28] /= (enc_norm + 1e-8)
    seq[:, 29] /= (enc_norm + 1e-8)
    seq[:, 30] /= (enc_norm + 1e-8)

    # Drag regime
    drag_stable = 1.0 if (altitude > 600 or is_storm) else (1.0 if np.random.random() > 0.5 else 0.0)
    seq[:, 34] = drag_stable
    seq[:, 35] = 1.0 - drag_stable

    # Object catalog age
    seq[:, 40] = np.random.uniform(1, 3650)

    # Number of previous conjunctions (correlated: debris at low altitude has more)
    seq[:, 41] = np.random.poisson(10 + max(0, 600 - altitude) / 20)

    # Calibrated risk (our target: 1 = real risk, 0 = safe)
    calibrated_risk = 1.0 if real_risk else 0.0

    return torch.from_numpy(seq).float(), calibrated_risk


def generate_dataset(num_events, risk_ratio=0.02):
    sequences = []
    labels = []
    num_risk = int(num_events * risk_ratio)

    print(f"Generating {num_events} events ({num_risk} high-risk, {num_events - num_risk} safe)...")

    for i in range(num_risk):
        seq, label = generate_cdm_sequence(real_risk=True, seed=i)
        sequences.append(seq)
        labels.append(label)
        if (i + 1) % 5000 == 0:
            print(f"  High-risk: {i+1}/{num_risk}")

    for i in range(num_events - num_risk):
        seq, label = generate_cdm_sequence(real_risk=False, seed=100000 + i)
        sequences.append(seq)
        labels.append(label)
        if (i + 1) % 5000 == 0:
            print(f"  Safe: {i+1}/{num_events - num_risk}")

    print(f"Done. {len(sequences)} sequences, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    # Shuffle to mix positive and negative
    indices = np.random.permutation(len(sequences))
    sequences = [sequences[i] for i in indices]
    labels = [labels[i] for i in indices]
    return sequences, torch.tensor(labels, dtype=torch.float32)


def pad_collate(batch):
    sequences = []
    labels = []
    max_len = max(s[0].shape[0] for s in batch)

    for seq, label in batch:
        t, f = seq.shape
        if t < max_len:
            pad = torch.zeros(max_len - t, f)
            padded = torch.cat([seq, pad], dim=0)
            mask = torch.cat([torch.ones(t), torch.zeros(max_len - t)])
        else:
            padded = seq[:max_len]
            mask = torch.ones(max_len)
        sequences.append(padded)
        labels.append(label)

    return torch.stack(sequences), torch.tensor(labels, dtype=torch.float32), torch.stack([m for m in [torch.ones(max_len)]])
