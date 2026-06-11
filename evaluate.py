import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CDMRiskModel
from train import CDMDataset, pad_collate_fn


@torch.no_grad()
def precision_recall_curve(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    for cdm_seq, labels, mask in loader:
        time_to_tca = cdm_seq[:, :, 0]
        _, risk_logit, _ = model(cdm_seq, time_to_tca, mask)
        all_probs.append(torch.sigmoid(risk_logit).cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    sorted_idx = torch.argsort(probs, descending=True)
    sorted_labels = labels[sorted_idx]
    n_pos = sorted_labels.sum().int()

    results = {}
    for k in [1, 5, 10, 50, 100, 500]:
        k = min(k, len(sorted_labels))
        if k > 0:
            tp = sorted_labels[:k].sum().item()
            results[f'P@{k}'] = tp / k
            results[f'R@{k}'] = tp / max(n_pos, 1)

    for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
        preds = (probs > th).float()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        results[f'th={th:.1f}_TPR'] = tpr
        results[f'th={th:.1f}_FPR'] = fpr

    sorted_labels_np = sorted_labels.numpy()
    cumsum = np.cumsum(sorted_labels_np)
    total_pos = cumsum[-1] if len(cumsum) > 0 else 1

    for frac in [0.01, 0.05, 0.10]:
        idx = int(len(sorted_labels_np) * frac)
        if idx > 0 and total_pos > 0:
            results[f'recall_at_{int(frac*100)}%_load'] = cumsum[idx] / total_pos

    return results


@torch.no_grad()
def compute_fpr_at_recall(model, loader, target_recall=0.95):
    """Find the FPR when TPR = target_recall."""
    model.eval()
    all_probs = []
    all_labels = []

    for cdm_seq, labels, mask in loader:
        time_to_tca = cdm_seq[:, :, 0]
        _, risk_logit, _ = model(cdm_seq, time_to_tca, mask)
        all_probs.append(torch.sigmoid(risk_logit).cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    sorted_idx = torch.argsort(probs, descending=True)
    sorted_labels = labels[sorted_idx].numpy()
    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    cum_pos = np.cumsum(sorted_labels)
    recall = cum_pos / max(n_pos, 1)

    for i in range(len(sorted_labels)):
        if recall[i] >= target_recall:
            fp = i + 1 - cum_pos[i]
            fpr = fp / max(n_neg, 1)
            threshold = probs[sorted_idx[i]].item()
            return fpr, threshold

    return 1.0, 0.0


def compute_maneuver_savings(model, loader, baseline_threshold=1e-6, model_threshold=0.5):
    """Estimate how many maneuvers the model saves vs threshold-based baseline."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for cdm_seq, labels, mask in loader:
            time_to_tca = cdm_seq[:, :, 0]
            _, risk_logit, _ = model(cdm_seq, time_to_tca, mask)
            all_probs.append(torch.sigmoid(risk_logit).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    baseline_maneuvers = len(probs)
    model_maneuvers = (probs >= model_threshold).sum().item()

    saved = baseline_maneuvers - model_maneuvers
    saved_pct = 100.0 * saved / max(baseline_maneuvers, 1)

    baseline_safe = baseline_maneuvers - labels.sum().item()
    model_safe = ((probs < model_threshold) & (labels == 0)).sum().item()

    missed = ((probs < model_threshold) & (labels == 1)).sum().item()

    return {
        'total_events': baseline_maneuvers,
        'baseline_maneuvers': baseline_maneuvers,
        'model_maneuvers': model_maneuvers,
        'saved': saved,
        'saved_pct': saved_pct,
        'missed_real_risks': missed,
    }
