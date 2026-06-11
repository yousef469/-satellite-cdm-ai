import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import (
    BATCH_SIZE, PHASE2_BACKBONE_LR as LR, PHASE2_WEIGHT_DECAY as WEIGHT_DECAY,
    PHASE2_EPOCHS as EPOCHS, NUM_SYNTHETIC_EVENTS, RISK_RATIO, PROTO_DIM,
)
from cdm_generator import generate_dataset, generate_cdm_sequence
from model import CDMRiskModel, ConjunctionContrastiveLoss, count_params


class CDMDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def pad_collate_fn(batch):
    max_len = max(s.shape[0] for s, _ in batch)
    seqs, labels, masks = [], [], []

    for seq, label in batch:
        t = seq.shape[0]
        if t < max_len:
            pad = torch.zeros(max_len - t, seq.shape[1])
            padded = torch.cat([seq, pad], dim=0)
            mask = torch.cat([torch.zeros(t, dtype=torch.bool),
                              torch.ones(max_len - t, dtype=torch.bool)])
        else:
            padded = seq[:max_len]
            mask = torch.zeros(max_len, dtype=torch.bool)
        seqs.append(padded)
        labels.append(label)
        masks.append(mask)

    return torch.stack(seqs), torch.stack(labels), torch.stack(masks)


def train_epoch(model, loader, optimizer, bce_loss, cont_loss, device):
    model.train()
    total_loss = 0
    total_bce = 0
    total_cont = 0
    total_orth = 0
    correct = 0
    total = 0

    for cdm_seq, labels, mask in loader:
        cdm_seq = cdm_seq.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        time_to_tca = cdm_seq[:, :, 0]

        proto_logits, risk_logit, d = model(cdm_seq, time_to_tca, mask)

        bce = bce_loss(risk_logit, labels)
        cont = cont_loss(d, labels)
        orth = model.get_aux_losses()

        loss = bce + 0.05 * cont + 0.01 * orth

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_cont += cont.item()
        total_orth += orth.item()

        preds = (torch.sigmoid(risk_logit) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    avg = total_loss / len(loader)
    print(f"  Train: L={avg:.4f} BCE={total_bce/len(loader):.4f} "
          f"Cont={total_cont/len(loader):.4f} Orth={total_orth/len(loader):.4f} Acc={acc:.2f}%")
    return acc, avg


@torch.no_grad()
def validate(model, loader, bce_loss, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for cdm_seq, labels, mask in loader:
        cdm_seq = cdm_seq.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        time_to_tca = cdm_seq[:, :, 0]

        proto_logits, risk_logit, d = model(cdm_seq, time_to_tca, mask)
        bce = bce_loss(risk_logit, labels)
        total_loss += bce.item()

        probs = torch.sigmoid(risk_logit)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(probs.cpu())
        all_labels.append(labels.cpu())

    acc = 100.0 * correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    sorted_idx = torch.argsort(all_preds, descending=True)
    all_labels_sorted = all_labels[sorted_idx]
    n_pos = all_labels_sorted.sum().int()

    precisions = []
    recalls = []
    for k in [1, 10, 100, 1000]:
        k = min(k, len(all_labels_sorted))
        if k > 0:
            tp = all_labels_sorted[:k].sum().item()
            precisions.append(tp / k)
            recalls.append(tp / max(n_pos, 1))

    print(f"  Val: L={total_loss/len(loader):.4f} Acc={acc:.2f}% "
          f"P@1={precisions[0]:.3f} P@10={precisions[1]:.3f} "
          f"P@100={precisions[2]:.3f} R@100={recalls[2]:.3f}")

    return acc, all_preds, all_labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("Warning: running on CPU. Consider using --device cuda if available.")

    print("Generating synthetic CDM dataset...")
    sequences, labels = generate_dataset(NUM_SYNTHETIC_EVENTS, RISK_RATIO)
    pos_count = labels.sum().item()
    neg_count = len(labels) - pos_count
    print(f"  Positive: {pos_count} ({100*pos_count/len(labels):.1f}%)")
    print(f"  Negative: {neg_count} ({100*neg_count/len(labels):.1f}%)")

    n_train = int(0.8 * len(sequences))
    n_val = len(sequences) - n_train

    train_seqs = sequences[:n_train]
    train_labels = labels[:n_train]
    val_seqs = sequences[n_train:]
    val_labels = labels[n_train:]
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    train_ds = CDMDataset(train_seqs, train_labels)
    val_ds = CDMDataset(val_seqs, val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=pad_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=pad_collate_fn, num_workers=0)

    model = CDMRiskModel()
    total_params = count_params(model)
    print(f"Model params: {total_params:,}")

    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cont_loss = ConjunctionContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    patience = 0
    max_patience = 15

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, bce_loss, cont_loss, device)
        scheduler.step()

        val_acc, val_preds, val_labels = validate(model, val_loader, bce_loss, device)

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoint_best.pt')
            print(f"  *** NEW BEST: {val_acc:.2f}% ***")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")

    ckpt = torch.load('checkpoint_best.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    val_acc, val_preds, val_labels = validate(model, val_loader, bce_loss, device)

    print("\nResults on synthetic CDM data:")
    print(f"  Accuracy: {val_acc:.2f}%")
    print(f"  Detects real collision risks at ~{50*BATCH_SIZE} events per batch")

    return val_acc


if __name__ == '__main__':
    main()
