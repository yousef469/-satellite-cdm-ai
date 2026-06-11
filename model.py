import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    NUM_CDM_FEATURES, HIDDEN_DIM, TCN_BLOCKS, TCN_DROPOUT,
    NUM_SUBSPACES, SUBSPACE_DIM, PROTO_DIM, NUM_PROTOTYPES, TEMPERATURE,
)


class TimeEncoding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(1, dim // 2)
        self.periods = nn.Parameter(torch.exp(torch.linspace(0, 4, dim // 2)), requires_grad=False)

    def forward(self, t):
        phase = t / self.periods
        return torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)


class DepthwiseTCNBlock(nn.Module):
    def __init__(self, d_model, dilation, dropout=0.15):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = x.permute(0, 2, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(r + x)


class DepthwiseTCN(nn.Module):
    def __init__(self, d_model=HIDDEN_DIM, num_blocks=TCN_BLOCKS, dropout=TCN_DROPOUT):
        super().__init__()
        dilations = [2 ** i for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            DepthwiseTCNBlock(d_model, d, dropout) for d in dilations
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class CDMHead(nn.Module):
    def __init__(self, d_model, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=mask)
        return self.norm(out).squeeze(1)


class PrototypeDistributionModule(nn.Module):
    def __init__(self, input_dim=HIDDEN_DIM, num_subspaces=NUM_SUBSPACES, subspace_dim=SUBSPACE_DIM, num_prototypes=NUM_PROTOTYPES):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        proto_dim = num_subspaces * subspace_dim

        self.subspace_proj = nn.ModuleList([
            nn.Linear(input_dim, subspace_dim, bias=False)
            for _ in range(num_subspaces)
        ])
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proto_dim))
        nn.init.normal_(self.prototypes, std=0.01)
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(TEMPERATURE)))

    def forward(self, x):
        subspaces = [proj(x) for proj in self.subspace_proj]
        d = torch.cat(subspaces, dim=-1)
        d_norm = F.normalize(d, dim=-1)

        p_norm = F.normalize(self.prototypes, dim=-1)
        proto_sim = torch.mm(d_norm, p_norm.t())
        temp = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        proto_logits = proto_sim / temp

        return proto_logits, d_norm

    def orthogonality_loss(self):
        loss = 0.0
        count = 0
        for i in range(self.num_subspaces):
            for j in range(i + 1, self.num_subspaces):
                Wi = F.normalize(self.subspace_proj[i].weight, dim=-1)
                Wj = F.normalize(self.subspace_proj[j].weight, dim=-1)
                loss += (Wi @ Wj.t()).norm(p='fro') ** 2
                count += 1
        return loss / count if count > 0 else 0.0


class CDMRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_enc = TimeEncoding(dim=64)
        time_enc_dim = 64

        self.norm_in = nn.LayerNorm(NUM_CDM_FEATURES)

        self.input_proj = nn.Sequential(
            nn.Linear(NUM_CDM_FEATURES + time_enc_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.temporal = DepthwiseTCN(HIDDEN_DIM, TCN_BLOCKS, TCN_DROPOUT)
        self.reasoning = CDMHead(HIDDEN_DIM, num_heads=2)
        self.pdm = PrototypeDistributionModule(HIDDEN_DIM, NUM_SUBSPACES, SUBSPACE_DIM, NUM_PROTOTYPES)

        self.risk_head = nn.Linear(NUM_CDM_FEATURES, 1)

    def forward(self, cdm_seq, time_to_tca, mask=None):
        te = self.time_enc(time_to_tca.unsqueeze(-1))
        x = self.norm_in(cdm_seq)
        x = torch.cat([x, te], dim=-1)
        x = self.input_proj(x)
        x = self.temporal(x)
        pooled = self.reasoning(x, mask=mask)
        proto_logits, d = self.pdm(pooled)

        # Risk from the LAST VALID CDM's raw features
        if mask is not None:
            # mask: True = padded position, False = real data
            # Find last unmasked position for each sample in batch
            lengths = (~mask).sum(dim=1, dtype=torch.long) - 1  # last valid index
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(cdm_seq.shape[0], device=cdm_seq.device)
            last_feats = cdm_seq[batch_idx, lengths]
        else:
            last_feats = cdm_seq[:, -1, :]
        risk = self.risk_head(last_feats).squeeze(-1)

        return proto_logits, risk, d

    def get_aux_losses(self):
        return self.pdm.orthogonality_loss()


class ConjunctionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, d, labels):
        d = F.normalize(d, dim=-1)
        sim = torch.mm(d, d.t()) / self.temp
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1)
        pos = (exp_sim * pos_mask).sum(dim=1)
        has_pos = pos_mask.sum(dim=1) > 0

        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=d.device)

        loss = -(torch.log(pos[has_pos] / (denom[has_pos] + 1e-8) + 1e-8)).mean()
        return loss


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
