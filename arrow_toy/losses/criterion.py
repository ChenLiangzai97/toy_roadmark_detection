"""Hungarian matcher + criterion with the ignore semantics (design doc 9)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

NUM_CLASSES = 6
NO_OBJECT = 6
IGNORE_INDEX = -100


def _valid_gt(targets: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervisable GT: class ids, normalized centers [x_n,y_n], directions."""
    return (targets["cls"], targets["center"], targets["dir"])


@torch.no_grad()
def hungarian_match(pred: dict, targets: dict, cls_weight: float = 1.0,
                    center_weight: float = 5.0) -> list[tuple[list[int], list[int]]]:
    """Returns per-batch (query_idx, gt_idx) pairs. Only arrow_valid GT.

    cost = -log p(c) + 5 * L1(center)
    """
    logits = pred["pred_logits"].float()
    centers = pred["pred_center_norm"].float()
    out = []
    B = logits.shape[0]
    for b in range(B):
        gt_mask = targets["mask"][b] > 0          # [N_gt] valid
        if gt_mask.sum() == 0:
            out.append(([], []))
            continue
        cls = targets["cls"][b][gt_mask]          # [Nv]
        ctr = targets["center"][b][gt_mask]       # [Nv,2]
        prob = logits[b].softmax(dim=-1)          # [Q,7]
        p_gt = prob[:, cls].clamp_min(1e-8)       # [Q,Nv]
        cost_cls = -torch.log(p_gt)               # [Q,Nv]
        cost_ctr = (centers[b].unsqueeze(1) - ctr.unsqueeze(0)).abs().sum(-1)
        C = cls_weight * cost_cls + center_weight * cost_ctr
        C_np = C.cpu().numpy()
        if C_np.shape[0] < C_np.shape[1]:
            raise RuntimeError(
                f"more valid GT ({C_np.shape[1]}) than queries "
                f"({C_np.shape[0]}); refuse to truncate")
        qi, gi = linear_sum_assignment(C_np)
        out.append((list(map(int, qi)), list(map(int, gi))))
    return out


def arrow_criterion(pred: dict, targets: dict, matches,
                    class_weight: float = 1.0, center_weight: float = 5.0,
                    orientation_weight: float = 1.0,
                    no_object_weight: float = 0.1
                    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Losses per design 9.2 with sample-level ignore (6.6).

    - matched query -> its class
    - unmatched query -> no-object IF allow_no_object else IGNORE_INDEX
    - class CE weights [1]*6 + [0.1]; weighted mean semantics
    - center L1 over matched; orientation cosine over matched+orientation_valid
    Graph-compatible zeros when nothing is supervised.
    """
    logits = pred["pred_logits"].float()
    centers = pred["pred_center_norm"].float()
    raw_dir = pred["pred_direction_raw"].float()
    B, Q = logits.shape[:2]

    # ---- build classification targets for the whole batch (vectorized) ----
    tgt = torch.full((B, Q), IGNORE_INDEX, dtype=torch.long,
                     device=logits.device)
    pos_b, pos_q, pos_g = [], [], []
    for b in range(B):
        qi, gi = matches[b]
        if not gi:
            continue
        valid_idx = torch.nonzero(targets["mask"][b] > 0).squeeze(-1)
        gt_rows = valid_idx[torch.tensor(gi, device=logits.device)]
        q_t = torch.tensor(qi, device=logits.device)
        tgt[b, q_t] = targets["cls"][b][gt_rows]
        pos_b.extend([b] * len(qi))
        pos_q.extend(qi)
        pos_g.extend(gt_rows.tolist())
    allow = targets["allow_no_object"].to(logits.device).view(B, 1)
    unmatched_and_ignored = (tgt == IGNORE_INDEX)
    background_mask = unmatched_and_ignored & allow
    tgt[background_mask] = NO_OBJECT

    # classification CE (weighted mean over participating queries)
    keep = tgt != IGNORE_INDEX
    if keep.any():
        w = torch.ones(NUM_CLASSES + 1, device=logits.device)
        w[NO_OBJECT] = no_object_weight
        cls_loss = F.cross_entropy(logits[keep], tgt[keep], weight=w)
    else:
        cls_loss = logits.sum() * 0.0

    n_pos = len(pos_q)
    if n_pos:
        q_idx = torch.tensor(pos_q, device=logits.device)
        b_idx = torch.tensor(pos_b, device=logits.device)
        g_idx = torch.tensor(pos_g, device=logits.device)
        pred_c = centers[b_idx, q_idx]
        gt_c = targets["center"][b_idx, g_idx]
        center_loss = (pred_c - gt_c).abs().sum(-1).mean()
        ov = targets["orientation_valid"][b_idx, g_idx] > 0
        if ov.any():
            d_pred, ok = safe_norm(raw_dir[b_idx[ov], q_idx[ov]])
            d_gt = targets["dir"][b_idx, g_idx][ov]
            ok_f = ok.float()
            cos = (d_pred * d_gt).sum(-1)
            ori_loss = ((1 - cos) * ok_f).sum() / ok_f.sum().clamp_min(1.0)
            n_ori = int(ov.sum())
        else:
            ori_loss = raw_dir.sum() * 0.0
            n_ori = 0
    else:
        center_loss = centers.sum() * 0.0
        ori_loss = raw_dir.sum() * 0.0
        n_ori = 0
    total = (class_weight * cls_loss + center_weight * center_loss
             + orientation_weight * ori_loss)
    stats = {"n_pos": n_pos, "n_ori": n_ori}
    return {
        "loss_cls": cls_loss,
        "loss_center": center_loss,
        "loss_ori": ori_loss,
        "loss_total": total,
    }, stats


def safe_norm(v: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    n = v.norm(dim=-1)
    ok = n > eps
    return v / n.clamp_min(eps).unsqueeze(-1), ok
