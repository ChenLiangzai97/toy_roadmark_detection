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

    cls_losses = []
    center_losses = []
    ori_losses = []
    n_pos = 0
    n_ori = 0
    for b in range(B):
        qi, gi = matches[b]
        allow_no = bool(targets["allow_no_object"][b])
        tgt = torch.full((Q,), IGNORE_INDEX, dtype=torch.long,
                         device=logits.device)
        if gi:
            valid_idx = torch.nonzero(targets["mask"][b] > 0).squeeze(-1)
            gt_rows = valid_idx[torch.tensor(gi, device=logits.device)]
            tgt[torch.tensor(qi, device=logits.device)] = targets["cls"][b][gt_rows]
        unmatched = (tgt == IGNORE_INDEX).nonzero().squeeze(-1)
        if allow_no:
            tgt[unmatched] = NO_OBJECT
        keep = tgt != IGNORE_INDEX
        if keep.any():
            w = torch.ones(NUM_CLASSES + 1, device=logits.device)
            w[NO_OBJECT] = no_object_weight
            cls_losses.append(
                F.cross_entropy(logits[b][keep], tgt[keep], weight=w))
        else:
            cls_losses.append(logits[b].sum() * 0.0)
        if gi:
            q_idx = torch.tensor(qi, device=logits.device)
            g_idx = torch.tensor(gi, device=logits.device)
            pred_c = centers[b][q_idx]
            valid_rows = torch.nonzero(targets["mask"][b] > 0).squeeze(-1)
            gt_rows = valid_rows[g_idx]
            gt_c = targets["center"][b][gt_rows]
            center_losses.append((pred_c - gt_c).abs().sum(-1).mean())
            n_pos += len(qi)
            # orientation on matched with orientation_valid
            ov = targets["orientation_valid"][b][gt_rows] > 0
            if ov.any():
                d_pred, ok = safe_norm(raw_dir[b][q_idx[ov]])
                d_gt = targets["dir"][b][gt_rows[ov]]
                ok_f = ok.float()
                # cosine loss: 1 - dot; zero-vector rows contribute 0 grad
                cos = (d_pred * d_gt).sum(-1)
                loss = ((1 - cos) * ok_f).sum() / ok_f.sum().clamp_min(1.0)
                ori_losses.append(loss)
                n_ori += int(ov.sum())
            else:
                ori_losses.append(raw_dir[b].sum() * 0.0)
        else:
            center_losses.append(centers[b].sum() * 0.0)
            ori_losses.append(raw_dir[b].sum() * 0.0)
    cls_loss = torch.stack(cls_losses).mean()
    center_loss = torch.stack(center_losses).mean()
    ori_loss = torch.stack(ori_losses).mean()
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
