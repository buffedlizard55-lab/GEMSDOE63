"""Hide-and-recover spatial holdout and plausible label-world ensemble generator.

Protocol rules:
- Withhold whole fault segments with a buffer (300 m / 3 px).
- Visible faults masked pixel-exactly.
- Shift withheld segments by up to +/- 4 px (calibrated to Hermant et al. 400 m offset).
- Drop 10-30% of withheld segments.
- Add decoy segments at observed segment density.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from .metric import dti_binary, cvar_penalized_score

BUFFER_PX = 3


class LabelWorldEnsemble:
    def __init__(self, labels_path: str, holdout_fold: str = "nw", seed: int = 20261011):
        import rasterio
        with rasterio.open(labels_path) as src:
            self.labels = src.read(1)
            self.nodata = src.nodata
            self.foot = (self.labels != self.nodata)
            self.cat = (self.labels == 1)

        self.h, self.w = self.labels.shape
        self.mid_r, self.mid_c = self.h // 2, self.w // 2

        struct = ndi.generate_binary_structure(2, 2)
        lab, n_segs = ndi.label(self.cat, structure=struct)
        rr, cc = np.nonzero(lab)
        labels_flat = lab[rr, cc]
        order = np.argsort(labels_flat)
        rr_sorted = rr[order]
        cc_sorted = cc[order]
        labels_sorted = labels_flat[order]
        splits = np.flatnonzero(np.diff(labels_sorted)) + 1
        split_indices = np.concatenate(([0], splits, [len(labels_sorted)]))

        self.seg_list = []
        self.holdout_segs = []
        self.visible_segs = []

        for i in range(len(split_indices) - 1):
            idx0 = split_indices[i]
            idx1 = split_indices[i+1]
            s_rr = rr_sorted[idx0:idx1]
            s_cc = cc_sorted[idx0:idx1]
            self.seg_list.append((s_rr, s_cc))
            cr = s_rr.mean()
            cc_mean = s_cc.mean()
            # 4 quadrants: nw, ne, sw, se
            is_nw = (cr < self.mid_r and cc_mean < self.mid_c)
            if holdout_fold == "nw" and is_nw:
                self.holdout_segs.append(i)
            else:
                self.visible_segs.append(i)

        # Build visible catalogue mask and buffer
        self.visible_cat = np.zeros_like(self.cat)
        for i in self.visible_segs:
            s_rr, s_cc = self.seg_list[i]
            self.visible_cat[s_rr, s_cc] = True

        self.visible_buffer = ndi.binary_dilation(self.visible_cat, structure=struct, iterations=BUFFER_PX)
        # Scored domain: footprint excluding visible faults and buffer
        self.scored_domain = self.foot & ~self.visible_buffer

        self.rng = np.random.default_rng(seed)

    def generate_worlds(self, n_worlds: int = 12, shift_max: int = 4, drop_range: tuple[float, float] = (0.10, 0.30), decoy_frac: float = 0.05) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate N plausible label worlds on the holdout.
        Returns list of (world_mask, world_distance_transform).
        """
        worlds = []
        for _ in range(n_worlds):
            drop_rate = self.rng.uniform(drop_range[0], drop_range[1])
            keep = self.rng.uniform(0, 1, size=len(self.holdout_segs)) > drop_rate
            kept_ids = [self.holdout_segs[k] for k in range(len(self.holdout_segs)) if keep[k]]

            w_mask = np.zeros((self.h, self.w), dtype=bool)
            for s_idx in kept_ids:
                s_rr, s_cc = self.seg_list[s_idx]
                dy = self.rng.integers(-shift_max, shift_max + 1)
                dx = self.rng.integers(-shift_max, shift_max + 1)
                sh_r = np.clip(s_rr + dy, 0, self.h - 1)
                sh_c = np.clip(s_cc + dx, 0, self.w - 1)
                w_mask[sh_r, sh_c] = True

            # Decoy segments
            n_decoys = int(len(self.holdout_segs) * decoy_frac)
            decoy_segs = self.rng.choice(len(self.seg_list), size=n_decoys, replace=True)
            for d_idx in decoy_segs:
                d_rr, d_cc = self.seg_list[d_idx]
                dy = self.rng.integers(0, self.mid_r - 50) - int(d_rr.mean())
                dx = self.rng.integers(0, self.mid_c - 50) - int(d_cc.mean())
                sh_r = np.clip(d_rr + dy, 0, self.h - 1)
                sh_c = np.clip(d_cc + dx, 0, self.w - 1)
                w_mask[sh_r, sh_c] = True

            w_mask &= self.scored_domain
            dg = distance_transform_edt(~w_mask)
            worlds.append((w_mask, dg))
        return worlds

    def evaluate_cvar(self, pred_dots: np.ndarray, worlds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        """Evaluate a binary prediction mask across all label worlds using exact DTI."""
        p = pred_dots & self.scored_domain
        if p.sum() == 0:
            return cvar_penalized_score([0.0] * len(worlds))

        dp = distance_transform_edt(~p)
        scores = []
        for w_mask, dg in worlds:
            k_dp = np.maximum(1.0 - dp[w_mask] / 3.0, 0.0)
            tp = float(k_dp.sum())
            fn = float(w_mask.sum()) - tp
            k_dg = np.maximum(1.0 - dg[p] / 3.0, 0.0)
            fp = float((1.0 - k_dg).sum())
            dti = tp / (tp + 0.2 * fp + 0.8 * fn + 1e-12)
            scores.append(dti)

        return cvar_penalized_score(scores)
