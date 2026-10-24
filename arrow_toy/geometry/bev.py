"""BEV grid contract (design doc 4.3/4.4).

BEV images and features: row 0 = farthest, col 0 = leftmost
("front at top, left at left").
  x = x_max - (r + 0.5) * dx
  y = y_max - (c + 0.5) * dy
Feature maps are 0.25 m/cell (64x32 over x in [2,18], y in [-4,4]); the
high-resolution GT instance mask is 0.05 m/cell (320x160).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

BEV_RANGE_DEFAULT = (2.0, 18.0, -4.0, 4.0)  # xmin, xmax, ymin, ymax
BEV_FEATURE_HW = (64, 32)                   # 0.25 m/cell
BEV_MASK_HW = (320, 160)                    # 0.05 m/cell


@dataclass(frozen=True)
class BevGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    rows: int
    cols: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.rows

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.cols

    @staticmethod
    def feature(x_min=2.0, x_max=18.0, y_min=-4.0, y_max=4.0,
                rows=BEV_FEATURE_HW[0], cols=BEV_FEATURE_HW[1]) -> "BevGrid":
        return BevGrid(x_min, x_max, y_min, y_max, rows, cols)

    @staticmethod
    def mask(x_min=2.0, x_max=18.0, y_min=-4.0, y_max=4.0,
             rows=BEV_MASK_HW[0], cols=BEV_MASK_HW[1]) -> "BevGrid":
        return BevGrid(x_min, x_max, y_min, y_max, rows, cols)

    # -- mapping --

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        x = self.x_max - (r + 0.5) * self.dx
        y = self.y_max - (c + 0.5) * self.dy
        return x, y

    def metric_to_rc(self, x: float, y: float) -> tuple[float, float]:
        """Continuous (r, c); use floor for the containing cell index."""
        r = (self.x_max - x) / self.dx - 0.5
        c = (self.y_max - y) / self.dy - 0.5
        return r, c

    def metric_to_cell(self, x: float, y: float) -> tuple[int, int]:
        r, c = self.metric_to_rc(x, y)
        return int(math.floor(r + 0.5)), int(math.floor(c + 0.5))

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def normalize(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.x_min) / (self.x_max - self.x_min),
                (y - self.y_min) / (self.y_max - self.y_min))

    def denormalize(self, x_n: float, y_n: float) -> tuple[float, float]:
        return (self.x_min + x_n * (self.x_max - self.x_min),
                self.y_min + y_n * (self.y_max - self.y_min))

    def all_cell_centers(self) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = self.cell_center(r, c)
                xs.append(x)
                ys.append(y)
        return xs, ys

    def corners_metric(self) -> dict[str, tuple[float, float]]:
        """Named corners for semantics checks (front/back x, left/right y)."""
        return {
            "front_left": (self.x_max, self.y_max),
            "front_right": (self.x_max, self.y_min),
            "back_left": (self.x_min, self.y_max),
            "back_right": (self.x_min, self.y_min),
        }

    def corners_rc(self) -> dict[str, tuple[int, int]]:
        return {
            "front_left": (0, 0),
            "front_right": (0, self.cols - 1),
            "back_left": (self.rows - 1, 0),
            "back_right": (self.rows - 1, self.cols - 1),
        }


def bev_range_dict(grid: BevGrid) -> dict:
    return {"x_min": grid.x_min, "x_max": grid.x_max,
            "y_min": grid.y_min, "y_max": grid.y_max}


def grid_from_config(cfg: Sequence[float], hw: Sequence[int]) -> BevGrid:
    x_min, x_max, y_min, y_max = cfg
    return BevGrid(x_min, x_max, y_min, y_max, int(hw[0]), int(hw[1]))
