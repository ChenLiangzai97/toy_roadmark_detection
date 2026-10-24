// BEV grid contract mirroring arrow_toy/geometry/bev.py.
// Row 0 = farthest (x_max), col 0 = leftmost (y_max).
export interface BevRange {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}

export const BEV_RANGE_DEFAULT: BevRange = {
  x_min: 2.0, x_max: 18.0, y_min: -4.0, y_max: 4.0,
};
export const BEV_FEATURE_HW: [number, number] = [64, 32]; // 0.25 m/cell
export const BEV_MASK_HW: [number, number] = [320, 160];  // 0.05 m/cell

export class BevGrid {
  readonly rows: number;
  readonly cols: number;
  constructor(
    public readonly xMin: number, public readonly xMax: number,
    public readonly yMin: number, public readonly yMax: number,
    rows: number, cols: number
  ) {
    this.rows = rows;
    this.cols = cols;
  }
  get dx(): number { return (this.xMax - this.xMin) / this.rows; }
  get dy(): number { return (this.yMax - this.yMin) / this.cols; }

  static feature(r: BevRange = BEV_RANGE_DEFAULT, hw = BEV_FEATURE_HW): BevGrid {
    return new BevGrid(r.x_min, r.x_max, r.y_min, r.y_max, hw[0], hw[1]);
  }
  static mask(r: BevRange = BEV_RANGE_DEFAULT, hw = BEV_MASK_HW): BevGrid {
    return new BevGrid(r.x_min, r.x_max, r.y_min, r.y_max, hw[0], hw[1]);
  }

  cellCenter(r: number, c: number): [number, number] {
    return [this.xMax - (r + 0.5) * this.dx, this.yMax - (c + 0.5) * this.dy];
  }
  metricToRc(x: number, y: number): [number, number] {
    return [(this.xMax - x) / this.dx - 0.5, (this.yMax - y) / this.dy - 0.5];
  }
  metricToCell(x: number, y: number): [number, number] {
    const [r, c] = this.metricToRc(x, y);
    return [Math.floor(r + 0.5), Math.floor(c + 0.5)];
  }
  normalize(x: number, y: number): [number, number] {
    return [
      (x - this.xMin) / (this.xMax - this.xMin),
      (y - this.yMin) / (this.yMax - this.yMin),
    ];
  }
  denormalize(xn: number, yn: number): [number, number] {
    return [
      this.xMin + xn * (this.xMax - this.xMin),
      this.yMin + yn * (this.yMax - this.yMin),
    ];
  }
  contains(x: number, y: number): boolean {
    return x >= this.xMin && x <= this.xMax && y >= this.yMin && y <= this.yMax;
  }
}
