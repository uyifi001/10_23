#!/usr/bin/env python3
"""Post-process deal.II PF-CZM output.

Usage:
  python postprocess.py --csv force_displacement.csv --out force_displacement.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="force_displacement.csv", help="force-displacement CSV from deal.II run")
    p.add_argument("--out", default="force_displacement.png", help="output figure path")
    p.add_argument("--vtu", default=None, help="optional VTU file for quick field summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["imposed_displacement"], df["reaction_force"], "-o", ms=3)
        ax.set_xlabel("Imposed displacement (mm)")
        ax.set_ylabel("Reaction force (N/mm)")
        ax.set_title("Force-Displacement")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=180)
        print(f"Saved {args.out}")
    except Exception as exc:  # noqa: BLE001
        print(f"Plot skipped: {exc}")

    if args.vtu:
        try:
            import meshio

            mesh = meshio.read(args.vtu)
            print(f"VTU points={mesh.points.shape[0]}, cells={sum(len(c.data) for c in mesh.cells)}")
            print("Point data fields:", list(mesh.point_data.keys()))
            print("Cell data fields:", list(mesh.cell_data_dict.keys()))
        except Exception as exc:  # noqa: BLE001
            print(f"VTU summary skipped: {exc}")


if __name__ == "__main__":
    main()
