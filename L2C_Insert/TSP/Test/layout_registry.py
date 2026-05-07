from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    supports_rate: bool
    region_key: Optional[str]
    region_label: str


LAYOUT_SPECS: Dict[str, LayoutSpec] = {
    "explosion": LayoutSpec(
        name="explosion",
        supports_rate=True,
        region_key="explosion_regions",
        region_label="explosion",
    ),
    "implosion": LayoutSpec(
        name="implosion",
        supports_rate=False,
        region_key="implosion_regions",
        region_label="implosion",
    ),
    "clustered": LayoutSpec(
        name="clustered",
        supports_rate=False,
        region_key="cluster_regions",
        region_label="cluster",
    ),
}


def get_layout_spec(layout: str) -> LayoutSpec:
    if layout not in LAYOUT_SPECS:
        supported = ", ".join(sorted(LAYOUT_SPECS.keys()))
        raise ValueError(f"Unknown layout={layout!r}. Supported: {supported}")
    return LAYOUT_SPECS[layout]


def build_generation_metadata(
    *,
    layout: str,
    seed: Optional[int],
    range_min: float,
    range_max: float,
    rate: float,
    num_centers: int,
    layout_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    spec = get_layout_spec(layout)
    metadata: Dict[str, Any] = {
        "layout": layout,
        "seed": seed,
        "range_min": float(range_min),
        "range_max": float(range_max),
        "rate": float(rate) if spec.supports_rate else None,
        "num_centers": int(num_centers),
    }
    if layout_metadata:
        metadata.update(dict(layout_metadata))
    return metadata


def create_default_layout_metadata(args, coords: np.ndarray) -> Optional[Dict[str, Any]]:
    layout = str(getattr(args, "layout", ""))
    if layout not in LAYOUT_SPECS:
        return None
    spec = get_layout_spec(layout)
    avg_range = (float(args.range_min) + float(args.range_max)) / 2.0

    metadata: Dict[str, Any] = build_generation_metadata(
        layout=layout,
        seed=None,
        range_min=float(args.range_min),
        range_max=float(args.range_max),
        rate=float(args.rate),
        num_centers=int(args.num_centers),
    )
    if spec.region_key:
        regions = []
        for i in range(int(args.num_centers)):
            if int(args.num_centers) > 1:
                angle = 2 * np.pi * i / int(args.num_centers)
                radius_offset = 0.2
                center = [
                    0.5 + radius_offset * np.cos(angle),
                    0.5 + radius_offset * np.sin(angle),
                ]
            else:
                center = [0.5, 0.5]
            regions.append({"center": center, "radius": float(avg_range)})
        metadata[spec.region_key] = regions
    metadata["normalization"] = {
        "min_coords": [0.0, 0.0],
        "max_coords": [1.0, 1.0],
    }
    return metadata
