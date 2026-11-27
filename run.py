#!/usr/bin/env python3
"""

Each separate Python stage is executed in one shared namespace so the
existing validated notebook computations, intermediate arrays, and helper
functions are preserved exactly rather than silently changing the method.
"""
import argparse
import os
import sys
from pathlib import Path

STAGES = [
    "01_setup.py",
    "02_earth_observation.py",
    "03_sample_features.py",
    "04_monthly_lst.py",
    "05_data_quality_plots.py",
    "06_e1_lst_prediction.py",
    "07_e2_context_reference.py",
    "08_e3_geoheat.py",
    "09_e4_temporal_stability.py"
    
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="HeatContext: Dubai GeoFM-guided urban thermal reference analysis")
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"), help="Google Earth Engine-enabled Google Cloud project ID")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for Earth Engine CSVs and results")
    parser.add_argument("--refresh-data", action="store_true", help="Re-download Earth Engine data even if cached")
    parser.add_argument("--skip-maps", action="store_true", help="Skip full-resolution satellite panels and optional HTML map")
    parser.add_argument("--list-stages", action="store_true", help="Show pipeline stages and exit")
    args = parser.parse_args(argv)
    if args.list_stages:
        for i, step in enumerate(STAGES, 1):
            print(f"{i:02d}. {step}")
        return 0
    if not args.project:
        parser.error("Supply --project PROJECT_ID or set EE_PROJECT in the environment")
    os.environ["EE_PROJECT"] = args.project
    os.environ["HEATCONTEXT_OUTPUT_DIR"] = str(Path(args.output_dir).expanduser().resolve())
    os.environ["HEATCONTEXT_REFRESH_DATA"] = "1" if args.refresh_data else "0"
    os.environ.setdefault("MPLBACKEND", "Agg")
    stage_dir = Path(__file__).resolve().parent / "heatcontext" / "stages"
    namespace = {"__name__": "__heatcontext_pipeline__", "__file__": str(Path(__file__).resolve())}
    for n, filename in enumerate(STAGES, 1):
        if args.skip_maps and filename in {"11_raster_downloads.py", "12_satellite_maps.py", "13_interactive_map.py"}:
            print(f"[skip] {filename} (--skip-maps)", flush=True)
            continue
        path = stage_dir / filename
        print(f"\n[{n:02d}/{len(STAGES)}] {filename}", flush=True)
        source = path.read_text(encoding="utf-8")
        # Explicit same-session execution preserves original model/data semantics.
        try:
            exec(compile(source, str(path), "exec"), namespace)
        except Exception:
            print(f"FAILED in {filename}; partial outputs remain in {args.output_dir}", file=sys.stderr)
            raise
    print("\nCompleted. Results are in:", os.environ["HEATCONTEXT_OUTPUT_DIR"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
