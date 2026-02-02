from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any

class StreamWriter:
    def __init__(self, outdir: str | Path, manifest: Dict[str, Any]):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.outdir/"run.json", manifest)
        (self.outdir/"checkpoints").mkdir(exist_ok=True)

    def write_checkpoint(self, step: int, t: float, arrays: Dict[str, np.ndarray]):
        fn = self.outdir/"checkpoints"/f"chk_{step:06d}.npz"
        np.savez_compressed(fn, t=t, **arrays)
        return fn

    def append_metric(self, row: Dict[str, Any]):
        with open(self.outdir/"metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def _write_json(self, path: Path, obj: Dict[str, Any]):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def write_metric(self, row: dict) -> None:
        """Backward-compatible alias."""
        self.append_metric(row)