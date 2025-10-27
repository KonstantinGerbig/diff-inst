import numpy as np
from pathlib import Path
from diffinst.io_utils import StreamWriter

def test_stream_writer(tmp_path: Path):
    out = tmp_path / "out"
    w = StreamWriter(out, {"hello": "world"})
    fn = w.write_checkpoint(0, 0.0, {"a": np.arange(4)})
    assert fn.exists()
    w.append_metric({"step": 0, "ok": True})
    assert (out / "metrics.jsonl").exists()