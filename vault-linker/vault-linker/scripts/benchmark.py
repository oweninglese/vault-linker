import time
import json
from pathlib import Path

start = time.perf_counter()
time.sleep(0.01)
end = time.perf_counter()

out = {
    "total_seconds": end - start
}

Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
Path("benchmarks/results/run.json").write_text(
    json.dumps(out, indent=2),
    encoding="utf-8",
)
print("Benchmark written.")
