from pathlib import Path
import tempfile

from vault_linker.config import BuildConfig
from vault_linker.pipeline.build import build

def test_build_example_vault_creates_output_and_hubs() -> None:
    repo = Path(__file__).resolve().parents[2]
    input_vault = repo / "examples" / "example_vault"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        tags = Path(td) / "TAGS.txt"

        cfg = BuildConfig(
            input_vault=input_vault,
            output_vault=out,
            cache_dir=cache,
            tags_file=tags,
            no_links=False,
            learn_tags=True,
            dry_run=False,
        )
        stats = build(cfg)

        assert stats.processed >= 1
        assert (out / "example.md").exists()
        assert (out / "Canada.md").exists()
