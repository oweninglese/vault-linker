from vault_linker.core.tags import normalize_tag_list, canonicalize_tag

def test_canonicalize_tag() -> None:
    assert canonicalize_tag(" Public Health ") == "Public-Health"

def test_normalize_tag_list() -> None:
    assert normalize_tag_list(["Canada", "canada", "Public Health"]) == ["Canada", "Public-Health"]
