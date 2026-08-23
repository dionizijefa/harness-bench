from pathlib import Path

import pytest

from harness_bloat_bench import harness_prefetch


@pytest.mark.parametrize(
    ("harness", "target"),
    [
        ("claude_code_agent", "ensure_claude_cached"),
        ("codex_agent", "ensure_codex_cached"),
        ("deepseek_harness", "ensure_docker_built_tree"),
        ("hermes_agent", "ensure_docker_built_tree"),
        ("omp_agent", "ensure_omp_cached"),
        ("pi_agent", "ensure_docker_built_tree"),
        ("pi_rlm_runtime", "ensure_docker_built_tree"),
        ("prime_agent", "ensure_docker_built_tree"),
    ],
)
def test_every_harness_dispatches_to_the_shared_cache(
    harness: str, target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cached = tmp_path / harness
    calls = []

    def fake_cache(*args, **kwargs):
        calls.append((args, kwargs))
        return cached

    monkeypatch.setattr(harness_prefetch, target, fake_cache)

    assert harness_prefetch.ensure_harness_cached(harness, "1.2.3", "x64") == (
        cached,
    )
    assert len(calls) == 1


def test_opencode_prefetches_glibc_and_musl_for_modern_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variants: list[bool] = []

    def fake_cache(version, artifact_version, arch, *, musl):
        variants.append(musl)
        return tmp_path / ("musl" if musl else "glibc")

    monkeypatch.setattr(harness_prefetch, "ensure_opencode_cached", fake_cache)

    paths = harness_prefetch.ensure_harness_cached("opencode", "1.18.1", "x64")

    assert paths == (tmp_path / "glibc", tmp_path / "musl")
    assert variants == [False, True]


def test_old_opencode_release_only_prefetches_its_available_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variants: list[bool] = []

    def fake_cache(version, artifact_version, arch, *, musl):
        variants.append(musl)
        return tmp_path / "opencode"

    monkeypatch.setattr(harness_prefetch, "ensure_opencode_cached", fake_cache)

    paths = harness_prefetch.ensure_harness_cached("opencode", "0.1.196", "x64")

    assert paths == (tmp_path / "opencode",)
    assert variants == [False]


def test_unknown_harness_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown harness"):
        harness_prefetch.ensure_harness_cached("mystery", "1.0.0")
