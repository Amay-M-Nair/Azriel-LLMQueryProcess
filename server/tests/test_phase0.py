"""The "done when" checks for Phase 0 subphases that are about the repo itself."""

import re
import subprocess

import pytest

from utils import config

KEY_SHAPES = re.compile(
    r"AIza[0-9A-Za-z_-]{30,}"          # Google API key
    r"|AQ\.[A-Za-z0-9_-]{40,}"         # Google key, newer format
    r"|sk-ant-[A-Za-z0-9_-]{20,}"      # Anthropic
    r"|gsk_[A-Za-z0-9]{40,}"           # Groq
)


def tracked_files() -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=config.REPO_ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.splitlines()


# --- 0.1 Secret hygiene -------------------------------------------------------

@pytest.mark.roadmap("0.1")
def test_no_key_shaped_string_is_tracked():
    leaks = []
    for name in tracked_files():
        path = config.REPO_ROOT / name
        if not path.is_file() or path.suffix in {".svg", ".png", ".jpg", ".pdf", ".docx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        leaks += [f"{name}: {m.group(0)[:8]}..." for m in KEY_SHAPES.finditer(text)]
    assert not leaks, f"key-shaped strings in tracked files: {leaks}"


@pytest.mark.roadmap("0.1")
def test_env_example_has_every_placeholder_and_no_values():
    lines = (config.REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    settings = dict(
        line.split("=", 1) for line in lines
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    )
    for name in ("GROQ_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "SEC_USER_AGENT", "DATABASE_URL"):
        assert name in settings, f"{name} missing from .env.example"
    filled = {k: v for k, v in settings.items() if v.strip()}
    assert not filled, f".env.example carries values: {sorted(filled)}"


# --- 0.2 Remove what the new design replaces ---------------------------------

@pytest.mark.roadmap("0.2")
def test_hosting_leftovers_are_gone():
    for name in ("packages.txt", "server/tests/test_deployment.py"):
        assert not (config.REPO_ROOT / name).exists(), f"{name} still exists"
    for setting in ("API_URL", "COLLECTION_MODE", "PUBLIC"):
        assert not hasattr(config, setting), f"config.{setting} still exists"

    import api.client
    assert not hasattr(api.client, "HttpClient")


# --- 0.3 Repo layout and rename -----------------------------------------------

@pytest.mark.roadmap("0.3")
def test_layout():
    root = config.REPO_ROOT
    assert config.PROJECT_ROOT == root / "server"
    for package in ("api", "backend", "ingestion", "vectorstore", "utils", "evaluation", "tools"):
        assert (root / "server" / package / "__init__.py").exists(), f"server/{package} missing"
    assert (root / "web").is_dir()
    assert config.DATA_DIR == root / "data", "data/ must stay at the repo root (git-ignored)"


@pytest.mark.roadmap("0.3")
def test_renamed():
    readme = (config.REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Grounded QA for SEC Filings")
    from utils import logs
    assert logs.get("x").name.startswith("groundedqa.")
    assert config.LOG_FILE.name == "groundedqa.log"


# --- 0.5 Version pinning ------------------------------------------------------

def requirement_pins() -> dict[str, str]:
    pins = {}
    for raw in (config.REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        assert "==" in line, f"not pinned exactly: {raw.strip()!r}"
        name, version = line.split("==", 1)
        pins[re.sub(r"\[.*\]", "", name).strip().lower()] = version.strip()
    return pins


@pytest.mark.roadmap("0.5")
def test_every_requirement_is_pinned_and_installed_at_that_version():
    from importlib.metadata import PackageNotFoundError, version

    mismatched = {}
    for name, pinned in requirement_pins().items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            mismatched[name] = (pinned, "not installed")
            continue
        if installed.split("+")[0] != pinned:          # torch reports 2.15.0+cpu
            mismatched[name] = (pinned, installed)
    assert not mismatched, f"pinned vs installed: {mismatched}"


@pytest.mark.roadmap("0.5")
def test_versions_checked_is_logged():
    roadmap = (config.REPO_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    section = roadmap.split("## Versions checked", 1)[1].split("\n## ", 1)[0]
    assert "| 0 | groq |" in section


# --- 0.7 Memory spike for Render ----------------------------------------------

RENDER_FREE_MB = 512


@pytest.mark.roadmap("0.7")
def test_the_memory_spike_was_measured_and_decided():
    import json

    path = config.PROJECT_ROOT / "evaluation" / "results" / "spike_memory.json"
    assert path.exists(), "run  python -m tools.spike_memory  (from server/)"
    report = json.loads(path.read_text(encoding="utf-8"))
    peaks = {m["configuration"]: m["peak_mb"] for m in report["measurements"]}
    chosen = peaks["embedder (fp32) + reranker (int8)"]
    assert chosen < RENDER_FREE_MB, f"the chosen configuration peaks at {chosen} MB"

    roadmap = (config.REPO_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    findings = roadmap.split("## Findings", 1)[1]
    assert "### 0.7" in findings and "**Decision:**" in findings
