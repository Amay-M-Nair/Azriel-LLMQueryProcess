"""Puts server/ on sys.path so tests can import its packages, and keeps
ROADMAP.md's boxes in step with the tests that define them.

pytest prepends the directory holding this conftest.py, which is what makes
`from backend import ...` work without installing the project. pyproject.toml
also lists server/ under pythonpath, for runs that start elsewhere.

A test marked `@pytest.mark.roadmap("x.y")` is the "done when" check for
subphase x.y. After a run, each subphase that had marked tests in it is ticked
if they all passed and unticked otherwise (see tools/roadmap.py). Set
GROUNDEDQA_NO_ROADMAP=1 to run without touching ROADMAP.md.
"""

import os

_marked: dict[str, str] = {}          # test node id -> subphase
_outcomes: dict[str, bool] = {}       # subphase -> every marked test passed


def pytest_collection_modifyitems(items):
    for item in items:
        marker = item.get_closest_marker("roadmap")
        if marker and marker.args:
            _marked[item.nodeid] = str(marker.args[0])


def pytest_runtest_logreport(report):
    subphase = _marked.get(report.nodeid)
    if subphase is None:
        return
    # A skip counts as not done: a check that did not run proved nothing.
    ok = report.passed or (report.when != "call" and not (report.failed or report.skipped))
    if report.when == "call" or report.failed or report.skipped:
        _outcomes[subphase] = _outcomes.get(subphase, True) and ok


def pytest_sessionfinish(session, exitstatus):
    if not _outcomes or os.environ.get("GROUNDEDQA_NO_ROADMAP") == "1":
        return
    if exitstatus == 2:               # interrupted: outcomes are incomplete
        return
    from tools import roadmap

    if roadmap.update(_outcomes):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            done = sorted(k for k, v in _outcomes.items() if v)
            reporter.write_line(f"ROADMAP.md updated (passing: {', '.join(done) or 'none'})")
