import json

import pytest

from conftest import SCENARIO_DIR
from radsafe.cli import main

EXAMPLE = str(SCENARIO_DIR / "cs137_bench_source.yaml")


def test_run_prints_report(capsys):
    assert main(["run", EXAMPLE, "-n", "2000"]) == 0
    out = capsys.readouterr().out
    assert "Cs-137 bench source" in out
    for token in ("unshielded", "narrow_beam", "linear_buildup", "practical_hvl", "P(dose > limit)"):
        assert token in out


def test_run_writes_json(tmp_path, capsys):
    target = tmp_path / "report.json"
    assert main(["run", EXAMPLE, "-n", "2000", "--seed", "1", "--json", str(target)]) == 0
    report = json.loads(target.read_text())
    assert report["simulation"] == {"n_samples": 2000, "seed": 1}
    nb = report["results"]["narrow_beam"]
    assert nb["p5_msv"] <= nb["median_msv"] <= nb["p95_msv"]
    assert report["results"]["linear_buildup"]["median_ratio_to_narrow_beam"] > 1.0
    assert len(nb["exceedance"]) == 2


def test_run_is_reproducible(capsys):
    main(["run", EXAMPLE, "-n", "3000", "--seed", "9"])
    first = capsys.readouterr().out
    main(["run", EXAMPLE, "-n", "3000", "--seed", "9"])
    assert capsys.readouterr().out == first


def test_validate_ok_and_error(tmp_path, capsys):
    assert main(["validate", EXAMPLE]) == 0
    assert "ok" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    assert main(["validate", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_file_is_a_clean_error(capsys):
    assert main(["run", "/nonexistent/scenario.yaml"]) == 2
    assert "error:" in capsys.readouterr().err


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Cs-137" in out and "lead" in out and "linear_buildup" in out


def test_plot_writes_png(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    target = tmp_path / "summary.png"
    assert main(["run", EXAMPLE, "-n", "2000", "--plot", str(target)]) == 0
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
