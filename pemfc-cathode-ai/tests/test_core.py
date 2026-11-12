"""Core tests: data, physics, ML hygiene. Run with: pytest -q"""
import numpy as np
import pytest

from conftest import RAW_OK

pytestmark = pytest.mark.skipif(not RAW_OK, reason="run python download_data.py first")


def test_curves_and_no_leakage():
    from src.data import load_polarization
    from src.models import FEATURES
    pol = load_polarization()
    assert pol["curve_id"].nunique() > 200
    assert "power_density" not in FEATURES          # derived from the target: would leak
    assert "source_file" not in FEATURES            # group id, not a predictor
    assert (pol["current_density"] > 0).all()


def test_loss_breakdown_adds_up():
    from src.data import load_polarization
    from src.physics import breakdown_from_curve
    pol = load_polarization()
    g = pol[pol.curve_id == pol.curve_id.iloc[0]]
    b = breakdown_from_curve(g)
    assert abs(b["kinetic"] + b["ohmic"] + b["mass_transport"] - b["total_deficit"]) < 1e-9


def test_injected_faults_move_the_right_descriptor():
    from src.data import load_polarization
    from src.physics import curve_descriptors, inject_fault
    pol = load_polarization()
    g = pol[pol.curve_id == pol.curve_id.iloc[5]]
    base = curve_descriptors(g)
    ohm = curve_descriptors(inject_fault(g, "ohmic", 0.3))
    mt = curve_descriptors(inject_fault(g, "mass_transport", 0.2))
    assert ohm.asr_ohm_cm2 > base.asr_ohm_cm2 + 0.1
    assert mt.mt_loss_at_jmax > base.mt_loss_at_jmax + 0.05


def test_surrogate_refuses_extrapolation():
    from src.data import load_polarization
    from src.models import DesignSurrogate
    s = DesignSurrogate().fit(load_polarization())
    with pytest.raises(ValueError):
        s.predict_curve(pressure=60, relative_humidity=50, membrane_compression=5, nafion_percent=20)
    assert s.cv_mae_ < 0.06


def test_holdout_keeps_files_apart():
    from src.data import load_polarization
    from src.models import holdout_split
    tr, te = holdout_split(load_polarization())
    assert set(tr.source_file).isdisjoint(set(te.source_file))
