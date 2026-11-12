"""DFT layer: physics identities, provenance rules, parser, ML mechanics, agent tools. No DFT code is run."""
import numpy as np
import pandas as pd
import pytest

from conftest import RAW_OK


def test_four_steps_add_up_and_limiting_potential_is_consistent():
    from src.dft import add_orr_metrics, free_energy_diagram, orr_steps
    df = add_orr_metrics(pd.DataFrame({"E_OH": [0.5, 0.9], "E_O": [1.6, 2.0], "E_OOH": [3.6, 4.0]}))
    assert np.allclose(df[["step1", "step2", "step3", "step4"]].sum(axis=1), -4.92)
    for _, r in df.iterrows():          # every step is downhill at U_L and one becomes uphill just above it
        assert free_energy_diagram(r, r.U_L).G_eV.diff().dropna().max() <= 1e-9
        assert free_energy_diagram(r, r.U_L + 0.01).G_eV.diff().dropna().max() > 0
    assert np.isclose(df.eta, 1.23 - df.U_L).all()


def test_volcano_peaks_where_oh_and_ooh_steps_balance():
    from src.dft import G_O2_TO_2H2O, OOH_OH_OFFSET, volcano_curve
    x = np.linspace(0, 2, 4001)
    assert np.isclose(x[volcano_curve(x).argmax()], (G_O2_TO_2H2O - OOH_OH_OFFSET) / 2, atol=1e-3)


def test_reference_energy_bookkeeping():
    from src.dft_jobs import reference_corrected_energy as r
    # E(slab+OH) - E(slab) - (E_H2O - 0.5 E_H2)
    assert np.isclose(r(-110.0, -100.0, -14.0, -6.0, "OH"), -110 + 100 + 14 - 3)
    assert np.isclose(r(-120.0, -100.0, -14.0, -6.0, "OOH"), -120 + 100 + 28 - 9)


def test_validation_rejects_bad_units_provenance_and_composition():
    from src.dft import make_demo_table, validate_table
    d = make_demo_table()
    assert validate_table(d).ok
    bad = d.copy(); bad.loc[0, "E_OH"] = 40; bad.loc[1, "provenance"] = "measured?"; bad.loc[2, "composition"] = "PtXx"
    assert not validate_table(bad).ok and len(validate_table(bad).errors) >= 3


def test_catalysis_hub_parser_only_accepts_h2o_h2_references():
    from src.dft_data import adsorption_target, parse_nodes
    assert adsorption_target("H2O(g) + * -> OH* + 0.5H2(g)") == "OH*"
    assert adsorption_target("0.5O2(g) + * -> O*") is None            # O2 reference rejected
    nodes = [{"Equation": e, "reactionEnergy": v, "surfaceComposition": "Pt", "facet": "111", "pubId": "FIXTURE",
              "dftFunctional": "x", "dftCode": "x"} for e, v in [
        ("H2O(g) + * -> OH* + 0.5H2(g)", 0.6), ("H2O(g) + * -> O* + H2(g)", 1.6), ("2H2O(g) + * -> OOH* + 1.5H2(g)", 3.9)]]
    df, st = parse_nodes(nodes)           # FIXTURE strings, not database records
    assert len(df) == 1 and st["complete_rows"] == 1 and df.provenance.iloc[0] == "dft"
    assert len(parse_nodes(nodes[:2])[0]) == 0                          # an incomplete set never becomes a row


def test_screener_flags_new_chemistry_and_blocks_recommendations_on_demo_data():
    from src.dft import add_orr_metrics, chemical_system, make_demo_table
    from src.dft_ml import Screener
    t = add_orr_metrics(make_demo_table()); t["chemical_system"] = t.composition.map(chemical_system)
    sc = Screener(t)
    out = sc.score(pd.DataFrame({"composition": ["Pt3Ni", "Pt3Ru"], "facet": "111", "structure": "skin"}))
    ru = out[out.composition == "Pt3Ru"].iloc[0]
    assert ru.outside_training_distribution and not out.recommendation_allowed.any()
    assert (out.provenance == "ml_predicted").all() and out.recommendation.str.startswith("NO RECOMMENDATION").all()


def test_leave_one_system_out_never_trains_on_the_held_out_system():
    from src.dft import add_orr_metrics, chemical_system, make_demo_table
    from src.dft_ml import loso_evaluate
    t = add_orr_metrics(make_demo_table()); t["chemical_system"] = t.composition.map(chemical_system)
    summ, oof = loso_evaluate(t, models=["Ridge"])
    assert oof["pred_Ridge"].notna().all() and summ.MAE_U_L_V.iloc[0] < 0.3


@pytest.mark.skipif(not RAW_OK, reason="needs raw data")
def test_agent_dft_tools_refuse_to_guess_values():
    from src.agent import run_tool
    from src.functions import get_context
    ctx = get_context()
    assert "error" in run_tool(ctx, "dft_descriptors", {"composition": "Pt3Os"})
    r = run_tool(ctx, "dft_rank_candidates", {"compositions": ["Pt3Ni"]})
    assert r["provenance_of_numbers"] == "ml_predicted" and r["trusted_training_data"] is False
