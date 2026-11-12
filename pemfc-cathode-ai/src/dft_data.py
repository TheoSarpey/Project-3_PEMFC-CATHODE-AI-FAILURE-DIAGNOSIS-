"""Getting DFT adsorption energies into the schema: a public database (Catalysis-Hub) or your own CSV.

STATUS: the parser is tested on hand-written fixture strings only.  The live query could not be run while this
was written (no network), so the GraphQL field names below follow the public Catalysis-Hub API as the author
remembers it and may need small edits.  The parser reports how many records it kept and why it dropped the rest,
and it never invents a value: a row is produced only when OH*, O* and OOH* energies for the same surface, facet,
publication and functional are all present with the H2O/H2 reference.

Licence/citation: check the terms of each database record and cite the publication id kept in `reference`.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

API = "https://api.catalysis-hub.org/graphql"
QUERY = """{ reactions(first: %(n)d, reactants: "%(reactant)s", chemicalComposition: "%(comp)s") {
  edges { node { Equation reactionEnergy surfaceComposition facet sites pubId dftCode dftFunctional } } } }"""

ADSORBATES = {"OH*": "E_OH", "O*": "E_O", "OOH*": "E_OOH"}
ALLOWED_GAS = {"H2O(g)", "H2(g)"}


def _side(text: str) -> dict[str, float]:
    out = {}
    for term in [t.strip() for t in text.split("+") if t.strip()]:
        m = re.match(r"^(\d*\.?\d*(?:/\d+)?)\s*(.+)$", term)
        coef_s, species = (m.group(1), m.group(2)) if m else ("", term)
        if species[0].isdigit():                    # defensive: a species cannot start with a digit here
            coef_s, species = species, ""
        coef = 1.0
        if coef_s:
            coef = float(coef_s.split("/")[0]) / (float(coef_s.split("/")[1]) if "/" in coef_s else 1.0)
        out[species.strip()] = out.get(species.strip(), 0.0) + coef
    return out


def adsorption_target(equation: str) -> str | None:
    """Return 'OH*', 'O*' or 'OOH*' if the equation forms that adsorbate from H2O(g) and H2(g) only.

    With only H2O and H2 allowed, atom balance fixes the stoichiometry, so the reaction energy equals
    E(X*) - E(*) - (a*E_H2O - b*E_H2), exactly the reference used by ``src/dft.py``.  Equations referenced to
    O2 (or anything else) are rejected: they cannot be converted without the (error-prone) O2 energy.
    """
    if "->" not in equation:
        return None
    left, right = (_side(x) for x in equation.split("->", 1))
    species = {k for k in list(left) + list(right) if k not in ("*", "")}
    ads = [s for s in species if s in ADSORBATES and s in right]
    gases = species - set(ads)
    if len(ads) == 1 and gases <= ALLOWED_GAS and ads[0] in right and right[ads[0]] == 1.0 and left.get("*", 0) == 1.0:
        return ads[0]
    return None


def parse_nodes(nodes: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Group records into one row per (surface, facet, publication, functional) with all three energies."""
    best: dict[tuple, dict[str, float]] = defaultdict(dict)
    meta: dict[tuple, dict] = {}
    stats = {"records": len(nodes), "adsorption_records_kept": 0, "rejected_reference_or_format": 0}
    for n in nodes:
        tgt = adsorption_target(n.get("Equation", ""))
        if tgt is None or n.get("reactionEnergy") is None:
            stats["rejected_reference_or_format"] += 1
            continue
        key = (n.get("surfaceComposition"), str(n.get("facet")), n.get("pubId"), n.get("dftFunctional"))
        col = ADSORBATES[tgt]
        e = float(n["reactionEnergy"])
        best[key][col] = min(e, best[key].get(col, e))        # most stable site
        meta[key] = {"code": n.get("dftCode"), "functional": n.get("dftFunctional")}
        stats["adsorption_records_kept"] += 1
    rows = []
    for key, vals in best.items():
        if set(vals) == set(ADSORBATES.values()) and key[0]:
            comp, facet, pub, func = key
            facet_s = re.sub(r"\D", "", facet)[:3]
            rows.append({"system_id": f"CATHUB-{pub}-{comp}-{facet_s}", "composition": comp, "facet": facet_s,
                         **vals, "provenance": "dft", "functional": func, "code": meta[key]["code"],
                         "reference": f"Catalysis-Hub publication id {pub}"})
    stats["complete_rows"] = len(rows)
    return pd.DataFrame(rows), stats


def fetch_catalysis_hub(composition_filter: str = "~Pt", reactant: str = "H2O", n: int = 500, timeout: int = 60):
    """Query the public API (needs network) and return (table, stats). Verify the result before using it."""
    import requests
    r = requests.post(API, json={"query": QUERY % {"n": n, "reactant": reactant, "comp": composition_filter}}, timeout=timeout)
    r.raise_for_status()
    edges = r.json()["data"]["reactions"]["edges"]
    return parse_nodes([e["node"] for e in edges])
