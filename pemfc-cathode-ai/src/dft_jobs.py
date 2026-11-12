"""Small, calculator-agnostic adsorption-energy workflow with ASE.

STATUS: **not executed** while this project was assembled: ASE was not installed in the build environment and
no DFT code was available.  The functions follow the ASE API as documented, and the energy bookkeeping (the part
most prone to mistakes) lives in ``reference_corrected_energy`` which *is* unit-tested.  Run
``python -m src.dft_jobs --selftest`` on your machine first: it relaxes a toy Pt(111)+O system with ASE's EMT
potential and checks that the plumbing produces a table row.

Calculators and what they mean for ``provenance``
    "emt"      ASE EMT toy potential: tests the workflow only; values are NOT chemically meaningful -> emt_toy
    "mace"     MACE-MP foundation potential (pip install mace-torch)                              -> mlip
    "chgnet"   CHGNet universal potential (pip install chgnet)                                    -> mlip
    "custom"   any ASE calculator you build yourself (GPAW, Quantum ESPRESSO, VASP via ASE)        -> dft
Machine-learned potentials are fast and approximate; confirm key candidates with DFT.

Reference scheme (same as src/dft.py):  E_ads(X) = E(slab+X) - E(slab) - (a*E(H2O) - b*E(H2)) with
    OH : a=1, b=0.5     O : a=1, b=1     OOH : a=2, b=1.5
"""
from __future__ import annotations

import argparse
from typing import Callable

import pandas as pd

from .dft import ELEMENTS, parse_composition

REFS = {"OH": (1.0, 0.5), "O": (1.0, 1.0), "OOH": (2.0, 1.5)}
PROV = {"emt": "emt_toy", "mace": "mlip", "chgnet": "mlip", "custom": "dft"}


def reference_corrected_energy(e_slab_ads: float, e_slab: float, e_h2o: float, e_h2: float, adsorbate: str) -> float:
    """Adsorption energy (eV) relative to H2O and H2 (unit-tested; no ASE needed)."""
    a, b = REFS[adsorbate]
    return e_slab_ads - e_slab - (a * e_h2o - b * e_h2)


def get_calculator(name: str, factory: Callable | None = None):
    """Return a fresh ASE calculator. `factory` is required for name='custom' (your DFT setup)."""
    if name == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", default_dtype="float64")
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator()
    if name == "custom" and factory is not None:
        return factory()
    raise ValueError("unknown calculator, or 'custom' without a factory")


def make_slab(composition: str, facet: str = "111", structure: str = "bulk", size=(2, 2, 4), vacuum: float = 10.0):
    """Starting geometry only: an fcc slab of Pt with other elements substituted in a fixed pattern.

    bulk    : minority atoms replace Pt in every layer in proportion to the formula
    skin    : top layer stays pure Pt, the minority atoms go into layers 2-4 (a Pt skin)
    Relax it and look at it: this is a crude model of an ordered alloy, not a thermodynamic ground state.
    """
    from ase.build import fcc100, fcc110, fcc111
    comp = parse_composition(composition)
    host = "Pt" if "Pt" in comp else max(comp, key=comp.get)
    builder = {"111": fcc111, "100": fcc100, "110": fcc110}[facet]
    a = sum(ELEMENTS[e]["a"] * f for e, f in comp.items())            # Vegard's-law lattice constant
    slab = builder(host, size=size, a=a, vacuum=vacuum, orthogonal=False if facet == "111" else True)
    others = [(e, f) for e, f in comp.items() if e != host]
    if others:
        z = slab.positions[:, 2]
        layers = sorted(set(z.round(2)))
        eligible = [i for i, zi in enumerate(z.round(2)) if (structure != "skin" or zi != layers[-1])]
        n_swap = round(sum(f for _, f in others) * len(slab))
        step = max(1, len(eligible) // max(1, n_swap))
        chosen = eligible[::step][:n_swap]
        symbols = slab.get_chemical_symbols()
        k = 0
        for e, f in others:
            for _ in range(round(f * len(slab))):
                if k < len(chosen):
                    symbols[chosen[k]] = e
                    k += 1
        slab.set_chemical_symbols(symbols)
    return slab


def add_adsorbate_to(slab, adsorbate: str, site: str = "ontop", height: float = 1.8):
    from ase import Atoms
    from ase.build import add_adsorbate
    mol = {"O": Atoms("O"), "OH": Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)]),
           "OOH": Atoms("OOH", positions=[(0, 0, 0), (0, 0, 1.45), (0.9, 0, 1.9)])}[adsorbate]
    out = slab.copy()
    add_adsorbate(out, mol, height=height, position=site)
    return out


def relax(atoms, calc, fmax: float = 0.05, steps: int = 200, fix_bottom_layers: int = 2):
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    atoms = atoms.copy()
    z = atoms.positions[:, 2].round(2)
    layers = sorted(set(z))[:fix_bottom_layers]
    atoms.set_constraint(FixAtoms(indices=[i for i, zi in enumerate(z) if zi in layers]))
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms, float(atoms.get_potential_energy())


def molecule_energy(name: str, calc) -> float:
    from ase.build import molecule
    m = molecule(name)
    m.center(vacuum=8.0)
    from ase.optimize import BFGS
    m.calc = calc
    BFGS(m, logfile=None).run(fmax=0.05, steps=100)
    return float(m.get_potential_energy())


def adsorption_row(composition: str, facet: str, structure: str, calculator: str, factory=None,
                   sites=("ontop", "fcc", "hcp", "bridge"), reference: str = "") -> dict:
    """Relax the clean slab and OH/O/OOH on each site, keep the most stable site, return one table row."""
    e_h2o = molecule_energy("H2O", get_calculator(calculator, factory))
    e_h2 = molecule_energy("H2", get_calculator(calculator, factory))
    slab = make_slab(composition, facet, structure)
    slab_relaxed, e_slab = relax(slab, get_calculator(calculator, factory))
    row = {"system_id": f"{calculator.upper()}-{composition}-{facet}-{structure}", "composition": composition,
           "facet": facet, "structure": structure, "provenance": PROV[calculator], "code": f"ASE/{calculator}",
           "reference": reference or f"calculated here with {calculator}"}
    for ads in ("OH", "O", "OOH"):
        energies = []
        for site in sites:
            try:
                _, e = relax(add_adsorbate_to(slab_relaxed, ads, site), get_calculator(calculator, factory))
                energies.append(reference_corrected_energy(e, e_slab, e_h2o, e_h2, ads))
            except Exception:                              # a site that does not converge is skipped, not guessed
                continue
        if not energies:
            raise RuntimeError(f"no converged site for {ads} on {composition}({facet})")
        row[f"E_{ads}"] = min(energies)
    return row


def run_candidates(candidates: pd.DataFrame, calculator: str = "emt", factory=None) -> pd.DataFrame:
    return pd.DataFrame([adsorption_row(r.composition, r.facet, getattr(r, "structure", "bulk"), calculator, factory)
                         for r in candidates.itertuples()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        out = run_candidates(pd.DataFrame({"composition": ["Pt"], "facet": ["111"], "structure": ["bulk"]}), "emt")
        print(out.round(3).to_string(index=False))
        print("EMT rows are tagged emt_toy: they test the workflow, they are not adsorption energies.")
