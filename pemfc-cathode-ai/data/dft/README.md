# DFT table: schema and rules

The product reads `data/dft/dft_table.csv` if it exists (validated by `src/dft.py::validate_table`);
otherwise it uses a **synthetic demo table** and switches every recommendation off. You can also upload a table
in the app's *DFT discovery* tab.

## Columns
| column | required | meaning |
|---|---|---|
| `system_id` | yes | unique id |
| `composition` | yes | e.g. `Pt`, `Pt3Ni`, `PtCo` (elements known to `src/dft.py::ELEMENTS`) |
| `facet` | yes | `111`, `100` or `110` |
| `E_OH`, `E_O`, `E_OOH` | yes | adsorption energies in eV **referenced to H2O and H2** (OH: H2O - 1/2 H2; O: H2O - H2; OOH: 2 H2O - 3/2 H2) |
| `provenance` | yes | `dft`, `mlip`, `literature_user`, `emt_toy`, `synthetic_demo`, `ml_predicted` |
| `structure` | no | `bulk` or `skin` |
| `d_band_center`, `formation_energy`, `surface_energy`, `segregation_energy`, `dissolution_potential` | no | eV (potential in V) |
| `functional`, `code`, `reference` | no (but cite) | how and from where the value came |

Only `dft`, `mlip` and `literature_user` rows may support a recommendation. Never mix energies computed with
different references or functionals without saying so; include a Pt(111) row computed with the same settings.

## Where DFT values can come from
1. **Your own calculations** with `src/dft_jobs.py` (ASE; bring your calculator).
2. **Catalysis-Hub** (public): `src/dft_data.py::fetch_catalysis_hub` (live query not tested here; check licence and cite the publication).
3. **A CSV copied from a paper** (`literature_user`, with `reference`).

`data/dft/dft_table_template.csv` holds the header only. No DFT value is bundled with this repository.
