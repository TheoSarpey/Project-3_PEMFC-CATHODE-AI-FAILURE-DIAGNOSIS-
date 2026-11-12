# Troubleshooting report: curve 254

**Test family:** Standard Test of Nafion Membrane 112  
**Conditions:** pressure 5.0, RH 80.0, compression 5.0, Nafion 20.0, activation set None

## Summary
The activation/kinetic zone at low current appears weak, with a kinetic loss of 0.3797 V and a Tafel slope of 134.0767 mV/dec. A gas-crossover measurement would help check the crossover possibility, alongside the suspected cathode catalyst-layer ORR activity. As a cautious next step, obtain that measurement before deciding on any intervention.

## Evidence
* Severity (z against peers): 2.35
* Loss budget at j = 738 mA/cm2 (V): kinetic 0.380, ohmic 0.387, mass transport -0.001
* Peak power 305.532 mW/cm2; apparent resistance 0.5248 ohm cm2

## Parameters to check
catalyst loading, Pt utilisation / ECSA, ionomer content, humidity, pressure

## Troubleshooting notes (sources)
* **[K01] ORR kinetics at the PEMFC cathode** (textbook): The oxygen reduction reaction (ORR) at the cathode is far slower than hydrogen oxidation at the anode, so the cathode causes most of the activation (kinetic) voltage loss in a PEM fuel cell.
* **[K13] Low open-circuit voltage: crossover, short, contamination** (textbook): In H2/O2 an open-circuit voltage close to 1 V is typical for a healthy Pt/C cell; a clearly lower value suggests hydrogen crossover through a thin or damaged membrane, an internal electrical short, a leak, or a poisoned or inactive cathode.
* **[K24] Symptom-to-cause guide for cathode failures** (textbook): Low open-circuit voltage: check crossover, internal short, leaks and contamination.

## Possible improvements
* pressure: 5 -> 25: predicted +120 mW/cm2 (surrogate, interpolation within measured design)
* membrane_compression: 5 -> 12: predicted +49 mW/cm2 (surrogate, interpolation within measured design)
* pressure: 5 -> 15: predicted +47 mW/cm2 (surrogate, interpolation within measured design)

## Confirm before acting
A polarisation curve narrows the cause but does not prove it. Confirm with EIS (HFR and arc), a crossover measurement, CV/ECSA, or a repeat run. Data: Pt/C cathodes in oxygen only (see note K26).
