# NWChem Comparison Report

**Date:** 2026-05-12  
**Molecule:** SP-NoInitXYZ-Mol_Methane  
**Reference variant:** nw720_ks (Version of output in manuscript)
**Author:** Ching Ting Leung

## Files Compared

| Variant | N Atoms | N Segments | Surface Area (Å²) | Status |
|---|---|---|---|---|
| nw720_ks | 5 | 289 | 55.3860 | OK |
| nw720_yk | 5 | 289 | 55.3860 | OK |
| nw723_ks | 5 | 289 | 55.3860 | OK |
| nw723_yk | 5 | 289 | 55.3860 | OK |
| nw73_ks | 5 | 289 | 55.3860 | OK |
| nw73_yk | 5 | 289 | 55.3860 | OK |

- **nw720_ks**: `sp-compare/nw720_ks/SP-NoInitXYZ-Mol_Methane/SP-NoInitXYZ-Mol_Methane_0/output.nw`
- **nw720_yk**: `sp-compare/nw720_yk/SP-NoInitXYZ-Mol_Methane/SP-NoInitXYZ-Mol_Methane_0/output.nw`
- **nw723_ks**: `sp-compare/nw723_ks/SP-NoInitXYZ-Mol_Methane/SP-NoInitXYZ-Mol_Methane_0/output.nw`
- **nw723_yk**: `sp-compare/nw723_yk/SP-NoInitXYZ-Mol_Methane/SP-NoInitXYZ-Mol_Methane_0/output.nw`
- **nw73_ks**: `sp-compare/nw73_ks/SP-NoInitXYZ-Mol_Methane/SP-NoInitXYZ-Mol_Methane_0/output.nw`
- **nw73_yk**: `sp-compare/nw73_yk/SP-NoInitXYZ-Mol_Methane/SP-NoInitXYZ-Mol_Methane_0/output.nw`

## Surface Area (Å²)

| Variant | Surface Area (Å²) |
|---|---|
| nw720_ks | 55.386000 |
| nw720_yk | 55.386000 |
| nw723_ks | 55.386000 |
| nw723_yk | 55.386000 |
| nw73_ks | 55.386000 |
| nw73_yk | 55.386000 |

| Comparison | Absolute Diff (Å²) | Relative Diff |
|---|---|---|
| nw720_yk − nw720_ks | +0.000000 | +0.0000% |
| nw723_ks − nw720_ks | +0.000000 | +0.0000% |
| nw723_yk − nw720_ks | +0.000000 | +0.0000% |
| nw73_ks − nw720_ks | +0.000000 | +0.0000% |
| nw73_yk − nw720_ks | +0.000000 | +0.0000% |

## Number of Surface Segments

| Variant | N Segments |
|---|---|
| nw720_ks | 289 |
| nw720_yk | 289 |
| nw723_ks | 289 |
| nw723_yk | 289 |
| nw73_ks | 289 |
| nw73_yk | 289 |

## Segment Areas (a.u.²)

| Variant | Mean | Std | Min | Max |
|---|---|---|---|---|
| nw720_ks | 0.684386 | 0.454716 | 0.000063 | 1.402354 |
| nw720_yk | 0.684386 | 0.454716 | 0.000063 | 1.402354 |
| nw723_ks | 0.684386 | 0.454716 | 0.000063 | 1.402354 |
| nw723_yk | 0.684386 | 0.454716 | 0.000063 | 1.402354 |
| nw73_ks | 0.684386 | 0.454716 | 0.000063 | 1.402354 |
| nw73_yk | 0.684386 | 0.454716 | 0.000063 | 1.402354 |

**Element-wise differences** (same segment count across variants):

| Comparison | Max |Δ| | Mean |Δ| | RMS |
|---|---|---|---|
| nw720_yk − nw720_ks | 0.000000 | 0.000000 | 0.000000 |
| nw723_ks − nw720_ks | 0.000000 | 0.000000 | 0.000000 |
| nw723_yk − nw720_ks | 0.000000 | 0.000000 | 0.000000 |
| nw73_ks − nw720_ks | 0.000000 | 0.000000 | 0.000000 |
| nw73_yk − nw720_ks | 0.000000 | 0.000000 | 0.000000 |

## Segment–Atom Assignments

Number of surface segments assigned to each atom index:

| Variant | Atom 0 | Atom 1 |
|---|---|---|
| nw720_ks | 215 | 74 |
| nw720_yk | 215 | 74 |
| nw723_ks | 215 | 74 |
| nw723_yk | 215 | 74 |
| nw73_ks | 215 | 74 |
| nw73_yk | 215 | 74 |

## Final Atom Coordinates

**Reference:** nw720_ks — 5 atoms: C, H, H, H, H

### nw720_yk vs nw720_ks

- **Max displacement:** 0.000000 Å (atom 1: C)
- **Mean displacement:** 0.000000 Å

| Atom | Element | Δ (Å) | ref (nw720_ks) | variant (nw720_yk) |
|---|---|---|---|---|
| 1 | C | 0.000000 | (-0.0000, -0.0000, 0.0000) | (-0.0000, -0.0000, 0.0000) |
| 2 | H | 0.000000 | (-0.6624, -0.8497, -0.1721) | (-0.6624, -0.8497, -0.1721) |
| 3 | H | 0.000000 | (-0.4472, 0.9012, -0.4222) | (-0.4472, 0.9012, -0.4222) |
| 4 | H | 0.000000 | (0.1469, 0.1352, 1.0727) | (0.1469, 0.1352, 1.0727) |
| 5 | H | 0.000000 | (0.9627, -0.1866, -0.4783) | (0.9627, -0.1866, -0.4783) |

### nw723_ks vs nw720_ks

- **Max displacement:** 0.000000 Å (atom 1: C)
- **Mean displacement:** 0.000000 Å

| Atom | Element | Δ (Å) | ref (nw720_ks) | variant (nw723_ks) |
|---|---|---|---|---|
| 1 | C | 0.000000 | (-0.0000, -0.0000, 0.0000) | (-0.0000, -0.0000, 0.0000) |
| 2 | H | 0.000000 | (-0.6624, -0.8497, -0.1721) | (-0.6624, -0.8497, -0.1721) |
| 3 | H | 0.000000 | (-0.4472, 0.9012, -0.4222) | (-0.4472, 0.9012, -0.4222) |
| 4 | H | 0.000000 | (0.1469, 0.1352, 1.0727) | (0.1469, 0.1352, 1.0727) |
| 5 | H | 0.000000 | (0.9627, -0.1866, -0.4783) | (0.9627, -0.1866, -0.4783) |

### nw723_yk vs nw720_ks

- **Max displacement:** 0.000000 Å (atom 1: C)
- **Mean displacement:** 0.000000 Å

| Atom | Element | Δ (Å) | ref (nw720_ks) | variant (nw723_yk) |
|---|---|---|---|---|
| 1 | C | 0.000000 | (-0.0000, -0.0000, 0.0000) | (-0.0000, -0.0000, 0.0000) |
| 2 | H | 0.000000 | (-0.6624, -0.8497, -0.1721) | (-0.6624, -0.8497, -0.1721) |
| 3 | H | 0.000000 | (-0.4472, 0.9012, -0.4222) | (-0.4472, 0.9012, -0.4222) |
| 4 | H | 0.000000 | (0.1469, 0.1352, 1.0727) | (0.1469, 0.1352, 1.0727) |
| 5 | H | 0.000000 | (0.9627, -0.1866, -0.4783) | (0.9627, -0.1866, -0.4783) |

### nw73_ks vs nw720_ks

- **Max displacement:** 0.000000 Å (atom 1: C)
- **Mean displacement:** 0.000000 Å

| Atom | Element | Δ (Å) | ref (nw720_ks) | variant (nw73_ks) |
|---|---|---|---|---|
| 1 | C | 0.000000 | (-0.0000, -0.0000, 0.0000) | (-0.0000, -0.0000, 0.0000) |
| 2 | H | 0.000000 | (-0.6624, -0.8497, -0.1721) | (-0.6624, -0.8497, -0.1721) |
| 3 | H | 0.000000 | (-0.4472, 0.9012, -0.4222) | (-0.4472, 0.9012, -0.4222) |
| 4 | H | 0.000000 | (0.1469, 0.1352, 1.0727) | (0.1469, 0.1352, 1.0727) |
| 5 | H | 0.000000 | (0.9627, -0.1866, -0.4783) | (0.9627, -0.1866, -0.4783) |

### nw73_yk vs nw720_ks

- **Max displacement:** 0.000000 Å (atom 1: C)
- **Mean displacement:** 0.000000 Å

| Atom | Element | Δ (Å) | ref (nw720_ks) | variant (nw73_yk) |
|---|---|---|---|---|
| 1 | C | 0.000000 | (-0.0000, -0.0000, 0.0000) | (-0.0000, -0.0000, 0.0000) |
| 2 | H | 0.000000 | (-0.6624, -0.8497, -0.1721) | (-0.6624, -0.8497, -0.1721) |
| 3 | H | 0.000000 | (-0.4472, 0.9012, -0.4222) | (-0.4472, 0.9012, -0.4222) |
| 4 | H | 0.000000 | (0.1469, 0.1352, 1.0727) | (0.1469, 0.1352, 1.0727) |
| 5 | H | 0.000000 | (0.9627, -0.1866, -0.4783) | (0.9627, -0.1866, -0.4783) |


## Abbreviations
nw720: NWChem version 7.2.0
nw730: NWChem version 7.2.3
nw73: NWChem version 7.3.0
ks: COSMO KS (Klamt–Schüürmann) Model
yk: COSMO YK (York–Karplus) Model
