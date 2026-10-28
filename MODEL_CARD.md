# Model card

## Intended use

Research-only demonstration of a compact plasma-protein classifier and two
clinical-trait classifiers for internally defined circulating proteo-niches.

## Not intended for

- Diagnosis, prognosis, triage, treatment selection or population screening.
- Assignment of a disease subtype independent of the UK Biobank clustering
  definition.
- Claims of external transportability; no external population validation has
  been performed.

## Primary split

- Random state: 42.
- Training: 70% (n = 30,765).
- Internal validation: 30% (n = 13,185).
- Feature selection and model fitting used training participants only.

## Models

- `proteomic_panel_7_3.pkl`: six-class LightGBM classifier using 30 proteins.
- `clinical_pn_fib_7_3.pkl`: binary PN-FIB clinical-trait classifier.
- `clinical_pn_inf_7_3.pkl`: binary PN-INF clinical-trait classifier.

The exact feature order, categorical levels, metrics and SHA-256 digests are
stored in `data/model_bundle.json`.

## Main limitations

- The target labels were derived by clustering the same underlying plasma
  protein matrix.
- The proteo-niche labels were generated before the training/validation split.
- The high internal AUC indicates reconstruction of the learned labels and is
  not independent diagnostic accuracy.
- The clinical classifiers are low-discrimination exploratory models.
- The cohort is predominantly White European.
