# Proteo-Niches Predictor

Research-use Streamlit application for the circulating proteo-niche project.

Current live application:
https://olink-subtype-predictor.streamlit.app/

The application has been updated to the 7:3 model bundle. The Streamlit
subdomain remains the historical name until a new deployment is created from
the renamed GitHub repository.

## Current model

- Main split: 70% training / 30% internal validation.
- Random seed: 42.
- Training participants: 30,765.
- Internal validation participants: 13,185.
- Protein-panel features were selected using the training split only.
- The clinical PN-FIB and PN-INF models were also trained using the same split.

Models are stored in `models/`; panels, metrics and model metadata are stored in
`data/`. The original 8:2 assets are retained under `legacy_8_2/` for audit and
rollback.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the model-bundle checks with:

```powershell
python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Interpretation boundaries

The classifier reproduces labels derived from the same UK Biobank protein
matrix. Its internal AUC measures label reconstruction and is not external
diagnostic or prognostic accuracy. PN-SIG and PN-HOR are small classes whose
class-specific results are exploratory. This application is not a medical
device and must not be used for clinical decisions.

## Related repository

Analysis code:
https://github.com/liang6156/ukb_olink_Proteo-Niches
