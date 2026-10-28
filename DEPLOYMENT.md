# Streamlit deployment

The existing application at
`https://olink-subtype-predictor.streamlit.app/` has been updated and currently
loads the 7:3 training-only model bundle. Its subdomain is independent of the
GitHub repository name.

To publish the renamed endpoint
`https://proteo-niches-predictor.streamlit.app/`:

1. Sign in to Streamlit Community Cloud with the `liang6156` GitHub account.
2. Create an application from
   `https://github.com/liang6156/Proteo-Niches-Predictor`.
3. Use branch `main`, entrypoint `app.py`, and Python 3.11 or newer.
4. Set the custom subdomain to `proteo-niches-predictor`.
5. Open the new URL and verify that the sidebar reports training n = 30,765,
   validation n = 13,185, seed 42, and the 7:3 model bundle.
6. Only after the new URL returns HTTP 200 should
   `https://proteo-niches-predictor.streamlit.app/` replace the historical URL
   in the manuscript.

No repository secret is needed by the application. It reads only the bundled
models in `models/`, metadata in `data/`, and the UMAP image at the repository
root.
