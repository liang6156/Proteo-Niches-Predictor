"""Proteo-Niches Predictor research-use web application.

The deployed models were trained on a fixed 70/30 stratified split
(random_state=42). The application does not diagnose disease and does not
replace clinical assessment.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "data" / "model_bundle.json"

st.set_page_config(
    page_title="Proteo-Niches Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_bundle():
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    models = {
        "proteomic": joblib.load(
            APP_DIR / bundle["proteomic_model"]["file"]
        ),
        "PN-FIB": joblib.load(
            APP_DIR / bundle["clinical_models"]["PN-FIB"]["file"]
        ),
        "PN-INF": joblib.load(
            APP_DIR / bundle["clinical_models"]["PN-INF"]["file"]
        ),
    }
    return bundle, models


def categorical_map(model):
    """Return feature-name to training-category mapping for one LightGBM model."""
    feature_names = list(model.feature_names_in_)
    indices = model.booster_.params.get("categorical_column", [])
    levels = getattr(model.booster_, "pandas_categorical", [])
    return {
        feature_names[position]: list(category_levels)
        for position, category_levels in zip(indices, levels)
    }


def make_model_row(values, feature_names, category_map):
    row = {}
    for feature in feature_names:
        value = values.get(feature)
        if feature in category_map:
            row[feature] = pd.Categorical(
                [str(value)],
                categories=[str(level) for level in category_map[feature]],
            )
        else:
            row[feature] = pd.to_numeric(
                pd.Series([value]), errors="coerce"
            )
    return pd.DataFrame(row)


def probability_table(classes, probabilities, niche_names):
    rows = [
        {
            "Class": class_code,
            "Proteo-niche": niche_names.get(class_code, class_code),
            "Probability": float(probability),
        }
        for class_code, probability in zip(classes, probabilities)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )


def render_research_notice():
    st.warning(
        "Research use only. This application demonstrates transfer of "
        "internally defined proteo-niche labels and is not a diagnostic or "
        "clinical decision-support device."
    )


bundle, models = load_bundle()
niche_names = bundle["niche_names"]
proteomic_features = bundle["proteomic_model"]["features"]
clinical_features = bundle["clinical_models"]["PN-FIB"]["features"]
clinical_defaults = bundle["clinical_feature_defaults"]
clinical_categories = categorical_map(models["PN-FIB"])

st.sidebar.title("Proteo-Niches")
st.sidebar.caption("7:3 training-only model bundle · 2026-10-28")
st.sidebar.markdown(
    f"Training set: **{bundle['split']['n_train']:,}**  \n"
    f"Internal validation set: **{bundle['split']['n_validation']:,}**  \n"
    "Split seed: **42**"
)
st.sidebar.divider()
mode = st.sidebar.radio(
    "Analysis mode",
    [
        "Overview",
        "Channel 1 · Clinical traits",
        "Channel 2 · 30-protein panel",
    ],
)

if mode == "Overview":
    st.title("Proteo-Niches Predictor")
    st.subheader("A research tool for plasma-proteome state classification")
    render_research_notice()

    st.markdown(
        "The application uses LightGBM models trained in **70%** of the "
        "43,950 UK Biobank participants and evaluated in the held-out **30%**. "
        "Protein-panel features were selected using the training split only."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Training", f"{bundle['split']['n_train']:,}")
    metric_cols[1].metric("Validation", f"{bundle['split']['n_validation']:,}")
    metric_cols[2].metric(
        "Macro AUC",
        f"{bundle['proteomic_model']['summary_metrics']['AUC_macro']:.3f}",
    )
    metric_cols[3].metric(
        "Accuracy",
        f"{bundle['proteomic_model']['summary_metrics']['Accuracy']:.3f}",
    )

    image_path = APP_DIR / "2900_annotated_umap_final.png"
    if image_path.exists():
        st.image(
            str(image_path),
            caption="UMAP visualization of the data-defined proteo-niches",
            use_container_width=True,
        )

    st.subheader("Interpretation boundaries")
    st.markdown(
        "- The six labels are derived from a clustering solution, not from an "
        "external diagnostic gold standard.\n"
        "- The high internal AUC measures label reconstruction within the same "
        "cohort and protein-panel context.\n"
        "- PN-SIG and PN-HOR are much smaller classes; class-specific results "
        "are exploratory.\n"
        "- Independent population validation, calibration and longitudinal "
        "testing are required before any clinical claim."
    )

    with st.expander("Model and data metadata"):
        st.json(
            {
                "model_version": bundle["model_version"],
                "split": bundle["split"],
                "proteomic_panel": bundle["proteomic_panel"],
                "clinical_panel_performance": bundle["clinical_performance"],
                "class_performance": bundle["proteomic_model"]["class_metrics"],
            }
        )

elif mode == "Channel 1 · Clinical traits":
    st.title("Channel 1 · Clinical-trait screening")
    render_research_notice()
    st.markdown(
        "The 20 clinical features were selected and the PN-FIB and PN-INF "
        "binary classifiers were trained using the 70% training split only. "
        "Inputs are prefilled with training-set medians or the most frequent "
        "training category; replace them with the participant's values."
    )

    values = {}
    left, right = st.columns([1.35, 1])
    with left:
        for feature in clinical_features:
            spec = clinical_defaults[feature]
            label = spec["phenotype"]
            if feature in clinical_categories:
                options = [str(item) for item in clinical_categories[feature]]
                default = str(spec["default"])
                index = options.index(default) if default in options else 0
                values[feature] = st.selectbox(label, options, index=index)
            else:
                minimum = float(spec["min"])
                maximum = float(spec["max"])
                default = float(spec["default"])
                values[feature] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    format="%.3f",
                )

    with right:
        st.subheader("Predicted probabilities")
        if st.button("Run clinical-trait model", use_container_width=True):
            rows = make_model_row(
                values,
                clinical_features,
                clinical_categories,
            )
            results = []
            for niche in ("PN-FIB", "PN-INF"):
                probability = float(
                    models[niche].predict_proba(rows)[0, 1]
                )
                results.append(
                    {
                        "Target niche": niche,
                        "Probability": probability,
                    }
                )
            result_df = pd.DataFrame(results).sort_values(
                "Probability", ascending=False
            )
            st.dataframe(
                result_df.style.format({"Probability": "{:.3f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "These probabilities do not include a validated clinical "
                "decision threshold. Treat them as exploratory model outputs."
            )

else:
    st.title("Channel 2 · 30-protein panel")
    render_research_notice()
    st.markdown(
        "Upload a CSV with the exact Olink protein columns listed below. "
        "The frozen 70/30 LightGBM model returns probabilities for all six "
        "internally defined labels."
    )

    template = pd.DataFrame(columns=proteomic_features)
    st.download_button(
        "Download input template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="proteo_niches_input_template.csv",
        mime="text/csv",
    )
    with st.expander("Required protein columns", expanded=False):
        st.code(", ".join(proteomic_features))

    uploaded = st.file_uploader("Upload proteomic data", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [
            feature for feature in proteomic_features
            if feature not in data.columns
        ]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            values = (
                data[proteomic_features]
                .iloc[[0]]
                .apply(pd.to_numeric, errors="coerce")
            )
            if values.isna().any(axis=None):
                st.error(
                    "The uploaded row contains missing or non-numeric values "
                    "in required protein columns."
                )
            else:
                model = models["proteomic"]
                probabilities = model.predict_proba(values)[0]
                table = probability_table(
                    list(model.classes_),
                    probabilities,
                    niche_names,
                )
                predicted_class = str(model.predict(values)[0])
                predicted = niche_names.get(
                    predicted_class, predicted_class
                )
                probability = float(table.iloc[0]["Probability"])

                st.success("Classification complete")
                st.metric(
                    "Most probable proteo-niche",
                    predicted,
                    f"{probability:.1%}",
                )
                st.dataframe(
                    table.style.format({"Probability": "{:.3f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
                if probability < 0.50:
                    st.warning(
                        "The maximum class probability is below 0.50. "
                        "Interpret this assignment with particular caution."
                    )

                st.subheader("Important interpretation limit")
                st.markdown(
                    "The classifier was trained to reproduce labels derived "
                    "from the same UK Biobank protein matrix. A high probability "
                    "does not establish an independently validated subtype, "
                    "disease diagnosis, prognosis or treatment indication."
                )
