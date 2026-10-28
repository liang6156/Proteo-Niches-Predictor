from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
BUNDLE_PATH = APP_DIR / "data" / "model_bundle.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_row(model, defaults):
    feature_names = list(model.feature_names_in_)
    categorical_indices = model.booster_.params.get("categorical_column") or []
    categorical_levels = getattr(model.booster_, "pandas_categorical", [])
    category_map = {
        feature_names[index]: list(levels)
        for index, levels in zip(categorical_indices, categorical_levels)
    }
    row = {}
    for feature in feature_names:
        value = defaults[feature]["default"]
        if feature in category_map:
            row[feature] = pd.Categorical(
                [str(value)],
                categories=[str(level) for level in category_map[feature]],
            )
        else:
            row[feature] = pd.to_numeric(pd.Series([value]), errors="coerce")
    return pd.DataFrame(row)


class ModelBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        cls.models = {
            "proteomic": joblib.load(
                APP_DIR / cls.bundle["proteomic_model"]["file"]
            ),
            "PN-FIB": joblib.load(
                APP_DIR / cls.bundle["clinical_models"]["PN-FIB"]["file"]
            ),
            "PN-INF": joblib.load(
                APP_DIR / cls.bundle["clinical_models"]["PN-INF"]["file"]
            ),
        }

    def test_split_matches_primary_analysis(self):
        split = self.bundle["split"]
        self.assertEqual(split["train_fraction"], 0.7)
        self.assertEqual(split["validation_fraction"], 0.3)
        self.assertEqual(split["random_state"], 42)
        self.assertEqual(split["n_train"], 30_765)
        self.assertEqual(split["n_validation"], 13_185)

    def test_model_files_match_manifest_hashes(self):
        entries = [self.bundle["proteomic_model"]]
        entries.extend(self.bundle["clinical_models"].values())
        for entry in entries:
            path = APP_DIR / entry["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256(path), entry["sha256"], path.name)

    def test_feature_contracts(self):
        self.assertEqual(
            list(self.models["proteomic"].feature_names_in_),
            self.bundle["proteomic_model"]["features"],
        )
        self.assertEqual(
            list(self.models["proteomic"].classes_),
            self.bundle["proteomic_model"]["classes"],
        )
        for niche in ("PN-FIB", "PN-INF"):
            self.assertEqual(
                list(self.models[niche].feature_names_in_),
                self.bundle["clinical_models"][niche]["features"],
            )

    def test_synthetic_prediction_paths(self):
        proteins = pd.DataFrame(
            np.zeros((1, len(self.bundle["proteomic_model"]["features"]))),
            columns=self.bundle["proteomic_model"]["features"],
        )
        proteomic_probability = self.models["proteomic"].predict_proba(proteins)
        self.assertEqual(proteomic_probability.shape, (1, 6))
        self.assertTrue(np.isfinite(proteomic_probability).all())
        self.assertAlmostEqual(float(proteomic_probability.sum()), 1.0, places=6)

        defaults = self.bundle["clinical_feature_defaults"]
        for niche in ("PN-FIB", "PN-INF"):
            row = training_row(self.models[niche], defaults)
            probability = self.models[niche].predict_proba(row)
            self.assertEqual(probability.shape, (1, 2))
            self.assertTrue(np.isfinite(probability).all())
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
