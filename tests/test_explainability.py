"""Focused test suite for Twinergy 2.0 SHAP explainability service.

Verifies:
1. Local SHAP output exists for an instance
2. SHAP feature names match the 22-feature LightGBM contract
3. Positive and negative contributions are decomposed and sorted correctly
4. No full-dataset SHAP computation occurs (sample bounding safety cap)
5. Existing ModelExplainer service compatibility
"""

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.schema import LIGHTGBM_FEATURES
from src.models.explainability import ModelExplainer, load_explainer


def test_local_shap_output_structure():
    """Verify local SHAP explanation structure and 22-feature contract."""
    explainer = load_explainer()
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=1)

    exp = explainer.explain_instance(sample.iloc[0])

    assert "base_value" in exp
    assert "predicted_log" in exp
    assert "predicted_kwh" in exp
    assert "contributions" in exp
    assert "top_positive_drivers" in exp
    assert "top_negative_drivers" in exp

    # Check feature contract: exactly 22 features
    assert len(exp["contributions"]) == 22
    extracted_feats = [c["feature"] for c in exp["contributions"]]
    assert set(extracted_feats) == set(LIGHTGBM_FEATURES)


def test_positive_negative_contributions_split():
    """Verify positive drivers (>0) and negative drivers (<0) are correctly identified."""
    explainer = load_explainer()
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=5)

    for idx in range(len(sample)):
        exp = explainer.explain_instance(sample.iloc[idx])
        for pos in exp["top_positive_drivers"]:
            assert pos["shap_value"] > 0, "Positive driver must have shap_value > 0"
        for neg in exp["top_negative_drivers"]:
            assert neg["shap_value"] < 0, "Negative driver must have shap_value < 0"


def test_additive_property():
    """Verify SHAP additive property: base_value + sum(shap_values) == predicted_log."""
    explainer = load_explainer()
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=3)

    for idx in range(len(sample)):
        exp = explainer.explain_instance(sample.iloc[idx])
        base_val = exp["base_value"]
        shap_sum = sum(c["shap_value"] for c in exp["contributions"])
        reconstructed_log = base_val + shap_sum
        np.testing.assert_allclose(
            reconstructed_log,
            exp["predicted_log"],
            rtol=1e-4,
            err_msg="Additive SHAP property violated",
        )


def test_sample_bounding_safety_cap():
    """Verify explain_sample caps samples to prevent full-dataset execution stalls."""
    explainer = load_explainer()
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=15)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = explainer.explain_sample(sample, max_samples=5)
        assert res["sample_size"] == 5, "Must cap evaluated sample size to 5"
        assert any("Sampling down" in str(item.message) for item in w)


def test_global_tree_importance_compatibility():
    """Verify global tree gain importance remains available alongside local SHAP."""
    explainer = load_explainer()
    df_gain = explainer.get_tree_feature_importance(importance_type="gain")

    assert len(df_gain) == 22
    assert "feature" in df_gain.columns
    assert "importance" in df_gain.columns
    assert df_gain["importance_type"].iloc[0] == "gain"


if __name__ == "__main__":
    print("Running test_local_shap_output_structure...")
    test_local_shap_output_structure()
    print("✅ test_local_shap_output_structure passed.")

    print("Running test_positive_negative_contributions_split...")
    test_positive_negative_contributions_split()
    print("✅ test_positive_negative_contributions_split passed.")

    print("Running test_additive_property...")
    test_additive_property()
    print("✅ test_additive_property passed.")

    print("Running test_sample_bounding_safety_cap...")
    test_sample_bounding_safety_cap()
    print("✅ test_sample_bounding_safety_cap passed.")

    print("Running test_global_tree_importance_compatibility...")
    test_global_tree_importance_compatibility()
    print("✅ test_global_tree_importance_compatibility passed.")

    print("\n🎉 ALL SHAP EXPLAINABILITY TESTS PASSED!")
