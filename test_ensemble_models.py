from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ensemble_models import SELECTED_MODEL, make_model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput, EnsembleResult
from run_experiments import METRICS, prepare_data, run

ROOT = Path(__file__).resolve().parent


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(160, 4))
        y = (x[:, 0] + x[:, 1] * 0.4 > 0).astype(int)
        cls.data = EnsembleInput(x[:120], y[:120], x[120:], y[120:], random_seed=42)

    def test_contract_shapes_and_probabilities(self):
        result = run_ensemble_experiment(self.data)
        self.assertIsInstance(result, EnsembleResult)
        self.assertIn("RandomForestClassifier", result.model_name)
        self.assertEqual(result.probabilities.shape, (40,))
        self.assertEqual(result.predictions.shape, (40,))
        self.assertTrue(np.isfinite(result.probabilities).all())
        self.assertTrue(((result.probabilities >= 0) & (result.probabilities <= 1)).all())
        np.testing.assert_array_equal(result.predictions, (result.probabilities >= 0.5).astype(int))
        self.assertGreater(result.roc_auc, 0.9)

    def test_metrics_match_independent_calculation(self):
        result = run_ensemble_experiment(self.data)
        self.assertAlmostEqual(result.roc_auc, roc_auc_score(self.data.test_target, result.probabilities))
        for name, function in [("accuracy", accuracy_score), ("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
            self.assertAlmostEqual(getattr(result, name), function(self.data.test_target, result.predictions))

    def test_seed_is_reproducible(self):
        a, b = run_ensemble_experiment(self.data), run_ensemble_experiment(self.data)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        np.testing.assert_array_equal(a.predictions, b.predictions)

    def test_seed_reaches_estimator(self):
        model = make_model(SELECTED_MODEL, 123)
        self.assertEqual(model.steps[-1][1].random_state, 123)

    def test_test_labels_never_affect_fit(self):
        a = run_ensemble_experiment(self.data)
        b = run_ensemble_experiment(replace(self.data, test_target=1-self.data.test_target))
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        np.testing.assert_array_equal(a.predictions, b.predictions)
        self.assertAlmostEqual(a.roc_auc + b.roc_auc, 1)

    def test_imputer_uses_training_values_only(self):
        train = self.data.train_features.copy()
        train[0, 0] = np.nan
        test = np.full_like(self.data.test_features, 1e9)
        model = make_model(SELECTED_MODEL, 42)
        with patch("ensemble_models.make_model", return_value=model):
            run_ensemble_experiment(replace(self.data, train_features=train, test_features=test))
        np.testing.assert_allclose(model.steps[0][1].statistics_, np.nanmedian(train, axis=0))

    def test_missing_and_empty_column(self):
        train, test = self.data.train_features.copy(), self.data.test_features.copy()
        train[:, 3] = np.nan
        test[:, 3] = np.nan
        test[0, 0] = np.nan
        result = run_ensemble_experiment(replace(self.data, train_features=train, test_features=test))
        self.assertTrue(np.isfinite(result.probabilities).all())

    def test_inputs_are_not_modified(self):
        before = self.data.train_features.copy()
        run_ensemble_experiment(self.data)
        np.testing.assert_array_equal(before, self.data.train_features)

    def test_dataframe_lists_and_column_targets(self):
        normal = run_ensemble_experiment(self.data)
        converted = replace(self.data, train_features=pd.DataFrame(self.data.train_features),
                            test_features=self.data.test_features.tolist(),
                            train_target=self.data.train_target.reshape(-1, 1),
                            test_target=self.data.test_target.reshape(-1, 1))
        np.testing.assert_array_equal(run_ensemble_experiment(converted).probabilities, normal.probabilities)

    def test_small_dataset(self):
        result = run_ensemble_experiment(EnsembleInput([[0], [1]], [0, 1], [[0], [1]], [0, 1]))
        self.assertEqual(result.probabilities.shape, (2,))
        self.assertTrue(0 <= result.roc_auc <= 1)

    def test_invalid_shapes(self):
        for change in ({"train_features": []}, {"test_features": [[1]]},
                       {"train_target": [0, 1]}, {"test_target": [0, 1]},
                       {"test_target": np.zeros((40, 2))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_ensemble_experiment(replace(self.data, **change))

    def test_invalid_labels(self):
        for change in ({"train_target": np.full(120, 2)}, {"test_target": np.full(40, 2)},
                       {"test_target": np.ones(40)}, {"train_target": np.ones(120)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_ensemble_experiment(replace(self.data, **change))

    def test_infinity_is_rejected(self):
        x = self.data.train_features.copy()
        x[0, 0] = np.inf
        with self.assertRaises(ValueError):
            run_ensemble_experiment(replace(self.data, train_features=x))


class ExperimentTests(unittest.TestCase):
    def test_split_matches_lab3(self):
        data, frame = prepare_data()
        self.assertEqual(frame.shape, (1000, 21))
        self.assertEqual(data.train_features.shape, (800, 20))
        self.assertEqual(data.test_features.shape, (200, 20))
        np.testing.assert_array_equal(np.bincount(data.test_target), [60, 140])

    def test_contract_reproduces_saved_metrics(self):
        data, _ = prepare_data()
        expected = json.loads((ROOT / "experiment_results.json").read_text())
        result = run_ensemble_experiment(data)
        for metric in METRICS:
            self.assertAlmostEqual(getattr(result, metric), expected[metric], delta=1e-6)
        np.testing.assert_allclose(result.probabilities, expected["probabilities"], atol=1e-12)

    def test_full_cv_experiment_and_figures(self):
        expected = json.loads((ROOT / "experiment_results.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            actual = run(directory)
            self.assertEqual(actual["selected_model"], SELECTED_MODEL)
            best = max(actual["experiments"], key=lambda row: row["cv_mean"]["roc_auc"])
            self.assertEqual(best["model"], SELECTED_MODEL)
            for metric in METRICS:
                self.assertAlmostEqual(actual[metric], expected[metric], delta=1e-6)
            for name in ("model_comparison.png", "confusion_matrix.png", "REPORT.md"):
                self.assertGreater((Path(directory) / name).stat().st_size, 1000)

    def test_visual_output(self):
        import matplotlib.pyplot as plt
        for name in ("model_comparison.png", "confusion_matrix.png"):
            image = plt.imread(ROOT / name)
            self.assertEqual(image.ndim, 3)
            figure, axis = plt.subplots(figsize=(12, 5))
            axis.imshow(image)
            axis.axis("off")
            figure.tight_layout()
        if plt.get_backend().lower() != "agg":
            plt.show()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
