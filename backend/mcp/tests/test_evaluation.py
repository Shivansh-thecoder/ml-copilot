import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from ml_engine.evaluation import compute_classification_metrics
from ml_engine.training import _sklearn_scoring_for
from ml_engine.tuning import _single_scorer_for


@pytest.mark.parametrize('labels', [[0, 1], ['no', 'yes'], [2, 7], ['a', 'b', 'c']])
def test_cv_tuning_and_evaluation_share_macro_f1(labels):
    y = np.array([labels[0]] * 8 + labels[1:] * 2)
    X = np.arange(len(y)).reshape(-1, 1)
    model = DummyClassifier(strategy='most_frequent').fit(X, y)
    predicted = model.predict(X)
    expected = f1_score(y, predicted, average='macro', zero_division=0)
    assert expected != pytest.approx(f1_score(y, predicted, average='weighted', zero_division=0))
    assert _sklearn_scoring_for('classification', len(labels))['f1'](model, X, y) == pytest.approx(expected)
    assert _single_scorer_for('f1')(model, X, y) == pytest.approx(expected)
    assert compute_classification_metrics(y, predicted)['f1'] == pytest.approx(expected)


def test_string_probability_metrics_follow_estimator_class_order():
    result = compute_classification_metrics(
        ['yes', 'no', 'yes', 'no'], ['yes', 'no', 'yes', 'no'],
        [[.9, .1], [.1, .9], [.8, .2], [.2, .8]], classes=['yes', 'no'],
    )
    assert result['roc_auc'] == 1.0
    assert result['pr_auc'] == 1.0


def test_single_class_holdout_omits_undefined_auc():
    result = compute_classification_metrics(['no', 'no'], ['no', 'no'], [[.9, .1], [.8, .2]], classes=['no', 'yes'])
    assert 'roc_auc' not in result and 'pr_auc' not in result
