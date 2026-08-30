from ml_engine.diagnostics import diagnose_fit


def test_overfitting_detected_for_bounded_metric():
    result = diagnose_fit(train_score=0.99, validation_score=0.71, metric_name="f1")
    assert result["diagnostic"] == "possible_overfitting"
    assert "0.9900" in result["evidence"]
    assert "0.7100" in result["evidence"]


def test_underfitting_detected_when_both_scores_poor():
    result = diagnose_fit(train_score=0.5, validation_score=0.48, metric_name="accuracy")
    assert result["diagnostic"] == "possible_underfitting"


def test_no_strong_signal_when_scores_are_close_and_good():
    result = diagnose_fit(train_score=0.85, validation_score=0.83, metric_name="f1")
    assert result["diagnostic"] == "no_strong_signal"


def test_overfitting_detected_for_unbounded_metric_rmse():
    # train RMSE much lower (better) than validation RMSE
    result = diagnose_fit(train_score=1.0, validation_score=15.0, metric_name="rmse")
    assert result["diagnostic"] == "possible_overfitting"


def test_evidence_always_includes_both_raw_scores():
    result = diagnose_fit(train_score=0.7, validation_score=0.65, metric_name="r2")
    assert result["train_score"] == 0.7
    assert result["validation_score"] == 0.65
    assert result["possible_action"]
