import numpy as np

from src.evaluation.faithfulness import pointing_game_hit, saliency_vs_mask_iou
from src.evaluation.trust_score import TrustScoreWeights, compute_trust_score
from src.xai.agreement import cross_method_agreement, iou


def test_iou_identical_maps_is_one():
    saliency = np.array([[0.9, 0.1], [0.1, 0.9]])
    mask = np.array([[1, 0], [0, 1]])
    assert saliency_vs_mask_iou(saliency, mask, threshold=0.5) == 1.0


def test_iou_disjoint_maps_is_zero():
    saliency = np.array([[0.9, 0.1], [0.1, 0.1]])
    mask = np.array([[0, 0], [0, 1]])
    assert saliency_vs_mask_iou(saliency, mask, threshold=0.5) == 0.0


def test_pointing_game_hit_true_when_peak_inside_mask():
    saliency = np.array([[0.1, 0.9], [0.1, 0.1]])
    mask = np.array([[0, 1], [0, 0]])
    assert pointing_game_hit(saliency, mask) is True


def test_pointing_game_hit_false_when_peak_outside_mask():
    saliency = np.array([[0.9, 0.1], [0.1, 0.1]])
    mask = np.array([[0, 1], [0, 0]])
    assert pointing_game_hit(saliency, mask) is False


def test_cross_method_agreement_identical_maps():
    maps = {"a": np.array([[1.0, 0.0], [0.0, 1.0]]), "b": np.array([[1.0, 0.0], [0.0, 1.0]])}
    result = cross_method_agreement(maps, threshold=0.5)
    assert result["mean_iou"] == 1.0


def test_trust_score_bounds():
    weights = TrustScoreWeights(uncertainty=0.34, faithfulness=0.33, agreement=0.33)
    best = compute_trust_score(0.0, 1.0, 1.0, weights)
    worst = compute_trust_score(1.0, 0.0, 0.0, weights)
    assert best == 1.0
    assert worst == 0.0


def test_trust_score_weights_must_sum_to_one():
    weights = TrustScoreWeights(uncertainty=0.5, faithfulness=0.5, agreement=0.5)
    try:
        compute_trust_score(0.1, 0.5, 0.5, weights)
        assert False, "expected ValueError"
    except ValueError:
        pass
