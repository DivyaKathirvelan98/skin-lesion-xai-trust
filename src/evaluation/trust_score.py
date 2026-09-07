"""Fuse predictive uncertainty, explanation faithfulness, and cross-method agreement into a
single per-prediction Trust Score: T = w1*(1 - normalized_uncertainty) + w2*faithfulness + w3*agreement.
"""
from dataclasses import dataclass


@dataclass
class TrustScoreWeights:
    uncertainty: float = 0.34
    faithfulness: float = 0.33
    agreement: float = 0.33

    def validate(self):
        total = self.uncertainty + self.faithfulness + self.agreement
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Trust score weights must sum to 1.0, got {total}")


def compute_trust_score(normalized_uncertainty: float, faithfulness: float, agreement: float,
                          weights: TrustScoreWeights = TrustScoreWeights()) -> float:
    """All three inputs must already be normalized/clipped to [0, 1].

    `normalized_uncertainty` should be predictive variance or entropy scaled to [0, 1]
    (e.g., via a fixed max-entropy divisor for the number of classes).
    """
    weights.validate()
    confidence_term = 1.0 - normalized_uncertainty
    return (
        weights.uncertainty * confidence_term
        + weights.faithfulness * faithfulness
        + weights.agreement * agreement
    )
