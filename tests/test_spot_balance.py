import numpy as np

from user_workflows.feedback.spot_balance import run_balance_loop


def test_simulated_imbalanced_spots_converge_toward_uniformity():
    target = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    initial = target.copy()
    imbalance = np.array([0.25, 0.6, 1.4, 2.0], dtype=float)

    def measure(weights):
        return np.maximum(1e-12, weights * imbalance)

    _, history, measured = run_balance_loop(
        initial_weights=initial,
        target_weights=target,
        measure_fn=measure,
        max_iterations=60,
        uniformity_threshold=0.98,
        max_gain_step=1.25,
    )

    assert history[0]["uniformity"] < 0.8
    assert history[-1]["uniformity"] >= 0.98

    measured_norm = measured / measured.sum()
    assert np.max(np.abs(measured_norm - target)) < 0.03
