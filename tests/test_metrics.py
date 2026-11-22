"""Metric sanity checks on synthetic score distributions with known answers."""
import numpy as np

from common.metrics import auc, binary_metrics, cmc, eer, open_set_dir, tar_at_far, threshold_at_far


def test_perfect_separation():
    y = np.r_[np.ones(100), np.zeros(100)]
    s = np.r_[np.linspace(0.6, 1, 100), np.linspace(-0.2, 0.4, 100)]
    assert auc(y, s) == 1.0
    assert eer(y, s)[0] < 1e-6
    assert tar_at_far(y, s, 1e-3)[0] == 1.0


def test_random_scores_auc_half():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20000), np.zeros(20000)]
    s = rng.normal(size=40000)
    assert abs(auc(y, s) - 0.5) < 0.02
    assert abs(eer(y, s)[0] - 0.5) < 0.02


def test_gaussian_eer_matches_theory():
    # genuine ~ N(1,1), impostor ~ N(-1,1): EER = Phi(-1) = 0.1587, at threshold 0
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(200000), np.zeros(200000)]
    s = np.r_[rng.normal(1, 1, 200000), rng.normal(-1, 1, 200000)]
    e, t = eer(y, s)
    assert abs(e - 0.1587) < 0.005 and abs(t) < 0.02


def test_threshold_at_far_is_conservative():
    rng = np.random.default_rng(2)
    neg = rng.normal(size=10000)
    for far in (0.1, 0.01, 0.001):
        t = threshold_at_far(neg, far)
        assert (neg >= t).mean() <= far


def test_binary_metrics_counts():
    y = np.array([1, 1, 0, 0]); s = np.array([0.9, 0.2, 0.8, 0.1])
    m = binary_metrics(y, s, 0.5)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_cmc_and_open_set():
    g_lab = np.array(["a", "b", "c"])
    sim = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.8]])        # probe1 correct at rank1, probe2 (true b) at rank2
    curve, ranks, _ = cmc(sim, g_lab, np.array(["a", "b"]), max_rank=3)
    assert list(ranks) == [1, 2] and curve[0] == 0.5 and curve[1] == 1.0
    d, _ = open_set_dir(sim, np.array(["a", "b"]), np.array([[0.5, 0.4, 0.2]]), g_lab, far=0.0)
    assert d == 0.5


def test_head_pose_cues_are_directional():
    """Guided enrolment relies on these signs: nose to image-right => yaw > 0; nose toward the eyes => pitch falls."""
    import numpy as np
    from common.align import head_pose
    # frontal: eyes (30,40)/(70,40), nose (50,60), mouth (35,80)/(65,80)
    front = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], float)
    f = head_pose(front)
    assert abs(f["yaw"]) < 1e-6 and abs(f["roll_deg"]) < 1e-6
    turned = front.copy(); turned[2, 0] += 8                   # nose shifts to image-right (person turned to THEIR left)
    assert head_pose(turned)["yaw"] > f["yaw"] + 0.15
    mirrored = front.copy(); mirrored[:, 0] = 100 - mirrored[:, 0]
    assert abs(head_pose(mirrored)["yaw"] + f["yaw"]) < 1e-6   # mirror symmetry
    up = front.copy(); up[2, 1] -= 5                          # nose rises toward the eye line (head tilted up)
    assert head_pose(up)["pitch"] < f["pitch"] - 0.1
    th = np.deg2rad(20); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rolled = (front - front[:2].mean(0)) @ R.T + 50           # in-plane head roll must not change yaw/pitch
    assert abs(head_pose(rolled)["yaw"] - f["yaw"]) < 1e-6 and abs(head_pose(rolled)["pitch"] - f["pitch"]) < 1e-6
