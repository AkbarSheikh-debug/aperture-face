"""Face detection (YuNet, OpenCV Zoo, Apache-2.0) + our own 5-point similarity alignment.

Only the *detector* is third-party. Alignment is implemented here: the 5 landmarks are mapped onto the
standard 112x112 ArcFace template with a least-squares similarity transform (rotation + uniform scale +
translation), so every face has eyes / nose / mouth at the same pixel positions before it reaches the ResNet.
"""
from pathlib import Path

import cv2
import numpy as np

# Canonical landmark positions for a 112x112 face crop (left eye, right eye, nose, left mouth, right mouth),
# "left/right" meaning image left/right.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama (1991) least-squares similarity transform src -> dst. Returns a 2x3 matrix."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var_s = (s_c ** 2).sum() / n
    scale = np.trace(np.diag(S) @ D) / var_s
    t = mu_d - scale * R @ mu_s
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def order_landmarks(lm: np.ndarray) -> np.ndarray:
    """Make sure eyes and mouth corners are ordered image-left then image-right."""
    lm = lm.copy()
    if lm[0, 0] > lm[1, 0]:
        lm[[0, 1]] = lm[[1, 0]]
    if lm[3, 0] > lm[4, 0]:
        lm[[3, 4]] = lm[[4, 3]]
    return lm


def head_pose(landmarks: np.ndarray) -> dict:
    """Cheap head-pose cues from the 5 YuNet landmarks (no extra model).

    The landmarks are first de-rotated so the eye line is horizontal (removes in-plane roll), then
      yaw   = horizontal offset of the nose tip from the mid-eye point, in inter-eye distances.
              > 0: nose points toward the image's right, i.e. the person turned to THEIR left (camera not mirrored).
      pitch = (nose_y - eye_y) / (mouth_y - eye_y). It falls when the head tilts up and rises when it tilts down.
    These are relative cues: the guided enrolment compares them with the person's own frontal baseline,
    so the absolute value for an individual's face shape does not matter.
    """
    lm = order_landmarks(np.asarray(landmarks, np.float64).reshape(5, 2))
    eyes = lm[:2]
    eye_mid = eyes.mean(0)
    d = eyes[1] - eyes[0]
    roll = float(np.arctan2(d[1], d[0]))
    c, s = np.cos(-roll), np.sin(-roll)
    rot = (lm - eye_mid) @ np.array([[c, -s], [s, c]]).T
    eye_dist = float(np.linalg.norm(d)) + 1e-6
    mouth_y = rot[3:5, 1].mean()
    return {"yaw": float(rot[2, 0] / eye_dist),
            "pitch": float(rot[2, 1] / max(mouth_y, 1e-6)),
            "roll_deg": float(np.degrees(roll)),
            "eye_dist": eye_dist}


def warp_to_template(img: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    tmpl = ARCFACE_TEMPLATE * (size / 112.0)
    M = similarity_transform(order_landmarks(landmarks), tmpl)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def center_crop_fallback(img: np.ndarray, size: int = 112, frac: float = 0.45) -> np.ndarray:
    """Used only when the detector finds no face (logged and counted by the caller)."""
    h, w = img.shape[:2]
    side = int(min(h, w) * frac)
    cy, cx = int(h * 0.52), w // 2
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    crop = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class FaceAligner:
    """YuNet detector -> pick the subject face -> similarity-align to 112x112."""

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, nms_threshold, 5000)

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """Returns N x 15 array: x, y, w, h, 5 x (lx, ly), score."""
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    @staticmethod
    def pick_subject(faces: np.ndarray, img_shape) -> int:
        """Prefer large faces close to the image centre (LFW images are centred on the subject)."""
        h, w = img_shape[:2]
        areas = faces[:, 2] * faces[:, 3]
        cx = faces[:, 0] + faces[:, 2] / 2
        cy = faces[:, 1] + faces[:, 3] / 2
        dist = np.sqrt(((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2)
        score = areas / areas.max() - 2.0 * dist
        return int(np.argmax(score))

    def align(self, img_bgr: np.ndarray, size: int = 112, pick: str = "subject"):
        """Returns (aligned_crop, info dict). info['status'] in {'aligned', 'fallback_center'}."""
        faces = self.detect(img_bgr)
        if len(faces) == 0:
            return center_crop_fallback(img_bgr, size), {"status": "fallback_center", "n_faces": 0, "score": 0.0}
        idx = self.pick_subject(faces, img_bgr.shape) if pick == "subject" else int(np.argmax(faces[:, 2] * faces[:, 3]))
        f = faces[idx]
        lm = f[4:14].reshape(5, 2)
        crop = warp_to_template(img_bgr, lm, size)
        return crop, {"status": "aligned", "n_faces": int(len(faces)), "score": float(f[14]), "box": f[:4].tolist()}

    def align_all(self, img_bgr: np.ndarray, size: int = 112):
        """All faces in a frame (for the webcam demo / API). Returns list of (crop, box, score)."""
        out = []
        for f in self.detect(img_bgr):
            out.append((warp_to_template(img_bgr, f[4:14].reshape(5, 2), size), f[:4].astype(int), float(f[14])))
        return out
