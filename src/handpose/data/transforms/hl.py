import math
import numpy as np
import torch

# -------------------------
# Image transform (simple)
# -------------------------
class HLImageTransform:
    def __init__(self, image_size=224):
        from torchvision import transforms as T

        self.tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, sample):
        sample["pixel_values"] = self.tf(sample["image"])
        return sample


# -------------------------
# HL label transform
# -------------------------
def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps)


def _hand_frame_from_joints(hand21, hand_side: str):
    """
    Paper coordinate setting:
    - origin: wrist
    - +Y: wrist -> middle_mcp
    - +Z: perpendicular to Y and "left on the plane of the palm"
          (implemented by cross(Y, palm_vec), where palm_vec is along index->pinky)
    - +X: Y x Z

    Key issue: left/right hand mirroring.
    We enforce a consistent semantic: +Z always points to the palm-left direction
    in each hand's own coordinate. For the left hand, the raw construction tends to flip.
    We correct by mirroring axes for left hand so that local coordinates become comparable.
    """
    wrist = hand21[0]
    middle_mcp = hand21[9]
    index_mcp = hand21[5]
    pinky_mcp = hand21[17]

    y = _normalize(middle_mcp - wrist)
    palm_vec = _normalize(pinky_mcp - index_mcp)

    # z points to "left on palm plane" (right-hand rule with our palm_vec definition)
    z = _normalize(np.cross(y, palm_vec))
    x = _normalize(np.cross(y, z))

    R = np.stack([x, y, z], axis=1).astype(np.float32)  # 3x3 (columns)

    # Mirror fix for left hand to keep +Z meaning consistent across hands.
    # This is the common practical fix: flip X and Z for left to make frame right-handed
    # and keep "palm-left" consistent.
    if hand_side.lower().startswith("l"):
        R[:, 0] *= -1.0  # flip x
        R[:, 2] *= -1.0  # flip z

    return wrist.astype(np.float32), R  # origin, R(world axes of local frame)


# 21-joint parent list (common hand skeleton)
# 0 wrist
# thumb: 1-4, index: 5-8, middle: 9-12, ring: 13-16, pinky: 17-20
PARENTS_21 = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 0,
    6: 5,
    7: 6,
    8: 7,
    9: 0,
    10: 9,
    11: 10,
    12: 11,
    13: 0,
    14: 13,
    15: 14,
    16: 15,
    17: 0,
    18: 17,
    19: 18,
    20: 19,
}

NONROOT_20 = [i for i in range(21) if i != 0]


def _regional_vectors_local(hand21, hand_side: str):
    """
    Compute 20 regional vectors (child-parent) in hand-local coordinates (paper-defined).
    return: (20,3) float32
    """
    origin, R = _hand_frame_from_joints(hand21, hand_side=hand_side)
    Rt = R.T  # world->local rotation
    vecs = []
    for child in NONROOT_20:
        parent = PARENTS_21[child]
        v_world = hand21[child] - hand21[parent]
        v_local = Rt @ v_world
        vecs.append(v_local.astype(np.float32))
    return np.stack(vecs, axis=0)  # (20,3)


# -------------------------
# Paper-style spherical binning
# -------------------------
def _cartesian_to_spherical_angles_deg(v):
    """
    Match paper Eq.(2) convention:
      r = sqrt(x^2+y^2+z^2)
      theta = arccos(z/r) in [0,180]
      phi = atan2(y,x)  in [-180,180]
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    r = math.sqrt(x * x + y * y + z * z) + 1e-8
    theta = math.degrees(math.acos(max(-1.0, min(1.0, z / r))))
    phi = math.degrees(math.atan2(y, x))  # [-180,180]
    return theta, phi


def _theta_to_level(theta_deg: float):
    """
    From Fig.2: High / Normal / Low with boundaries around 22.5°, 67.5°, etc.
    The paper states Z divided into 3 intervals.
    Use:
      High: theta in [0, 22.5)
      Low : theta in (157.5, 180]
      Normal: otherwise
    """
    if theta_deg < 22.5:
        return "high"
    if theta_deg > 157.5:
        return "low"
    return "normal"


def _phi_to_sector9(phi_deg: float):
    """
    9 intervals on [-180,180], centered every 40 degrees:
      centers: -160,-120,-80,-40,0,40,80,120,160
    using half-width 20 degrees.
    We implement by shifting +180 then dividing into 9 equal bins (40deg each).
    bin index: 0..8
    """
    # map to [0,360)
    phi = (phi_deg + 360.0) % 360.0
    # shift so that 0 deg is centered in middle bin (bin 4)
    # easiest: move range to [-180,180) then compute:
    phi2 = ((phi_deg + 180.0) % 360.0) - 180.0  # [-180,180)
    # convert to [0,360) then bins of 40
    u = phi2 + 180.0
    b = int(u // 40.0)
    # clamp to 0..8
    return max(0, min(8, b))


def _sector9_to_dir8(sector9: int):
    """
    We need 8 horizontal directions + an extra 'center/around' bucket to get to 9.
    The paper basic symbols are 26 (6 faces + 12 edges + 8 corners).
    A typical way to realize 26 from (3 levels + planar dirs) is:
      - use 8 planar dirs for normal (N, NE, E, SE, S, SW, W, NW)
      - plus pure up and pure down
      => 8*3 + 2 = 26

    So we map sector9 (0..8) -> dir8 (0..7) by merging the extreme bins:
      sector9: 0..8
      merge (0 and 8) to same direction "near -180/+180" (west-ish depending on convention)
    """
    if sector9 == 8:
        return 0
    # now 0..7
    return sector9


# Class indexing convention (you must keep it fixed across training/eval)
# 26 classes total.
# We'll implement:
#  - class 0: UP (face +Z)
#  - class 1: DOWN (face -Z)
#  - classes 2..9   : NORMAL level, 8 dirs
#  - classes 10..17 : HIGH level,   8 dirs (corners/edges "up-ish")
#  - classes 18..25 : LOW level,    8 dirs (corners/edges "down-ish")
CLASS_UP = 0
CLASS_DOWN = 1
CLASS_NORMAL_BASE = 2
CLASS_HIGH_BASE = 10
CLASS_LOW_BASE = 18


def _vector_to_hl_class_nonthumb(v_local: np.ndarray):
    """
    Non-thumb fingers: "larger range in XY plane" -> keep 8-dir + 3-level discretization.
    """
    theta, phi = _cartesian_to_spherical_angles_deg(v_local)
    level = _theta_to_level(theta)

    # near pure up/down: ignore phi and use face centers
    if level == "high":
        # if it's strongly upward, map to UP face
        if theta < 22.5:
            return CLASS_UP
    if level == "low":
        if theta > 157.5:
            return CLASS_DOWN

    sector9 = _phi_to_sector9(phi)
    dir8 = _sector9_to_dir8(sector9)

    if level == "normal":
        return CLASS_NORMAL_BASE + dir8
    if level == "high":
        return CLASS_HIGH_BASE + dir8
    return CLASS_LOW_BASE + dir8


def _vector_to_hl_class_thumb(v_local: np.ndarray):
    """
    Thumb: paper says predominant motion along Z and smaller within XY.
    Practical discretization:
      - keep 3 levels by theta (high/normal/low)
      - compress phi into 3 coarse bins to reduce XY sensitivity:
          forward (around +Y), backward (around -Y), side (others)
    Then map to the same 26-class space by reusing the 8-dir slots but only using
    a subset of directions (duplicating bins to keep 26 fixed).
    This keeps training target space unchanged (26 classes), but thumb labels become
    less sensitive to phi.

    If you later get the authors' exact thumb mapping, replace this function only.
    """
    theta, phi = _cartesian_to_spherical_angles_deg(v_local)
    level = _theta_to_level(theta)

    # pure up/down
    if level == "high" and theta < 22.5:
        return CLASS_UP
    if level == "low" and theta > 157.5:
        return CLASS_DOWN

    # phi coarse bins based on y-axis forward/back in local frame
    # We'll define:
    #   forward:  phi in [-45, 45] around +X? (careful: phi is atan2(y,x))
    # Since phi=atan2(y,x):
    #   +Y corresponds to phi=+90
    #   -Y corresponds to phi=-90
    # We'll use around +Y / -Y / rest.
    if 45.0 <= phi <= 135.0:
        coarse = "forward"   # +Y
    elif -135.0 <= phi <= -45.0:
        coarse = "backward"  # -Y
    else:
        coarse = "side"

    # map coarse to a representative dir8 index (0..7):
    # use 3 anchors: E(0deg), N(90deg), S(-90deg) style depends on your convention.
    # We'll pick:
    #   forward  -> dir8 = 2 (represent "north-ish")
    #   backward -> dir8 = 6 (south-ish)
    #   side     -> dir8 = 0 (east/west-ish)
    if coarse == "forward":
        dir8 = 2
    elif coarse == "backward":
        dir8 = 6
    else:
        dir8 = 0

    if level == "normal":
        return CLASS_NORMAL_BASE + dir8
    if level == "high":
        return CLASS_HIGH_BASE + dir8
    return CLASS_LOW_BASE + dir8


def _vector_to_hl_class(v_local: np.ndarray, is_thumb: bool):
    v_local = np.asarray(v_local, dtype=np.float32)
    n = np.linalg.norm(v_local) + 1e-8
    v_local = v_local / n
    if is_thumb:
        return _vector_to_hl_class_thumb(v_local)
    return _vector_to_hl_class_nonthumb(v_local)


# In our 20 vectors order, which ones belong to thumb?
# NONROOT_20 corresponds to children [1..20] in order.
# Thumb joints are 1,2,3,4  => first 4 vectors are thumb vectors.
THUMB_VECTOR_INDICES = {0, 1, 2, 3}  # indices in the 20-vector array


class Keypoints3DToHLTargets:
    """
    Convert keypoints_3d (42,3) -> labels (40,) LongTensor.

    Assumes:
    - first 21 rows are right hand, next 21 rows are left hand
    """

    def __init__(self, assume_right_first=True):
        self.assume_right_first = assume_right_first

    def __call__(self, sample):
        kp = np.asarray(sample["keypoints_3d"], dtype=np.float32)
        if kp.shape != (42, 3):
            raise ValueError(f"Expected (42,3), got {kp.shape}")

        if self.assume_right_first:
            right = kp[:21]
            left = kp[21:]
        else:
            left = kp[:21]
            right = kp[21:]

        r_vecs = _regional_vectors_local(right, hand_side="right")  # (20,3)
        l_vecs = _regional_vectors_local(left, hand_side="left")    # (20,3)

        r_cls = []
        for i, v in enumerate(r_vecs):
            is_thumb = i in THUMB_VECTOR_INDICES
            r_cls.append(_vector_to_hl_class(v, is_thumb=is_thumb))
        r_cls = np.asarray(r_cls, dtype=np.int64)

        l_cls = []
        for i, v in enumerate(l_vecs):
            is_thumb = i in THUMB_VECTOR_INDICES
            l_cls.append(_vector_to_hl_class(v, is_thumb=is_thumb))
        l_cls = np.asarray(l_cls, dtype=np.int64)

        labels = np.concatenate([r_cls, l_cls], axis=0)  # (40,)
        sample["labels"] = torch.from_numpy(labels)      # LongTensor[40]
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample