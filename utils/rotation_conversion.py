import torch
import numpy as np

"""
Joint Selection & Group Constants for Sign Language Motion Generation
=====================================================================

Group definitions (by original 53-joint index):

  LOWER:  [0,1,2,3,4,5,7,8,10,11]  pelvis, spine1, hips, knees, ankles, feet (10)
  TORSO:  [6,9,12,13,14,15]         spine2, spine3, neck, L/R collar, head    (6)
  ARMS:   [16,17,18,19,20,21]       L/R shoulder, elbow, wrist                (6)
  LHAND:  [22-36]                    left hand fingers                         (15)
  RHAND:  [37-51]                    right hand fingers                        (15)
  JAW:    [52]                       jaw                                       (1)
                                                                         Total: 53

Three joint selection levels:

  all (53):          LOWER(10) + TORSO(6) + ARMS(6) + LHAND(15) + RHAND(15) + JAW(1)

Loss weight recommendation:
    0.0 * LOWER + 0.2 * TORSO + 5.0 * ARMS + 5.0 * LHAND + 5.0 * RHAND + 0.1 * JAW
"""

# ============================================================================
# Joint names
# ============================================================================

FULL_JOINT_NAMES = {
    0: 'pelvis', 1: 'left_hip', 2: 'right_hip', 3: 'spine1',
    4: 'left_knee', 5: 'right_knee', 6: 'spine2',
    7: 'left_ankle', 8: 'right_ankle', 9: 'spine3',
    10: 'left_foot', 11: 'right_foot', 12: 'neck',
    13: 'left_collar', 14: 'right_collar', 15: 'head',
    16: 'left_shoulder', 17: 'right_shoulder',
    18: 'left_elbow', 19: 'right_elbow',
    20: 'left_wrist', 21: 'right_wrist',
    52: 'jaw',
}
for i in range(15):
    FULL_JOINT_NAMES[22 + i] = f'lhand_{i}'
    FULL_JOINT_NAMES[37 + i] = f'rhand_{i}'
ALL_53_JOINTS = list(range(53))

'''
root(0) → spine1(3) → spine2(6) → spine3(9) → neck(12) → head(15)
                                              → L_collar(13) → L_shoulder(16) → L_elbow(18) → L_wrist(20) → L_hand...
'''
ROOT_INDICES = [0] ## control the whole body rotation
LOWER_BODY_INDICES = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_INDICES      = [3, 6, 9, 12, 13, 14, 15]
ARMS_INDICES       = [16, 17, 18, 19, 20, 21]
BODY_INDICES       = list(range(22))
LHAND_INDICES      = list(range(22, 37))
RHAND_INDICES      = list(range(37, 52))
JAW_INDICES        = [52]
ALL_INDICES        = [i for i in range(53)]

REMOVE_INDICES = set(LOWER_BODY_INDICES + JAW_INDICES)
UPPER_BODY_INDICES = [i for i in ALL_53_JOINTS if i not in REMOVE_INDICES]


def get_joint_slices(n_feats=3):
    """
    Return feature-level indices for each joint group.

    Args:
        n_feats: 3 (axis-angle) or 6 (rot6d)

    Returns:
        dict with: BODY, LOWER_BODY, TORSO, ARMS, LHAND, RHAND, JAW
        Each value is a list of int indices into dim=-1.
    """
    def idx(joints):
        out = []
        for j in joints:
            out.extend(range(j * n_feats, (j + 1) * n_feats))
        return out

    return {
        'ROOT':       idx(ROOT_INDICES),
        'BODY':       idx(BODY_INDICES),
        'LOWER_BODY': idx(LOWER_BODY_INDICES),
        'TORSO':      idx(TORSO_INDICES),
        'ARMS':       idx(ARMS_INDICES),
        'LHAND':      idx(LHAND_INDICES),
        'RHAND':      idx(RHAND_INDICES),
        'JAW':        idx(JAW_INDICES),
        'ALL':        idx(ALL_INDICES),
    }




# ============================================================================
# Rotation Conversion Utilities
# ============================================================================

def axis_angle_to_matrix(rotvec):
    """
    Convert axis-angle to rotation matrix using Rodrigues' formula.

    Args:
        rotvec: (*, 3) axis-angle vectors

    Returns:
        matrix: (*, 3, 3) rotation matrices
    """
    shape = rotvec.shape[:-1]
    angle = torch.norm(rotvec, dim=-1, keepdim=True)  # (*, 1)

    # Handle zero-angle case
    near_zero = (angle.squeeze(-1) < 1e-6)
    
    # Safe normalize
    axis = rotvec / (angle + 1e-8)  # (*, 3)

    cos_a = torch.cos(angle).unsqueeze(-1)  # (*, 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)  # (*, 1, 1)

    # Skew-symmetric matrix K
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).reshape(*shape, 3, 3)

    # Rodrigues: R = I + sin(a)*K + (1-cos(a))*K^2
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(*shape, 3, 3)
    R = I + sin_a * K + (1 - cos_a) * (K @ K)

    # For near-zero angles, use identity
    if near_zero.any():
        R[near_zero] = I[near_zero]

    return R


def matrix_to_rot6d(matrix):
    col1 = matrix[..., :, 0]  # (*, 3)
    col2 = matrix[..., :, 1]  # (*, 3)
    return torch.cat([col1, col2], dim=-1)  # (*, 6)


def rot6d_to_matrix(rot6d):
    """
    Convert 6D rotation back to rotation matrix via Gram-Schmidt.

    Args:
        rot6d: (*, 6) 6D rotation vectors

    Returns:
        matrix: (*, 3, 3) rotation matrices
    """
    shape = rot6d.shape[:-1]
    a1 = rot6d[..., :3]  # first column
    a2 = rot6d[..., 3:]  # second column

    # Gram-Schmidt orthogonalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (*, 3, 3)


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle.

    Args:
        matrix: (*, 3, 3) rotation matrices

    Returns:
        rotvec: (*, 3) axis-angle vectors
    """
    # Use the relationship: trace(R) = 1 + 2*cos(angle)
    # and the skew-symmetric part for the axis
    batch_shape = matrix.shape[:-2]

    # Angle from trace
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)  # (*)

    # Axis from skew-symmetric part: [R - R^T] / (2 sin(angle))
    skew = matrix - matrix.transpose(-2, -1)  # (*, 3, 3)
    axis = torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0]
    ], dim=-1)  # (*, 3)

    sin_angle = torch.sin(angle).unsqueeze(-1)  # (*, 1)
    near_zero = (angle < 1e-6).unsqueeze(-1)

    # Normalize axis
    axis = axis / (2.0 * sin_angle + 1e-8)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)

    rotvec = axis * angle.unsqueeze(-1)

    # Near-zero angle: return zeros
    rotvec = torch.where(near_zero.expand_as(rotvec), torch.zeros_like(rotvec), rotvec)

    return rotvec


def axis_angle_to_rot6d(rotvec):
    """Shortcut: axis-angle → 6D rotation."""
    return matrix_to_rot6d(axis_angle_to_matrix(rotvec))


def rot6d_to_axis_angle(rot6d):
    """Shortcut: 6D rotation → axis-angle."""
    return matrix_to_axis_angle(rot6d_to_matrix(rot6d))






def postprocess_motion(motion_raw, cfg):
    """
    Convert model output back to full SMPL-X axis-angle (T, 159).

    Args:
        motion_raw: np.ndarray (T, input_dim) — raw model output
                    e.g. (T, 264) if rot6d + upper_body
        cfg: config with USE_ROT6D, USE_UPPER_BODY

    Returns:
        np.ndarray (T, 159) — 53 joints × 3 axis-angle
    """
    use_rot6d = getattr(cfg, 'USE_ROT6D', False)
    use_upper_body = getattr(cfg, 'USE_UPPER_BODY', False)

    seq = torch.from_numpy(motion_raw).float()
    T = seq.shape[0]


    n_feats = 6 if use_rot6d else 3

    # (T, input_dim) → (T, N_joints, N_feats)
    seq = seq.reshape(T, -1, n_feats)

    # 6D → axis-angle
    if use_rot6d:
        seq = rot6d_to_axis_angle(seq)  # (T, N_joints, 3)

    # 44 upper body → 53 full body (lower body = zeros = neutral pose)

    return seq.reshape(T, -1).numpy()  # (T, 159)