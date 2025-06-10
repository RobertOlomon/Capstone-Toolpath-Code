"""
Orientation Utilities Module
----------------------------
This module contains functions for computing end-effector (EE) orientations,
primarily ensuring the EE's approach vector aligns with a desired direction.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import DEBUG_ALLOW_EE_ROLL

def compute_EE_orientation(cleaning_point, effective_normal, roll_angle=0.0):
    """@brief Compute the end-effector orientation.

    Aligns the EE Z-axis opposite to ``effective_normal`` and optionally applies a
    roll about that axis if ``DEBUG_ALLOW_EE_ROLL`` is enabled.

    @param cleaning_point 3D position of the target point (unused).
    @param effective_normal Desired surface normal at the cleaning point.
    @param roll_angle Additional roll in degrees about the EE Z-axis.

    @return Quaternion ``[x, y, z, w]`` representing the orientation. Returns the
    identity quaternion if a valid orientation cannot be computed.
    """
    if effective_normal is None:
        return R.identity().as_quat()


    norm_val = np.linalg.norm(effective_normal)
    if norm_val < 1e-9:
        return R.identity().as_quat()
    unit_effective_normal = effective_normal / norm_val

    # EE's Z-axis (approach vector) is opposite to the surface normal
    desired_ee_z_axis = -unit_effective_normal
    
    # Determine EE's X-axis using world reference vectors.
    world_ref_vectors = [np.array([0,0,1]), np.array([0,1,0]), np.array([1,0,0])]
    desired_ee_x_axis = None

    for world_ref in world_ref_vectors:
        if abs(np.dot(desired_ee_z_axis, world_ref)) < 0.999:
            potential_x_axis = np.cross(world_ref, desired_ee_z_axis)
            if np.linalg.norm(potential_x_axis) > 1e-9:
                desired_ee_x_axis = potential_x_axis / np.linalg.norm(potential_x_axis)
                break
    
    if desired_ee_x_axis is None:
        return R.identity().as_quat()
        
    # EE's Y-axis completes the right-handed coordinate system: Y = Z x X
    desired_ee_y_axis = np.cross(desired_ee_z_axis, desired_ee_x_axis)
    # Normalization for Y is good practice, though mathematically should be unit if X,Z are unit and ortho.
    norm_y = np.linalg.norm(desired_ee_y_axis)
    if norm_y < 1e-9: # Should not happen if X and Z are well-defined and orthogonal
        # print("Warning: EE Y-axis has zero norm. Defaulting orientation.")
        return R.identity().as_quat()
    desired_ee_y_axis /= norm_y
    
    # Construct the rotation matrix from these basis vectors
    # Columns are the new X, Y, Z axes in the original coordinate system
    rotation_matrix = np.column_stack((desired_ee_x_axis, desired_ee_y_axis, desired_ee_z_axis))
    
    # Convert rotation matrix to quaternion
    try:
        base_orientation_scipy_R = R.from_matrix(rotation_matrix)
    except ValueError:
        # This can happen if rotation_matrix is not a valid rotation matrix (e.g. determinant is -1, or not orthogonal)
        # This might indicate an issue in axis generation despite checks.
        # print(f"Warning: Invalid rotation matrix for EE. Matrix:\n{rotation_matrix}\nEffective Normal: {effective_normal}")
        # Fallback to identity
        return R.identity().as_quat()

    # Apply additional roll if enabled
    if DEBUG_ALLOW_EE_ROLL and abs(roll_angle) > 1e-6:
        roll_rotation = R.from_rotvec(np.deg2rad(roll_angle) * desired_ee_z_axis)
        final_orientation_scipy_R = roll_rotation * base_orientation_scipy_R
    else:
        final_orientation_scipy_R = base_orientation_scipy_R
        
    return final_orientation_scipy_R.as_quat()
