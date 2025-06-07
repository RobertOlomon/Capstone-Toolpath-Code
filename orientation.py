"""
Orientation Utilities Module
----------------------------
This module contains functions for computing end-effector (EE) orientations,
primarily ensuring the EE's approach vector aligns with a desired direction.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import DEBUG_ALLOW_EE_ROLL # Import debug flag for enabling EE roll

def compute_EE_orientation(cleaning_point, effective_normal, roll_angle=0.0):
    """
    Computes the end-effector (EE) orientation as a quaternion.

    The primary goal is to align the EE's Z-axis to be opposite to the `effective_normal`.
    The EE's X-axis is determined using the cross product of a world reference vector
    and the EE's Z-axis. The Y-axis completes the right-handed system.

    If `DEBUG_ALLOW_EE_ROLL` is True and `roll_angle` is non-zero, an additional
    rotation (roll) around the EE's Z-axis is applied.

    Parameters:
        cleaning_point (np.ndarray): (3,) 3D position of the target cleaning point. (Currently unused here)
        effective_normal (np.ndarray): (3,) Desired normal vector at the cleaning point.
        roll_angle (float): Additional roll angle in degrees around the EE's Z-axis.

    Returns:
        np.ndarray: (4,) Quaternion (x, y, z, w) representing the EE orientation.
                    Returns identity quaternion if normal is invalid and cannot form a basis.
    """
    if effective_normal is None:
        # Default to a safe orientation (e.g., EE Z points along world -Z)
        # This implies normal was [0,0,1]
        # Identity matrix for rotation: X->X, Y->Y, Z->Z.
        # If we want EE Z to be -world Z, then:
        # EE_X = world_X, EE_Y = -world_Y, EE_Z = -world_Z
        # return R.from_matrix(np.array([[1,0,0],[0,-1,0],[0,0,-1]])).as_quat()
        # For now, let's assume identity is a "neutral" fallback if normal is None.
        # print("Warning: compute_EE_orientation received None for effective_normal. Returning identity.")
        return R.identity().as_quat()


    norm_val = np.linalg.norm(effective_normal)
    if norm_val < 1e-9: 
        # If normal is zero vector, cannot determine orientation.
        # Fallback to a default, e.g. EE pointing down global -Z.
        # print("Warning: compute_EE_orientation received near-zero effective_normal. Returning identity.")
        return R.identity().as_quat()
    unit_effective_normal = effective_normal / norm_val

    # EE's Z-axis (approach vector) is opposite to the surface normal
    desired_ee_z_axis = -unit_effective_normal
    
    # Determine EE's X-axis using a world reference vector.
    # Try global Z-up first. If Z_ee is collinear, try Y-up, then X-up.
    world_ref_vectors = [np.array([0,0,1]), np.array([0,1,0]), np.array([1,0,0])]
    desired_ee_x_axis = None

    for world_ref in world_ref_vectors:
        if abs(np.dot(desired_ee_z_axis, world_ref)) < 0.999: # Check for non-collinearity
            # If not collinear, their cross product will be non-zero.
            potential_x_axis = np.cross(world_ref, desired_ee_z_axis)
            if np.linalg.norm(potential_x_axis) > 1e-9: # Ensure cross product is not zero
                desired_ee_x_axis = potential_x_axis / np.linalg.norm(potential_x_axis)
                break
    
    if desired_ee_x_axis is None:
        # This case is very rare: if desired_ee_z_axis is zero (handled earlier)
        # or if it aligns perfectly with all three world axes (impossible).
        # Fallback if no suitable X could be found (e.g. if z_axis was [0,0,0])
        # print("Warning: Could not determine EE X-axis. Defaulting orientation.")
        # A common default: Z points down, X points along world X.
        # desired_ee_z_axis = np.array([0,0,-1.0])
        # desired_ee_x_axis = np.array([1,0,0])
        # For now, if this unlikely case is hit, rely on identity from matrix conversion failure.
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

    # Apply additional roll if debug flag is enabled and roll_angle is significant
    if DEBUG_ALLOW_EE_ROLL and abs(roll_angle) > 1e-6:
        # Create a rotation for the roll around the EE's Z-axis (which is desired_ee_z_axis)
        # The rotation vector for roll is along the EE's Z-axis
        roll_rotation = R.from_rotvec(np.deg2rad(roll_angle) * desired_ee_z_axis)
        # Combine the base orientation with the roll: R_final = R_roll * R_base
        final_orientation_scipy_R = roll_rotation * base_orientation_scipy_R
    else:
        final_orientation_scipy_R = base_orientation_scipy_R
        
    return final_orientation_scipy_R.as_quat() # Return as [x, y, z, w]