""""
Scan and Point Processing Utilities Module
------------------------------------------
This module includes functions for reordering scan points, classifying scanned points
during toolpath execution (for simulation/visualization), and interpolating toolpath steps.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from orientation import compute_EE_orientation # Used for interpolation

def interpolate_toolpath_steps(toolpath, scan_spacing=10.0, tol_angle=1e-3, tol_normal=1e-3):
    """
    Interpolates groups of consecutive toolpath steps that share nearly identical
    part rotation angles, cleaning normals, and MUST_PULL_BACK_flag.

    Replaces such groups with a series of steps interpolated between the group's
    start and end EE positions. Interpolated steps are flagged.

    Parameters:
        toolpath (list): List of toolpath steps. Each step is an 8-element tuple:
            (EE_pos, EE_quat, angle, clean_pt, normal, class, unreach, MUST_PULL_BACK_flag)
        scan_spacing (float): Desired distance between interpolated EE positions (mm).
        tol_angle (float): Tolerance for comparing part rotation angles (degrees).
        tol_normal (float): Tolerance for comparing normal vectors (L2 norm of difference).
    
    Returns:
        list: A new toolpath list where groups are replaced by interpolated steps.
              All steps in the returned list are 11-element tuples:
              (EE_pos, EE_quat, angle, clean_pt, normal, "interpolated", unreach, 
               MUST_PULL_BACK_flag_from_group, True (interp_flag), new_pts_empty, double_pts_empty)
    """
    new_toolpath = []
    group = [] 
    
    def process_group(grp):
        if not grp: return

        if len(grp) > 1: 
            first_step = grp[0] 
            last_step = grp[-1] 
            start_EE = first_step[0] 
            end_EE = last_step[0]   
            
            # Check if start_EE or end_EE is None (can happen for unreachable points)
            if start_EE is None or end_EE is None:
                 # If endpoints are None, cannot interpolate. Add group as individual steps.
                for step_g in grp:
                    new_toolpath.append(
                        tuple(list(step_g)) + (False, np.empty((0,3)), np.empty((0,3))) # Add interp_flag, points
                    )
                return

            dist = np.linalg.norm(end_EE - start_EE)
            num_interpolated_scans = max(int(np.ceil(dist / scan_spacing)) + 1, 2) 
            
            must_pull_back_for_group = first_step[7] # MUST_PULL_BACK from first step

            for i in range(num_interpolated_scans):
                alpha = i / (num_interpolated_scans - 1) if num_interpolated_scans > 1 else 0.0
                interp_EE_pos = start_EE * (1 - alpha) + end_EE * alpha 
                
                # Recompute EE quat based on group's common cleaning point and normal
                # This assumes cleaning_point (idx 3) and normal (idx 4) are valid for compute_EE_orientation
                if first_step[3] is not None and first_step[4] is not None:
                    interp_EE_quat = compute_EE_orientation(first_step[3], first_step[4]) 
                else: # Fallback if normal/point is None (should not happen for interpolatable groups)
                    interp_EE_quat = first_step[1] # Use original quaternion

                new_step_tuple = (
                    interp_EE_pos, interp_EE_quat,
                    first_step[2], first_step[3], first_step[4], "interpolated", first_step[6],
                    must_pull_back_for_group, # MUST_PULL_BACK_flag (8th)
                    True,                     # Interpolation flag (9th)
                    np.empty((0, 3)),         # New points (10th)
                    np.empty((0, 3))          # Double points (11th)
                )
                new_toolpath.append(new_step_tuple)
        else: 
            step = grp[0] # Original 8-element tuple
            final_step_tuple = tuple(list(step)) + \
                               (False, np.empty((0,3)), np.empty((0,3))) # Add interp_flag=False, empty points
            new_toolpath.append(final_step_tuple)
    
    for step_item in toolpath: # Input step_item is 8-element
        # Cannot interpolate if core kinematic data is None
        is_interpolatable = not (step_item[0] is None or step_item[2] is None or step_item[4] is None)

        if not is_interpolatable:
            if group: 
                process_group(group)
                group = [] 
            new_toolpath.append(tuple(list(step_item)) + (False, np.empty((0,3)), np.empty((0,3))))
            continue

        if not group: 
            group.append(step_item)
        else:
            last_in_group = group[-1]
            # Group if angle, normal, AND must_pull_back_flag are consistent
            if abs(step_item[2] - last_in_group[2]) < tol_angle and \
               np.allclose(step_item[4], last_in_group[4], atol=tol_normal, equal_nan=True) and \
               step_item[7] == last_in_group[7]: # Compare MUST_PULL_BACK_flag (idx 7)
                group.append(step_item) 
            else: 
                process_group(group)
                group = [step_item]
                
    if group: 
        process_group(group)
        
    return new_toolpath

def reorder_scan_points_by_normals(points, normals, eps=0.3, min_samples=5):
    """
    Reorders scan points by first clustering them based on similar normal vectors
    using DBSCAN, then ordering points within each cluster using a greedy
    nearest-neighbor approach.
    """
    if points is None or normals is None or len(points) == 0 or len(normals) == 0:
        return np.empty((0,3)), np.empty((0,3))
        
    points_arr = np.array(points)
    normals_arr = np.array(normals)

    if points_arr.shape[0] != normals_arr.shape[0]:
        raise ValueError("Points and normals must have the same number of entries.")
    if points_arr.shape[0] == 0:
        return np.empty((0,3)), np.empty((0,3))

    try:
        norm_magnitudes = np.linalg.norm(normals_arr, axis=1, keepdims=True)
        safe_norm_magnitudes = np.where(norm_magnitudes < 1e-9, 1e-9, norm_magnitudes) # Avoid div by zero
        normalized_normals = normals_arr / safe_norm_magnitudes
        
        # DBSCAN can fail if min_samples > n_features_in, handle this
        actual_min_samples = min(min_samples, normalized_normals.shape[0])
        if actual_min_samples < 1 : actual_min_samples = 1 # Ensure positive

        clustering = DBSCAN(eps=eps, min_samples=actual_min_samples).fit(normalized_normals)
    except Exception as e: 
        print(f"DBSCAN clustering failed: {e}. Returning points in original order.")
        return points_arr, normals_arr

    labels = clustering.labels_
    unique_labels = np.unique(labels) # Sorted unique labels, -1 (noise) might be first

    ordered_indices = [] 
    # Process non-noise clusters first
    for lab in unique_labels: 
        if lab == -1: 
            continue 
        
        cluster_member_indices_original = np.where(labels == lab)[0] 
        if len(cluster_member_indices_original) == 0:
            continue
            
        # Use a mutable list of original indices for this cluster for pop operations
        current_cluster_original_indices_list = list(cluster_member_indices_original) 
        
        # Start greedy ordering: pick the point with the smallest original index in this cluster as start
        start_idx_in_list = np.argmin(current_cluster_original_indices_list)
        ordered_in_cluster_global_indices = [current_cluster_original_indices_list.pop(start_idx_in_list)] 
        
        while current_cluster_original_indices_list: 
            last_added_global_idx = ordered_in_cluster_global_indices[-1]
            last_point_coords = points_arr[last_added_global_idx]
            
            # Consider only points remaining in this cluster for distance calculation
            remaining_points_global_indices_in_cluster = np.array(current_cluster_original_indices_list)
            remaining_points_coords_in_cluster = points_arr[remaining_points_global_indices_in_cluster] 
            
            distances = np.linalg.norm(remaining_points_coords_in_cluster - last_point_coords, axis=1)
            
            # nearest_neighbor_local_idx is index into remaining_points_global_indices_in_cluster
            nearest_neighbor_local_idx = np.argmin(distances) 
            # Get the global index of this nearest neighbor
            nearest_neighbor_global_idx = remaining_points_global_indices_in_cluster[nearest_neighbor_local_idx]
            
            # Remove it from current_cluster_original_indices_list (find by value, then pop)
            current_cluster_original_indices_list.pop(current_cluster_original_indices_list.index(nearest_neighbor_global_idx))
            
            ordered_in_cluster_global_indices.append(nearest_neighbor_global_idx)
            
        ordered_indices.extend(ordered_in_cluster_global_indices)
        
    # Append noise points at the end, maintaining their original relative order
    noise_indices_original = np.where(labels == -1)[0]
    if len(noise_indices_original) > 0:
        ordered_indices.extend(list(noise_indices_original))

    if len(ordered_indices) != points_arr.shape[0]:
        # This might happen if DBSCAN produced unexpected labels or logic error.
        print(f"Warning: Reordering result size ({len(ordered_indices)}) mismatch with input size ({points_arr.shape[0]}). Returning original order.")
        return points_arr, normals_arr

    return points_arr[ordered_indices], normals_arr[ordered_indices]

def classify_step_scanned_points(step, scanned_count, dense_pts_local, dense_normals,
                                 part_origin, part_y_axis, part_center,
                                 scan_size, offset, offset_margin):
    """
    Simulates scanning for a single toolpath step.
    """
    pos, quat, angle, _cleaning_point, chosen_normal, _classification, _unreachable = step[:7]

    if pos is None or quat is None or angle is None or chosen_normal is None:
        return np.empty((0, 3)), np.empty((0, 3))
    if dense_pts_local.shape[0] == 0: 
        return np.empty((0,3)), np.empty((0,3))

    R_part_to_global_obj = R.from_rotvec(part_y_axis * np.deg2rad(angle))
    T_part_local_to_global = np.eye(4)
    T_part_local_to_global[:3, :3] = R_part_to_global_obj.as_matrix()
    T_part_local_to_global[:3, 3] = part_origin - R_part_to_global_obj.apply(part_center)

    dense_homog_local = np.hstack([dense_pts_local, np.ones((dense_pts_local.shape[0], 1))])

    T_EE_global = np.eye(4)
    T_EE_global[:3, :3] = R.from_quat(quat).as_matrix()
    T_EE_global[:3, 3] = pos
    
    T_global_to_EE_local = np.linalg.inv(T_EE_global)
    T_part_local_to_EE_local = T_global_to_EE_local @ T_part_local_to_global

    pts_in_EE_frame_homog = (T_part_local_to_EE_local @ dense_homog_local.T).T
    pts_in_EE_frame = pts_in_EE_frame_homog[:, :3] 

    geom_mask = (
        (pts_in_EE_frame[:, 0] >= -scan_size / 2) & (pts_in_EE_frame[:, 0] <= scan_size / 2) & 
        (pts_in_EE_frame[:, 1] >= -scan_size / 2) & (pts_in_EE_frame[:, 1] <= scan_size / 2) & 
        (pts_in_EE_frame[:, 2] >= offset - offset_margin) & (pts_in_EE_frame[:, 2] <= offset + offset_margin) 
    )

    dense_normals_global = R_part_to_global_obj.apply(dense_normals)
    norm_chosen_normal = np.linalg.norm(chosen_normal)
    if norm_chosen_normal < 1e-9: 
        normal_mask = np.zeros(dense_normals_global.shape[0], dtype=bool)
    else:
        unit_chosen_normal = chosen_normal / norm_chosen_normal
        dot_products = np.einsum('ij,j->i', dense_normals_global, unit_chosen_normal) 
        alignment_threshold = np.cos(np.deg2rad(45)) # Relaxed from 30 to 45 for wider acceptance
        normal_mask = (dot_products >= alignment_threshold)
    
    final_scan_mask = geom_mask & normal_mask
    
    # Ensure scanned_count is not empty if final_scan_mask can be non-empty
    if scanned_count.size == 0 and final_scan_mask.any():
        # This case should be prevented by N_dense_anim check in animation or
        # dense_pts_local check at start of this function.
        # If scanned_count is empty, it implies no dense points to track.
        return np.empty((0,3)), np.empty((0,3))


    newly_scanned_indices = np.where((scanned_count == 0) & final_scan_mask)[0]
    doubly_scanned_indices = np.where((scanned_count > 0) & final_scan_mask)[0]
    
    scanned_count[final_scan_mask] += 1
    
    new_points_coords_local = dense_pts_local[newly_scanned_indices]
    double_points_coords_local = dense_pts_local[doubly_scanned_indices]

    return new_points_coords_local, double_points_coords_local