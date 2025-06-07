"""
Toolpath Pruning Module
-----------------------
This module contains functions to prune redundant toolpath steps based on
simulated scan coverage.
"""
import numpy as np
from scan_utils import classify_step_scanned_points # Core function for simulating scan coverage

def prune_toolpath_steps(toolpath, dense_pts_local, dense_normals,
                         part_origin, part_y_axis, part_center,
                         scan_size, offset, offset_margin):
    """
    Prunes the toolpath by retaining only steps that contribute to new scan coverage.

    It simulates the scanning process for each step using `classify_step_scanned_points`
    and keeps a step if it scans any new points on a dense representation of the part.
    
    Input steps can be 8-element (from planner) or 11-element (from interpolation).
    Output steps are always 11-element tuples.
    
    Parameters:
        toolpath (list): Input list of toolpath steps.
                         8-elem: (EE_pos, EE_quat, angle, clean_pt, normal, class, unreach, MUST_PULL_BACK)
                         11-elem: (..., MUST_PULL_BACK, interp_flag, new_pts, double_pts)
        dense_pts_local (np.ndarray): (N,3) Dense point cloud of the part in its local frame.
        dense_normals (np.ndarray): (N,3) Normals for the dense point cloud.
        part_origin (np.ndarray): Global origin of the part at angle=0.
        part_y_axis (np.ndarray): Global axis of part rotation.
        part_center (np.ndarray): Local pivot point for part rotation.
        scan_size (float): Scanning area size for simulation.
        offset (float): Nominal EE-to-part distance for simulation.
        offset_margin (float): Scan depth tolerance for simulation.
    
    Returns:
        list: A new list of toolpath steps (`pruned_steps`). Each step is an 11-element tuple:
              (EE_pos, EE_quat, angle, clean_pt, normal, class, unreach, 
               MUST_PULL_BACK_flag, interpolation_flag, 
               new_points_array, double_points_array)
              Only steps that scan new points are included (unless they are marked as interpolated).
    """
    if dense_pts_local.shape[0] == 0: 
        # print("Warning: Pruning with empty dense_pts_local. All steps will have empty new/double points, retaining original steps.")
        processed_steps_for_no_dense_pts = []
        for step_data in toolpath:
            current_len = len(step_data)
            base_step_parts = list(step_data[:7]) if current_len >=7 else [None]*7 # core 7: pos to unreach
            must_pull_back_flag = step_data[7] if current_len >= 8 else False
            
            if current_len == 11: # Already has interp_flag and points (likely from interpolator)
                interp_flag = step_data[8]
                # Use existing points if available, otherwise empty.
                new_pts = step_data[9] if isinstance(step_data[9], np.ndarray) else np.empty((0,3))
                double_pts = step_data[10] if isinstance(step_data[10], np.ndarray) else np.empty((0,3))
            else: # From planner (8-element) or unknown, default interp_flag and empty points
                interp_flag = False 
                new_pts = np.empty((0,3))
                double_pts = np.empty((0,3))

            augmented_step = tuple(base_step_parts) + \
                             (must_pull_back_flag, interp_flag, new_pts, double_pts)
            processed_steps_for_no_dense_pts.append(augmented_step)
        return processed_steps_for_no_dense_pts


    N_dense = dense_pts_local.shape[0]
    scanned_count_for_pruning = np.zeros(N_dense, dtype=int) 
    
    pruned_steps = []

    for step_data in toolpath:
        # step_data can be 8 or 11 elements.
        # classify_step_scanned_points needs the first 7.
        newly_scanned_pts, rescanned_pts = classify_step_scanned_points(
            step_data[:7], 
            scanned_count_for_pruning, 
            dense_pts_local, dense_normals,
            part_origin, part_y_axis, part_center,
            scan_size, offset, offset_margin
        )
        
        current_len = len(step_data)
        core_step_info = list(step_data[:7]) # EE_pos to unreachable

        if current_len == 8: # Planner output: (core_7, MUST_PULL_BACK)
            must_pull_back_flag = step_data[7]
            interpolation_flag = False # Default, not from interpolator
        elif current_len == 11: # Interpolator output: (core_7, MUST_PULL_BACK, interp_flag, old_new_pts, old_double_pts)
            must_pull_back_flag = step_data[7]
            interpolation_flag = step_data[8]
            # Note: we are using newly_scanned_pts from the current classification, not old_new_pts from interpolator.
        else: 
            # Fallback for unexpected length
            # print(f"Warning: Pruning encountered step_data with unexpected length {current_len}.")
            must_pull_back_flag = step_data[7] if current_len > 7 else False
            interpolation_flag = step_data[8] if current_len > 8 else False
            
        augmented_step_tuple = tuple(core_step_info) + \
                               (must_pull_back_flag, interpolation_flag, 
                                newly_scanned_pts, rescanned_pts)
        
        # Pruning logic:
        # Keep if it's an interpolated step (interpolation_flag is True), OR
        # Keep if it scans new points.
        # Also keep if it's unreachable (pos is None), as these are markers for failed points.
        is_unreachable_step = core_step_info[0] is None 

        if interpolation_flag or newly_scanned_pts.shape[0] > 0 or is_unreachable_step:
            pruned_steps.append(augmented_step_tuple)
            
    return pruned_steps