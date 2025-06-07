"""
Toolpath Planning Module
------------------------
This module contains the main logic for planning a collision-free toolpath
for the laser-cleaning process.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from tqdm import tqdm
from trimesh.collision import CollisionManager

from constants import DEBUG_INTERPOLATE_TOOLPATH
from mesh_utils import get_ee_box_mesh, get_laser_beam_mesh 
from orientation import compute_EE_orientation
from collision_utils import candidate_collision_check_trimesh
from scan_utils import interpolate_toolpath_steps

def plan_toolpath(cleaning_points, cleaning_normals, part_origin, part_y_axis,
                                     part_center_pivot, 
                                     local_chuck_mesh, 
                                     offset=200,
                                     lambda_angle=0.1, lambda_vis=1.0, visibility_threshold=0.2,
                                     lambda_deviation=10.0, lambda_center=5.0, lambda_pitch=20.0,
                                     starting_angle=0.0, move_safety_margin=50.0, 
                                     table_threshold=1000.0, stl_mesh=None,
                                     ee_box_extents=[-140, 444.5, -130, 280, -257, 88],
                                     env_collision_manager=None, 
                                     scan_size_for_beam_check=25, 
                                     offset_margin_for_beam_check=5, 
                                     coarse_step=20, fine_step=5, fine_delta=20,
                                     disable_intermediate_collision=True):
    """@brief Plan a collision-free cleaning toolpath.

    @param cleaning_points Array of cleaning point positions.
    @param cleaning_normals Array of corresponding normals.
    @param part_origin Global origin of the part.
    @param part_y_axis Rotation axis of the part.
    @param part_center_pivot Local rotation pivot.
    @param local_chuck_mesh Chuck mesh or ``None``.
    @param offset Nominal EE offset distance.
    @param lambda_angle Cost weight for angle deviation.
    @param lambda_vis Cost weight for visibility.
    @param visibility_threshold Visibility metric threshold.
    @param lambda_deviation Cost weight for orientation deviation.
    @param lambda_center Cost weight for centering.
    @param lambda_pitch Cost weight for pitch.
    @param starting_angle Initial part rotation angle.
    @param move_safety_margin Safety margin for movement.
    @param table_threshold Z threshold for table clearance.
    @param stl_mesh Part mesh for collision checking.
    @param ee_box_extents EE collision box extents.
    @param env_collision_manager Environment collision manager.
    @param scan_size_for_beam_check Scan size for beam collision checks.
    @param offset_margin_for_beam_check Margin for beam distance checks.
    @param coarse_step Coarse angular step.
    @param fine_step Fine angular step.
    @param fine_delta Fine deviation angle.
    @param disable_intermediate_collision Disable mid-step collision checking.

    @return List of 8-element toolpath step tuples.
    """
    steps = []
    previous_angle = starting_angle
    previous_EE = None 

    cm_part_check = CollisionManager() 
    if stl_mesh is not None and not stl_mesh.is_empty:
        cm_part_check.add_object("part", stl_mesh)

    ee_box_base = get_ee_box_mesh(ee_box_extents)
    laser_beam_base = get_laser_beam_mesh(scan_size_for_beam_check, offset=offset, offset_margin=offset_margin_for_beam_check)

    def check_ee_vs_env_obstacle_collision(T_candidateEE_global):
        if env_collision_manager is None or not env_collision_manager._objs: return False
        return candidate_collision_check_trimesh(T_candidateEE_global, ee_box_base, env_collision_manager)

    def check_ee_vs_part_collision(T_candidateEE_global, R_part_global_mat):
        if stl_mesh is None or stl_mesh.is_empty or not cm_part_check._objs: return False
        
        ee_box_global = ee_box_base.copy()
        ee_box_global.apply_transform(T_candidateEE_global)
        
        T_part_local_origin_to_global = np.eye(4)
        T_part_local_origin_to_global[:3,:3] = R_part_global_mat
        T_part_local_origin_to_global[:3,3] = part_origin - (R_part_global_mat @ part_center_pivot)
        
        T_global_to_part_local = np.linalg.inv(T_part_local_origin_to_global)
        
        ee_box_in_part_local_frame = ee_box_global.copy()
        ee_box_in_part_local_frame.apply_transform(T_global_to_part_local)
        
        return cm_part_check.in_collision_single(ee_box_in_part_local_frame)

    def check_collisions_with_chuck(T_candidateEE_global, R_part_global_mat):
        if local_chuck_mesh is None or local_chuck_mesh.is_empty:
            return False

        T_chuck_local_origin_to_global = np.eye(4)
        T_chuck_local_origin_to_global[:3,:3] = R_part_global_mat
        T_chuck_local_origin_to_global[:3,3] = part_origin - (R_part_global_mat @ part_center_pivot)

        chuck_global_transformed = local_chuck_mesh.copy()
        chuck_global_transformed.apply_transform(T_chuck_local_origin_to_global)

        cm_chuck_temp = CollisionManager()
        cm_chuck_temp.add_object("chuck_global", chuck_global_transformed)

        ee_box_transformed_globally = ee_box_base.copy()
        ee_box_transformed_globally.apply_transform(T_candidateEE_global)
        if cm_chuck_temp.in_collision_single(ee_box_transformed_globally):
            return True 

        laser_beam_transformed_globally = laser_beam_base.copy()
        laser_beam_transformed_globally.apply_transform(T_candidateEE_global)
        if cm_chuck_temp.in_collision_single(laser_beam_transformed_globally):
            return True 
            
        return False 

    if cleaning_points.shape[0] == 0:
        print("Warning: No cleaning points provided to plan_toolpath.")
        return []

    for local_pt_raw, local_norm_raw in tqdm(zip(cleaning_points, cleaning_normals),
                                       total=len(cleaning_points),
                                       desc="Toolpath planning progress"):
        local_pt = np.asarray(local_pt_raw) 
        local_norm = np.asarray(local_norm_raw)

        best_overall_cost = np.inf
        chosen_EE_pos, chosen_EE_quat = None, None
        chosen_angle_val = previous_angle if previous_angle is not None else starting_angle
        chosen_cleaning_pt_global, chosen_effective_normal_global = None, None
        chosen_classification, chosen_unreachable = 'invalid', True
        chosen_must_pull_back = False
        
        angle_candidates_arr = np.arange(0, 360, coarse_step) + (previous_angle if previous_angle is not None else starting_angle)
        angles_rad_candidates = np.deg2rad(angle_candidates_arr)
        
        R_part_candidates_mats = R.from_rotvec(np.outer(angles_rad_candidates, part_y_axis)).as_matrix()
        
        P_clean_global_candidates = part_origin + np.einsum('ijk,k->ij', R_part_candidates_mats, local_pt - part_center_pivot)
        
        N_eff_global_candidates = np.einsum('ijk,k->ij', R_part_candidates_mats, local_norm)
        EE_pos_global_candidates = P_clean_global_candidates + N_eff_global_candidates * offset

        center_cost_nd = lambda_center * np.abs(EE_pos_global_candidates[:, 2] - part_origin[2]) 
        ang_cost_nd = lambda_angle * np.abs(angle_candidates_arr - (previous_angle if previous_angle is not None else 0.0))
        pitch_cost_nd = lambda_pitch * np.abs(N_eff_global_candidates[:, 2])
        
        norms_P_clean_global = np.linalg.norm(P_clean_global_candidates, axis=1)
        safe_norms_P_clean = np.where(norms_P_clean_global == 0, 1e-6, norms_P_clean_global)
        visibility_metric_nd = -np.einsum('ij,ij->i', P_clean_global_candidates, N_eff_global_candidates) / safe_norms_P_clean
        vis_cost_nd = lambda_vis * np.maximum(0, visibility_threshold - visibility_metric_nd)
        
        total_cost_nd = center_cost_nd + ang_cost_nd + pitch_cost_nd + vis_cost_nd + lambda_deviation

        for i in range(len(angle_candidates_arr)):
            curr_angle_deg, R_part_curr_mat = angle_candidates_arr[i], R_part_candidates_mats[i]
            P_cl_curr, N_eff_curr, EE_pos_curr, cost_curr = \
                P_clean_global_candidates[i], N_eff_global_candidates[i], EE_pos_global_candidates[i], total_cost_nd[i]
            
            valid_nd_candidate = True
            if not (EE_pos_curr[2] >= table_threshold and \
                    (EE_pos_curr[0] < part_origin[0] if part_origin[0] > 0 else EE_pos_curr[0] > part_origin[0]) and \
                    EE_pos_curr[2] >= P_cl_curr[2] and \
                    EE_pos_curr[2] >= (part_origin[2] - 100) and \
                    N_eff_curr[2] >= -0.1 and 
                    np.dot(EE_pos_curr - P_cl_curr, N_eff_curr) >= -1e-3): 
                valid_nd_candidate = False

            if valid_nd_candidate:
                ee_quat_curr = compute_EE_orientation(P_cl_curr, N_eff_curr, roll_angle=0.0)
                T_EE_curr_global = np.eye(4); T_EE_curr_global[:3,:3]=R.from_quat(ee_quat_curr).as_matrix(); T_EE_curr_global[:3,3]=EE_pos_curr
                
                if check_ee_vs_env_obstacle_collision(T_EE_curr_global) or \
                   check_ee_vs_part_collision(T_EE_curr_global, R_part_curr_mat):
                    valid_nd_candidate = False
                
                iter_must_pull_back_nd = False
                if valid_nd_candidate:
                    if check_collisions_with_chuck(T_EE_curr_global, R_part_curr_mat):
                        iter_must_pull_back_nd = True 

                if valid_nd_candidate and previous_EE is not None and not disable_intermediate_collision:
                    for t_interp in np.linspace(0,1,5)[1:-1]:
                        interp_pos = previous_EE*(1-t_interp) + EE_pos_curr*t_interp
                        T_spl = np.eye(4); T_spl[:3,:3]=T_EE_curr_global[:3,:3]; T_spl[:3,3]=interp_pos
                        if check_ee_vs_env_obstacle_collision(T_spl) or \
                           check_ee_vs_part_collision(T_spl, R_part_curr_mat) or \
                           check_collisions_with_chuck(T_spl, R_part_curr_mat): 
                            valid_nd_candidate = False; break
                
                if valid_nd_candidate and cost_curr < best_overall_cost:
                    best_overall_cost = cost_curr
                    chosen_EE_pos, chosen_EE_quat = EE_pos_curr.copy(), ee_quat_curr.copy()
                    chosen_angle_val, chosen_cleaning_pt_global = curr_angle_deg, P_cl_curr.copy()
                    chosen_effective_normal_global, chosen_classification = N_eff_curr.copy(), 'valid'
                    chosen_unreachable, chosen_must_pull_back = False, iter_must_pull_back_nd
        
        run_deviation_search = chosen_unreachable 
        if run_deviation_search:
            deviation_angles_deg = np.array([-15, -10, -5, 5, 10, 15])
            angles_for_dev_search = np.arange(0,360,coarse_step*2) + (previous_angle if previous_angle is not None else starting_angle)

            for part_angle_dev in angles_for_dev_search:
                R_part_dev_mat = R.from_rotvec(np.deg2rad(part_angle_dev)*part_y_axis).as_matrix()
                P_cl_dev = part_origin + R_part_dev_mat @ (local_pt - part_center_pivot)
                N_at_part_angle_dev = R_part_dev_mat @ local_norm

                dev_axis_ref = np.array([0,0,1])
                dev_axis = np.cross(N_at_part_angle_dev, dev_axis_ref)
                if np.linalg.norm(dev_axis) < 1e-6: dev_axis = np.array([1,0,0])
                else: dev_axis /= np.linalg.norm(dev_axis)

                for dev_deg in deviation_angles_deg:
                    R_dev_rot_mat = R.from_rotvec(np.deg2rad(dev_deg)*dev_axis).as_matrix()
                    N_eff_dv = R_dev_rot_mat @ N_at_part_angle_dev
                    if np.linalg.norm(N_eff_dv) < 1e-6: continue
                    N_eff_dv /= np.linalg.norm(N_eff_dv)
                    EE_pos_dv = P_cl_dev + N_eff_dv * offset

                    cost_dv = lambda_center*abs(EE_pos_dv[2]-part_origin[2]) + \
                              lambda_angle*abs(part_angle_dev-(previous_angle if previous_angle is not None else 0)) + \
                              lambda_pitch*abs(N_eff_dv[2])
                    norm_P_cl_dv = np.linalg.norm(P_cl_dev); safe_norm_P_cl_dv = 1e-6 if norm_P_cl_dv==0 else norm_P_cl_dv
                    vis_met_dv = -np.dot(P_cl_dev, N_eff_dv)/safe_norm_P_cl_dv
                    cost_dv += lambda_vis*max(0, visibility_threshold-vis_met_dv)
                    
                    valid_dv_cand = True
                    if not (EE_pos_dv[2]>=table_threshold and \
                            (EE_pos_dv[0]<part_origin[0] if part_origin[0]>0 else EE_pos_dv[0]>part_origin[0]) and \
                            EE_pos_dv[2]>=P_cl_dev[2] and EE_pos_dv[2]>=(part_origin[2]-100) and \
                            N_eff_dv[2]>=-0.1 and np.dot(EE_pos_dv-P_cl_dev, N_eff_dv)>=-1e-3):
                        valid_dv_cand = False
                    
                    if valid_dv_cand:
                        best_roll_dv, valid_orient_found_dv = None, False
                        for roll_cand_deg in np.linspace(-180,180,13):
                            ee_quat_dv_test = compute_EE_orientation(P_cl_dev, N_eff_dv, roll_angle=roll_cand_deg)
                            T_EE_dv_test = np.eye(4); T_EE_dv_test[:3,:3]=R.from_quat(ee_quat_dv_test).as_matrix(); T_EE_dv_test[:3,3]=EE_pos_dv
                            if not check_ee_vs_env_obstacle_collision(T_EE_dv_test) and \
                               not check_ee_vs_part_collision(T_EE_dv_test, R_part_dev_mat):
                                valid_orient_found_dv, best_roll_dv = True, roll_cand_deg; break
                        if not valid_orient_found_dv: valid_dv_cand = False
                        
                        iter_must_pull_back_dv = False
                        if valid_dv_cand:
                            ee_q_dv_final = compute_EE_orientation(P_cl_dev,N_eff_dv,roll_angle=best_roll_dv)
                            T_EE_dv_final = np.eye(4); T_EE_dv_final[:3,:3]=R.from_quat(ee_q_dv_final).as_matrix(); T_EE_dv_final[:3,3]=EE_pos_dv
                            if check_collisions_with_chuck(T_EE_dv_final, R_part_dev_mat):
                                iter_must_pull_back_dv = True
                        
                        if valid_dv_cand and cost_dv < best_overall_cost:
                            best_overall_cost = cost_dv
                            chosen_EE_pos = EE_pos_dv.copy()
                            chosen_EE_quat = compute_EE_orientation(P_cl_dev, N_eff_dv, roll_angle=best_roll_dv).copy()
                            chosen_angle_val, chosen_cleaning_pt_global = part_angle_dev, P_cl_dev.copy()
                            chosen_effective_normal_global, chosen_classification = N_eff_dv.copy(), 'deviated'
                            chosen_unreachable, chosen_must_pull_back = False, iter_must_pull_back_dv

        if chosen_unreachable: 
            R_def = R.from_rotvec(np.deg2rad(chosen_angle_val)*part_y_axis).as_matrix()
            chosen_cleaning_pt_global = part_origin + R_def @ (local_pt - part_center_pivot)
            chosen_effective_normal_global = R_def @ local_norm 

        step_tuple = (chosen_EE_pos, chosen_EE_quat, chosen_angle_val,
                      chosen_cleaning_pt_global, chosen_effective_normal_global,
                      chosen_classification, chosen_unreachable, chosen_must_pull_back)
        steps.append(step_tuple)
        
        if not chosen_unreachable:
            previous_angle = chosen_angle_val
            if chosen_EE_pos is not None: previous_EE = chosen_EE_pos.copy()

    if DEBUG_INTERPOLATE_TOOLPATH:
        steps = interpolate_toolpath_steps(steps, scan_spacing=10.0) 
    
    return steps