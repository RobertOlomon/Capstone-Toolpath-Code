"""
Toolpath Animation Module
-------------------------
This module provides functions to animate the planned toolpath using Plotly,
visualizing the part, end-effector, laser beam, scanned points, and chuck.
The animation can be split into two temporal halves for performance.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import plotly.graph_objects as go

from mesh_utils import get_laser_beam_mesh, get_ee_box_mesh 
from scan_utils import classify_step_scanned_points
from collision_utils import candidate_collision_check_trimesh


def animate_toolpath(toolpath, stl_mesh, part_origin, part_y_axis, part_center,
                     local_chuck_mesh=None,
                     chuck_pullback_distance=500.0,
                     offset=200, axis_length=400, scan_size=25, frame_duration=50,
                     ee_box_extents=[-140, 444.5, -130, 280, -257, 88],
                     table_mesh=None, back_wall_mesh=None, ceiling_mesh=None, right_wall_mesh=None,
                     offset_margin=5, display_animation=True, collision_manager=None,
                     debug_obstacles_only=False,
                     split_animation_halves=True,
                     return_figures=False):
    """
    Animates the toolpath or displays a static scene.
    If split_animation_halves is True, the animation is divided into two figures
    showing the first and second temporal halves of the toolpath.
    A final plot shows cumulative scan coverage.
    """
    # --- Shared Setup for Bounds and Debug View ---
    # ... (Bounds calculation and debug_obstacles_only block remains the same as before) ...
    all_meshes_for_bounds = []
    if stl_mesh and not stl_mesh.is_empty:
        all_meshes_for_bounds.append(stl_mesh)

    R_part_initial_bounds = R.from_rotvec(part_y_axis * np.deg2rad(0))
    if local_chuck_mesh and not local_chuck_mesh.is_empty:
        chuck_vertices_global_bounds = part_origin + R_part_initial_bounds.apply(local_chuck_mesh.vertices - part_center)
        temp_chuck_for_bounds = trimesh.Trimesh(vertices=chuck_vertices_global_bounds, faces=local_chuck_mesh.faces, process=False)
        if not temp_chuck_for_bounds.is_empty:
            all_meshes_for_bounds.append(temp_chuck_for_bounds)
    
    initial_scene_meshes = []
    if stl_mesh and not stl_mesh.is_empty:
        initial_part_verts = part_origin + R_part_initial_bounds.apply(stl_mesh.vertices - part_center)
        initial_scene_meshes.append(trimesh.Trimesh(vertices=initial_part_verts, faces=stl_mesh.faces, process=False))
    if local_chuck_mesh and not local_chuck_mesh.is_empty:
         initial_chuck_verts = part_origin + R_part_initial_bounds.apply(local_chuck_mesh.vertices - part_center)
         initial_scene_meshes.append(trimesh.Trimesh(vertices=initial_chuck_verts, faces=local_chuck_mesh.faces, process=False))

    static_obstacles = [m for m in [table_mesh, back_wall_mesh, ceiling_mesh, right_wall_mesh] if m is not None and not m.is_empty]
    all_meshes_for_bounds_calc = initial_scene_meshes + static_obstacles

    if not all_meshes_for_bounds_calc:
        combined_bounds_min = np.array([-500,-500,-500])
        combined_bounds_max = np.array([500,500,500])
    else:
        combined_mesh_for_bounds = trimesh.util.concatenate(all_meshes_for_bounds_calc)
        if combined_mesh_for_bounds.is_empty:
            combined_bounds_min = np.array([-500,-500,-500])
            combined_bounds_max = np.array([500,500,500])
        else:
            combined_bounds_min = combined_mesh_for_bounds.bounds[0]
            combined_bounds_max = combined_mesh_for_bounds.bounds[1]

    scene_center = (combined_bounds_min + combined_bounds_max) / 2.0
    scene_diag_raw = np.linalg.norm(combined_bounds_max - combined_bounds_min)
    scene_diag = scene_diag_raw if scene_diag_raw > 1e-3 else 1000.0 

    margin_bounds = scene_diag * 0.1 + 200 
    bound_range_val = scene_diag / 2.0 + margin_bounds
    
    scene_range_settings = dict(
        xaxis=dict(range=[scene_center[0] - bound_range_val, scene_center[0] + bound_range_val]),
        yaxis=dict(range=[scene_center[1] - bound_range_val, scene_center[1] + bound_range_val]),
        zaxis=dict(range=[scene_center[2] - bound_range_val, scene_center[2] + bound_range_val]),
        aspectmode='cube' 
    )

    if debug_obstacles_only:
        # ... (debug_obstacles_only block remains the same) ...
        debug_frame_data = []
        R_part_initial_debug = R.from_rotvec(part_y_axis * np.deg2rad(0)) 
        
        if stl_mesh and not stl_mesh.is_empty:
            global_part_vertices_debug = part_origin + R_part_initial_debug.apply(stl_mesh.vertices - part_center)
            debug_frame_data.append(go.Mesh3d(
                x=global_part_vertices_debug[:, 0], y=global_part_vertices_debug[:, 1], z=global_part_vertices_debug[:, 2],
                i=stl_mesh.faces[:, 0], j=stl_mesh.faces[:, 1], k=stl_mesh.faces[:, 2],
                color='lightgrey', opacity=1, name='STL Part'
            ))

        if local_chuck_mesh is not None and not local_chuck_mesh.is_empty:
            global_chuck_vertices_debug = part_origin + R_part_initial_debug.apply(local_chuck_mesh.vertices - part_center)
            debug_frame_data.append(go.Mesh3d(
                x=global_chuck_vertices_debug[:, 0], y=global_chuck_vertices_debug[:, 1], z=global_chuck_vertices_debug[:, 2],
                i=local_chuck_mesh.faces[:, 0], j=local_chuck_mesh.faces[:, 1], k=local_chuck_mesh.faces[:, 2],
                color='slategray', opacity=0.5, name='Chuck' 
            ))

        if table_mesh is not None and not table_mesh.is_empty:
            debug_frame_data.append(go.Mesh3d(
                x=table_mesh.vertices[:, 0], y=table_mesh.vertices[:, 1], z=table_mesh.vertices[:, 2],
                i=table_mesh.faces[:, 0], j=table_mesh.faces[:, 1], k=table_mesh.faces[:, 2],
                color='brown', opacity=1, name='Table'
            ))
        if back_wall_mesh is not None and not back_wall_mesh.is_empty:
            debug_frame_data.append(go.Mesh3d(
                x=back_wall_mesh.vertices[:, 0], y=back_wall_mesh.vertices[:, 1], z=back_wall_mesh.vertices[:, 2],
                i=back_wall_mesh.faces[:, 0], j=back_wall_mesh.faces[:, 1], k=back_wall_mesh.faces[:, 2],
                color='grey', opacity=1, name='Back Wall'
            ))
        # ... (add other static obstacles similarly) ...
        
        fig_debug = go.Figure(
            data=debug_frame_data if debug_frame_data else None, 
            layout=go.Layout(title="Debug: Obstacles, Part, and Chuck Mesh", scene=scene_range_settings)
        )
        if display_animation:
            fig_debug.show()
        return

    # --- Scan Coverage Simulation (shared for final plot) ---
    # ... (this block remains the same) ...
    dense_pts_local_anim = np.empty((0,3))
    dense_normals_anim = np.empty((0,3))
    N_dense_anim = 0
    scanned_count_cumulative = np.empty(0, dtype=int)

    if stl_mesh and not stl_mesh.is_empty:
        dense_pts_local_anim_temp, dense_face_indices_anim = trimesh.sample.sample_surface(stl_mesh, 50000)
        if dense_pts_local_anim_temp.shape[0] > 0:
            dense_pts_local_anim = dense_pts_local_anim_temp
            dense_normals_anim = stl_mesh.face_normals[dense_face_indices_anim]
            N_dense_anim = dense_pts_local_anim.shape[0]
            scanned_count_cumulative = np.zeros(N_dense_anim, dtype=int)
    
    if toolpath and N_dense_anim > 0: 
        for step_data_for_scan_sim in toolpath:
            classify_step_scanned_points(step_data_for_scan_sim[:7], scanned_count_cumulative, 
                                         dense_pts_local_anim, dense_normals_anim,
                                         part_origin, part_y_axis, part_center,
                                         scan_size, offset, offset_margin)

    # --- Frame Generation (Single list for all frames initially) ---
    all_animation_frames = []

    if stl_mesh and not stl_mesh.is_empty:
        part_local_vertices = stl_mesh.vertices
        part_faces = stl_mesh.faces
        part_i_arr, part_j_arr, part_k_arr = part_faces[:, 0], part_faces[:, 1], part_faces[:, 2]
    else:
        part_local_vertices = np.empty((0,3)); part_faces = np.empty((0,3)); part_i_arr, part_j_arr, part_k_arr = [],[],[]

    base_ee_mesh_viz = get_ee_box_mesh(ee_box_extents) 
    base_laser_beam_mesh_viz = get_laser_beam_mesh(scan_size, offset=offset, offset_margin=offset_margin)

    for idx, current_step in enumerate(toolpath):
        pos, quat, angle, cleaning_point, chosen_normal, classification, unreachable = current_step[:7]
        must_pull_back_flag = current_step[7]
        new_points_in_step = current_step[9]
        double_points_in_step = current_step[10]
            
        single_frame_data_all_elements = [] 

        R_part_current_obj = R.from_rotvec(part_y_axis * np.deg2rad(angle))
        
        # Part Mesh
        if stl_mesh and not stl_mesh.is_empty:
            global_part_vertices_current = part_origin + R_part_current_obj.apply(part_local_vertices - part_center)
            single_frame_data_all_elements.append(go.Mesh3d(
                x=global_part_vertices_current[:, 0], y=global_part_vertices_current[:, 1], z=global_part_vertices_current[:, 2],
                i=part_i_arr, j=part_j_arr, k=part_k_arr, color='lightgrey', opacity=1, name='STL Part'
            ))

        # Chuck Mesh
        if local_chuck_mesh is not None and not local_chuck_mesh.is_empty:
            chuck_vertices_rotated = R_part_current_obj.apply(local_chuck_mesh.vertices - part_center)
            chuck_vertices_global = part_origin + chuck_vertices_rotated
            current_display_chuck_mesh_data = trimesh.Trimesh(vertices=chuck_vertices_global, faces=local_chuck_mesh.faces, process=False)
            if must_pull_back_flag:
                part_neg_y_local = np.array([0, -1, 0])
                part_neg_y_global = R_part_current_obj.apply(part_neg_y_local)
                pullback_translation_vec = part_neg_y_global * chuck_pullback_distance
                current_display_chuck_mesh_data.apply_translation(pullback_translation_vec)
            single_frame_data_all_elements.append(go.Mesh3d(
                x=current_display_chuck_mesh_data.vertices[:, 0], y=current_display_chuck_mesh_data.vertices[:, 1], z=current_display_chuck_mesh_data.vertices[:, 2],
                i=current_display_chuck_mesh_data.faces[:, 0], j=current_display_chuck_mesh_data.faces[:, 1], k=current_display_chuck_mesh_data.faces[:, 2],
                color='slategray', opacity=0.5, name='Chuck'
            ))

        # Static Obstacles
        if table_mesh is not None and not table_mesh.is_empty: single_frame_data_all_elements.append(go.Mesh3d(x=table_mesh.vertices[:,0], y=table_mesh.vertices[:,1], z=table_mesh.vertices[:,2], i=table_mesh.faces[:,0], j=table_mesh.faces[:,1], k=table_mesh.faces[:,2], color='brown', opacity=1, name='Table'))
        if back_wall_mesh is not None and not back_wall_mesh.is_empty: single_frame_data_all_elements.append(go.Mesh3d(x=back_wall_mesh.vertices[:,0], y=back_wall_mesh.vertices[:,1], z=back_wall_mesh.vertices[:,2], i=back_wall_mesh.faces[:,0], j=back_wall_mesh.faces[:,1], k=back_wall_mesh.faces[:,2], color='grey', opacity=1, name='Back Wall'))
        # ... add other static obstacles 

        if not unreachable and pos is not None and quat is not None: 
            T_EE_current = np.eye(4)
            T_EE_current[:3, :3] = R.from_quat(quat).as_matrix()
            T_EE_current[:3, 3] = pos
            
            # EE Coordinate Axes
            R_ee_current_mat = T_EE_current[:3, :3] 
            x_axis_end = pos + R_ee_current_mat[:, 0] * axis_length; y_axis_end = pos + R_ee_current_mat[:, 1] * axis_length; z_axis_end = pos + R_ee_current_mat[:, 2] * axis_length
            single_frame_data_all_elements.append(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(color='black', size=4), name='EE Position'))
            single_frame_data_all_elements.append(go.Scatter3d(x=[pos[0], x_axis_end[0]], y=[pos[1], x_axis_end[1]], z=[pos[2], x_axis_end[2]], mode='lines', line=dict(color='cyan', width=6), name='EE X'))
            single_frame_data_all_elements.append(go.Scatter3d(x=[pos[0], y_axis_end[0]], y=[pos[1], y_axis_end[1]], z=[pos[2], y_axis_end[2]], mode='lines', line=dict(color='magenta', width=6), name='EE Y'))
            single_frame_data_all_elements.append(go.Scatter3d(x=[pos[0], z_axis_end[0]], y=[pos[1], z_axis_end[1]], z=[pos[2], z_axis_end[2]], mode='lines', line=dict(color='blue', width=6), name='EE Z'))
            
            # EE Collision Mesh
            is_colliding_current = False
            if collision_manager is not None and collision_manager._objs : 
                is_colliding_current = candidate_collision_check_trimesh(T_EE_current, base_ee_mesh_viz, collision_manager)
            transformed_ee_mesh = base_ee_mesh_viz.copy(); transformed_ee_mesh.apply_transform(T_EE_current)
            ee_mesh_color = 'red' if is_colliding_current else 'green' 
            single_frame_data_all_elements.append(go.Mesh3d(x=transformed_ee_mesh.vertices[:,0],y=transformed_ee_mesh.vertices[:,1],z=transformed_ee_mesh.vertices[:,2],i=transformed_ee_mesh.faces[:,0],j=transformed_ee_mesh.faces[:,1],k=transformed_ee_mesh.faces[:,2],color=ee_mesh_color,opacity=0.5,name='EE Box'))
            
            # Laser Beam
            transformed_beam_mesh = base_laser_beam_mesh_viz.copy(); transformed_beam_mesh.apply_transform(T_EE_current)
            single_frame_data_all_elements.append(go.Mesh3d(x=transformed_beam_mesh.vertices[:,0],y=transformed_beam_mesh.vertices[:,1],z=transformed_beam_mesh.vertices[:,2],i=transformed_beam_mesh.faces[:,0],j=transformed_beam_mesh.faces[:,1],k=transformed_beam_mesh.faces[:,2],color='orange',opacity=0.8,name='Laser Beam'))

            # Scanned Points
            if new_points_in_step.shape[0] > 0:
                global_new_points = part_origin + R_part_current_obj.apply(new_points_in_step - part_center)
                single_frame_data_all_elements.append(go.Scatter3d(x=global_new_points[:,0],y=global_new_points[:,1],z=global_new_points[:,2],mode='markers',marker=dict(color='blue',size=4),name='New Points'))
            if double_points_in_step.shape[0] > 0:
                global_double_points = part_origin + R_part_current_obj.apply(double_points_in_step - part_center)
                single_frame_data_all_elements.append(go.Scatter3d(x=global_double_points[:,0],y=global_double_points[:,1],z=global_double_points[:,2],mode='markers',marker=dict(color='red',size=4),name='Rescanned'))
        
        elif cleaning_point is not None: 
             single_frame_data_all_elements.append(go.Scatter3d(
                x=[cleaning_point[0]], y=[cleaning_point[1]], z=[cleaning_point[2]],
                mode='markers', marker=dict(color='purple', size=12, symbol='cross'), name='Unreachable Target'
            ))

        frame_title_suffix = " (Chuck Pulled Back)" if must_pull_back_flag else ""
        all_animation_frames.append(go.Frame(
            data=single_frame_data_all_elements, name=f"Step {idx+1}",
            layout=go.Layout(title_text=f"Step {idx+1}: Rot = {angle:.1f}°{frame_title_suffix}")
        ))
    
    # --- Configure and Display Animation(s) ---
    num_total_frames = len(all_animation_frames)

    def create_animation_figure(frames_subset, title_prefix, frame_offset=0):
        if not frames_subset:
            return None
            
            slider_ctrl_steps = []
            for i, anim_frame in enumerate(frames_subset):
                # Frame name needs to be unique if figures share the same JS context,
                # but for separate figures, original names are fine.
                # Label should reflect the actual step number in the overall toolpath.
                actual_step_number = frame_offset + i + 1
                slider_ctrl_steps.append(dict(
                    method="animate", 
                    args=[[anim_frame.name], {"mode": "immediate", "frame": {"duration": frame_duration, "redraw": True}, "transition": {"duration": 0}}], 
                    label=f"{actual_step_number}"
                ))
            
            sliders = [dict(active=0, currentvalue={"prefix": "Frame: "}, pad={"t": 50}, steps=slider_ctrl_steps)]
            updatemenus = [dict(type="buttons", showactive=False, buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])])]
            
            # Initial data for the figure is the data of the first frame in the subset
            initial_data = frames_subset[0].data if frames_subset[0].data else None # Handle if first frame is empty
            # The 'frames' argument to Figure should be the rest of the frames in the subset
            figure_frames = frames_subset # Plotly handles the first frame from initial_data

            fig = go.Figure(
                data=initial_data, 
                layout=go.Layout(title=f"{title_prefix} (Frames {frame_offset+1}-{frame_offset+len(frames_subset)})", 
                                 scene=scene_range_settings, updatemenus=updatemenus, sliders=sliders),
                frames=figure_frames 
            )
            return fig

    figures = []
    if split_animation_halves and num_total_frames > 1:
            mid_point = num_total_frames // 2
            frames_part1 = all_animation_frames[:mid_point]
            frames_part2 = all_animation_frames[mid_point:]

            if frames_part1:
                fig1 = create_animation_figure(frames_part1, "Toolpath Animation - Part 1", frame_offset=0)
                if fig1:
                    if display_animation:
                        fig1.show()
                    if return_figures:
                        figures.append(fig1)

            if frames_part2:
                fig2 = create_animation_figure(frames_part2, "Toolpath Animation - Part 2", frame_offset=mid_point)
                if fig2:
                    if display_animation:
                        fig2.show()
                    if return_figures:
                        figures.append(fig2)
    else: # Show as a single animation
            fig_single = create_animation_figure(all_animation_frames, "Toolpath Animation", frame_offset=0)
            if fig_single:
                if display_animation:
                    fig_single.show()
                if return_figures:
                    figures.append(fig_single)
            
    # --- Display Final Scan Coverage Figure (remains the same) ---
    # ... (this block remains the same as the previous full version) ...
    if display_animation and toolpath and N_dense_anim > 0:
        final_ref_angle = toolpath[-1][2]
        R_final_ref = R.from_rotvec(part_y_axis * np.deg2rad(final_ref_angle))
        global_dense_pts_final_vis = part_origin + R_final_ref.apply(dense_pts_local_anim - part_center)
        
        unscanned_global_pts = global_dense_pts_final_vis[scanned_count_cumulative == 0]
        single_scanned_global_pts = global_dense_pts_final_vis[scanned_count_cumulative == 1]
        double_scanned_global_pts = global_dense_pts_final_vis[scanned_count_cumulative > 1]
        
        final_cloud_data = []
        if unscanned_global_pts.shape[0] > 0: final_cloud_data.append(go.Scatter3d(x=unscanned_global_pts[:,0], y=unscanned_global_pts[:,1], z=unscanned_global_pts[:,2], mode='markers', marker=dict(color='grey', size=3), name='Unscanned'))
        if single_scanned_global_pts.shape[0] > 0: final_cloud_data.append(go.Scatter3d(x=single_scanned_global_pts[:,0], y=single_scanned_global_pts[:,1], z=single_scanned_global_pts[:,2], mode='markers', marker=dict(color='blue', size=3), name='Scanned (1x)'))
        if double_scanned_global_pts.shape[0] > 0: final_cloud_data.append(go.Scatter3d(x=double_scanned_global_pts[:,0], y=double_scanned_global_pts[:,1], z=double_scanned_global_pts[:,2], mode='markers', marker=dict(color='red', size=3), name='Scanned (2x+)'))

        if final_cloud_data:
            fig_final_cloud = go.Figure(data=final_cloud_data)
            final_scene_layout = scene_range_settings.copy(); final_scene_layout['aspectmode'] = 'data'
            fig_final_cloud.update_layout(title="Final Dense Point Cloud Scan Coverage", scene=final_scene_layout)
            if display_animation:
                fig_final_cloud.show()
            if return_figures:
                figures.append(fig_final_cloud)
    elif display_animation and toolpath and N_dense_anim == 0:
        if stl_mesh and not stl_mesh.is_empty:
            print("Animation: No dense points were sampled from the part, skipping final scan coverage plot.")

    return figures
