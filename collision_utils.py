"""
Collision Utilities Module
--------------------------
This module provides functions for building collision managers and checking for collisions.
"""
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

def build_collision_manager(part_mesh, table_mesh=None, back_wall_mesh=None, ceiling_mesh=None, right_wall_mesh=None):
    """
    Builds a collision manager by adding the part and obstacle meshes.

    Obstacle meshes are added as their convex hulls for efficiency.

    Parameters:
        part_mesh (trimesh.Trimesh): Mesh for the part.
        table_mesh, back_wall_mesh, ceiling_mesh, right_wall_mesh (optional): 
            Meshes for the obstacles.

    Returns:
        trimesh.collision.CollisionManager: Configured collision manager.
    """
    cm = CollisionManager()
    if part_mesh is not None:
        cm.add_object("part", part_mesh)
    if table_mesh is not None:
        cm.add_object("table", table_mesh.convex_hull)
    if back_wall_mesh is not None:
        cm.add_object("back_wall", back_wall_mesh.convex_hull)
    if ceiling_mesh is not None:
        cm.add_object("ceiling", ceiling_mesh.convex_hull)
    if right_wall_mesh is not None:
        cm.add_object("right_wall", right_wall_mesh.convex_hull)
    return cm

def candidate_collision_check_trimesh(EE_pose, ee_box_mesh, collision_manager):
    """
    Checks if a candidate end-effector pose (applied to its collision box) 
    collides with any obstacle in the collision manager.

    Parameters:
        EE_pose (np.ndarray): 4x4 transformation matrix for the candidate EE pose.
        ee_box_mesh (trimesh.Trimesh): The base EE collision box mesh.
        collision_manager (CollisionManager): Collision manager containing obstacles.

    Returns:
        bool: True if collision detected, False otherwise.
    """
    candidate = ee_box_mesh.copy() # Create a copy to avoid modifying the original EE box
    candidate.apply_transform(EE_pose)
    return collision_manager.in_collision_single(candidate)