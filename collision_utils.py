"""
Collision Utilities Module
--------------------------
This module provides functions for building collision managers and checking for collisions.
"""
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

def build_collision_manager(part_mesh, table_mesh=None, back_wall_mesh=None, ceiling_mesh=None, right_wall_mesh=None):
    """@brief Build a collision manager with optional obstacles.

    Obstacle meshes are added using their convex hulls.

    @param part_mesh Mesh of the part.
    @param table_mesh Optional table mesh.
    @param back_wall_mesh Optional back wall mesh.
    @param ceiling_mesh Optional ceiling mesh.
    @param right_wall_mesh Optional right wall mesh.

    @return ``CollisionManager`` with all objects added.
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
    """@brief Check if an EE pose collides with any obstacle.

    This function transforms ``ee_box_mesh`` to ``EE_pose`` and queries
    ``collision_manager`` to determine if the candidate end effector would
    intersect any objects.

    @param EE_pose        4x4 pose matrix for the candidate EE.
    @param ee_box_mesh    Base EE collision box mesh.
    @param collision_manager Collision manager with obstacles.

    @return ``True`` if a collision is detected.
    """
    candidate = ee_box_mesh.copy()
    candidate.apply_transform(EE_pose)
    return collision_manager.in_collision_single(candidate)
