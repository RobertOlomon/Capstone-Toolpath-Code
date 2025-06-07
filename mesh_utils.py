"""
Mesh Utilities Module
---------------------
This module contains functions to create and manipulate mesh objects,
primarily for collision geometry and visualization aids like laser beams.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

def create_box_mesh(extents, pose=np.eye(4)):
    """
    Creates a convex-hull box mesh from given extents and pose.

    Parameters:
        extents (list): [x_min, x_max, y_min, y_max, z_min, z_max].
        pose (np.ndarray): 4x4 transformation matrix to apply to the box (default identity).

    Returns:
        trimesh.Trimesh: Transformed box mesh.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = extents
    vertices = np.array([[x, y, z] for x in (x_min, x_max)
                                    for y in (y_min, y_max)
                                    for z in (z_min, z_max)])
    # Create a Trimesh object from vertices and compute its convex hull
    box = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    box.apply_transform(pose)
    return box

def get_ee_box_mesh(ee_box_extents):
    """
    Generates the end-effector (EE) collision box mesh based on extents.
    The mesh is created at the origin, without any transformation.

    Parameters:
        ee_box_extents (list): [x_min, x_max, y_min, y_max, z_min, z_max].

    Returns:
        trimesh.Trimesh: EE collision box mesh.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = ee_box_extents
    vertices = np.array([[x, y, z] for x in (x_min, x_max)
                                    for y in (y_min, y_max)
                                    for z in (z_min, z_max)])
    ee_box = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    return ee_box

def get_laser_beam_mesh(scan_size, offset=200, offset_margin=5):
    """
    Generates a mesh representing the laser beam (scanning volume).
    The mesh is created at the origin, assuming the EE's Z-axis points towards the part.

    Parameters:
        scan_size (float): Size of the scanning square (X and Y dimensions).
        offset (float): Nominal offset distance from the EE origin to the part surface (along Z).
        offset_margin (float): Tolerance in the offset distance, defining the depth of the scan volume.

    Returns:
        trimesh.Trimesh: Mesh for the laser beam volume.
    """
    x_min = -scan_size / 2
    x_max = scan_size / 2
    y_min = -scan_size / 2
    y_max = scan_size / 2
    z_min = offset - offset_margin 
    z_max = offset + offset_margin 
    extents = [x_min, x_max, y_min, y_max, z_min, z_max]
    return create_box_mesh(extents, np.eye(4))

def create_cylinder_mesh(radius, height, sections=32, pose=np.eye(4)):
    """
    Creates a cylinder mesh.
    The cylinder is created with its axis along Z.

    Parameters:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        sections (int): Number of facets.
        pose (np.ndarray): 4x4 transformation matrix to apply.

    Returns:
        trimesh.Trimesh: Transformed cylinder mesh.
    """
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cyl.apply_transform(pose)
    return cyl

def get_chuck_mesh(part_mesh, diameter, length_into_part, length_away_from_part):
    """
    Creates the chuck collision mesh, positioned relative to the part's front face
    and aligned with the part's local Y-axis.

    The chuck is a cylinder. Its local Y-axis aligns with the part's local Y-axis.
    Its center is on the part's front Y-plane, at the part's XZ centroid of overall bounds.

    Parameters:
        part_mesh (trimesh.Trimesh): The mesh of the part. Used for positioning.
        diameter (float): Diameter of the chuck cylinder.
        length_into_part (float): Length the chuck extends from front face towards part's -Y.
        length_away_from_part (float): Length the chuck extends from front face towards part's +Y.

    Returns:
        trimesh.Trimesh: The chuck mesh in the part's local coordinate system.
                         Returns an empty Trimesh if part_mesh is empty or total length is zero.
    """
    if part_mesh is None or part_mesh.is_empty:
        # print("Warning: part_mesh is None or empty in get_chuck_mesh. Returning empty chuck.")
        return trimesh.Trimesh()

    radius = diameter / 2.0
    total_chuck_length = length_into_part + length_away_from_part

    if total_chuck_length <= 1e-6: 
        # print("Warning: Chuck total length is near zero. Returning empty chuck.")
        return trimesh.Trimesh()

    part_bounds = part_mesh.bounds
    # INSTEAD OF part_xz_centroid_of_bounds:
    part_local_geometric_centroid = part_mesh.centroid 

    mesh_local_front_y_coord = part_bounds[1, 1] 
    y_center_of_chuck_cylinder_local = mesh_local_front_y_coord + (length_away_from_part - length_into_part) / 2.0

    chuck_center_local_coords = np.array([
        part_local_geometric_centroid[0], # Use X from geometric centroid
        y_center_of_chuck_cylinder_local, 
        part_local_geometric_centroid[2]  # Use Z from geometric centroid
    ])
    # Create a base cylinder (trimesh creates it along Z-axis, centered at origin [0,0,0])
    base_cylinder = trimesh.creation.cylinder(radius=radius, height=total_chuck_length, sections=32)

    # Transformation to align and position the chuck:
    # 1. Rotate base cylinder (Z-aligned height) so its height aligns with part's Y-axis.
    #    A rotation of +90 degrees around the X-axis maps Z-axis to Y-axis.
    T_align_axis_to_Y = np.eye(4)
    T_align_axis_to_Y[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
    
    # 2. Translate the rotated cylinder to `chuck_center_local_coords`.
    T_translate_to_pos = np.eye(4)
    T_translate_to_pos[:3, 3] = chuck_center_local_coords
    
    final_transform_local = T_translate_to_pos @ T_align_axis_to_Y
    
    chuck_local = base_cylinder.copy()
    chuck_local.apply_transform(final_transform_local)
    
    return chuck_local