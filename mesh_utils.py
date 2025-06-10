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
    """@brief Create a box mesh.

    Builds a convex hull from the eight corner points defined by ``extents``
    and applies ``pose`` to transform the mesh.

    @param extents List ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    @param pose    4x4 transform applied to the box.

    @return Transformed ``trimesh.Trimesh`` object.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = extents
    vertices = np.array([[x, y, z] for x in (x_min, x_max)
                                    for y in (y_min, y_max)
                                    for z in (z_min, z_max)])
    box = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    box.apply_transform(pose)
    return box

def get_ee_box_mesh(ee_box_extents):
    """@brief Generate the EE collision box mesh.

    Convenience wrapper around ``create_box_mesh`` that does not apply any
    transform.

    @param ee_box_extents List ``[x_min, x_max, y_min, y_max, z_min, z_max]``.

    @return ``trimesh.Trimesh`` representing the EE box.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = ee_box_extents
    vertices = np.array([[x, y, z] for x in (x_min, x_max)
                                    for y in (y_min, y_max)
                                    for z in (z_min, z_max)])
    ee_box = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    return ee_box

def get_laser_beam_mesh(scan_size, offset=200, offset_margin=5):
    """@brief Generate the laser beam volume mesh.

    Creates a rectangular prism representing the laser scanning volume
    located ``offset`` millimeters from the part.

    @param scan_size Size of the scanning square in X and Y.
    @param offset Nominal EE to part distance.
    @param offset_margin Depth tolerance of the scan volume.

    @return ``trimesh.Trimesh`` for the laser beam.
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
    """@brief Create a cylinder mesh aligned with the Z-axis.

    The cylinder is generated along the Z-axis and then transformed by ``pose``.

    @param radius   Cylinder radius.
    @param height   Cylinder height.
    @param sections Number of facets.
    @param pose     4x4 transform applied to the cylinder.

    @return Transformed ``trimesh.Trimesh`` cylinder.
    """
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cyl.apply_transform(pose)
    return cyl

def get_chuck_mesh(part_mesh, diameter, length_into_part, length_away_from_part):
    """@brief Create the chuck collision mesh.

    Models a simple cylindrical chuck aligned with the part's local Y-axis.

    The chuck is aligned with the part's local Y-axis and centered on the front
    Y-plane.

    @param part_mesh            Mesh of the part.
    @param diameter             Chuck diameter.
    @param length_into_part     Length extending toward ``-Y``.
    @param length_away_from_part Length extending toward ``+Y``.

    @return ``trimesh.Trimesh`` for the chuck or an empty mesh when dimensions
    are invalid.
    """
    if part_mesh is None or part_mesh.is_empty:
        return trimesh.Trimesh()

    radius = diameter / 2.0
    total_chuck_length = length_into_part + length_away_from_part

    if total_chuck_length <= 1e-6:
        return trimesh.Trimesh()

    part_bounds = part_mesh.bounds
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
