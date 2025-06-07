"""
Grid Pattern Generation Module
------------------------------
This module implements various algorithms to generate scan points and 
associated normals from an STL mesh. Methods include:
  - Farthest point sampling.
  - Tiling of cylindrical and planar regions.
  - Region growing to detect flat surfaces.
  - Poisson disc sampling for uniform coverage.
  - A hybrid approach to combine non-feature, cylindrical, and planar regions.

It also includes a visualization function to display detected flat surfaces.
"""

import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.spatial.transform import Rotation as R
import trimesh
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN


# =============================================================================
# Farthest Point Sampling
# =============================================================================
def farthest_point_sampling(pts, r):
    """@brief Farthest point sampling of a point set.

    Iteratively picks the point farthest from those already selected until all
    remaining points are within ``r`` of a chosen sample.

    @param pts Array of points ``(N x 3)``.
    @param r   Distance threshold.

    @return Indices of the selected points.
    """
    pts = np.array(pts)
    N = pts.shape[0]
    if N == 0:
        return []
    selected = [0]
    distances = np.linalg.norm(pts - pts[0], axis=1)
    while np.max(distances) > r:
        next_index = np.argmax(distances)
        selected.append(next_index)
        new_dists = np.linalg.norm(pts - pts[next_index], axis=1)
        distances = np.minimum(distances, new_dists)
    return selected


# =============================================================================
# Vertex Curvature Estimation
# =============================================================================
def compute_vertex_curvature(mesh, k=20):
    """@brief Estimate per-vertex curvature from normal variation.

    For each vertex the normals of its ``k`` nearest neighbours are compared
    with the vertex normal to quantify how much the surface bends.

    @param mesh Mesh to evaluate.
    @param k    Number of nearest neighbors.

    @return Array of curvature values.
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    tree = cKDTree(vertices)
    curvatures = np.zeros(len(vertices))
    for i, (v, n) in enumerate(zip(vertices, normals)):
        dists, idx = tree.query(v, k=k)
        neighbor_normals = normals[idx]
        curvatures[i] = 1 - np.mean(np.dot(neighbor_normals, n))
    return curvatures


# =============================================================================
# Cylindrical Region Tiling
# =============================================================================
def segment_cylindrical_region(mesh, curvature_threshold=0.1):
    """@brief Identify cylindrical vertices via curvature.

    Vertices with curvature below ``curvature_threshold`` are labelled as
    belonging to cylindrical areas of the mesh.

    @param mesh Mesh to evaluate.
    @param curvature_threshold Maximum curvature value considered cylindrical.

    @return Boolean mask for cylindrical vertices.
    """
    curvatures = compute_vertex_curvature(mesh, k=20)
    return curvatures < curvature_threshold


def tile_cylindrical_region_with_ends(mesh, scan_size):
    """@brief Tile a cylindrical region including its ends.

    Estimates the cylinder axis and radius and returns evenly spaced points
    around the lateral surface as well as on the end caps.

    @param mesh Mesh object.
    @param scan_size Desired grid spacing.

    @return ``lateral_pts`` and ``lateral_normals`` arrays.
    """
    cyl_mask = segment_cylindrical_region(mesh, curvature_threshold=0.1)
    if not np.any(cyl_mask):
        return np.empty((0, 3)), np.empty((0, 3))
    
    cyl_vertices = mesh.vertices[cyl_mask]
    cyl_normals = mesh.vertex_normals[cyl_mask]
    center_est = np.mean(cyl_vertices, axis=0)
    pts_centered = cyl_vertices - center_est
    U, S, Vt = np.linalg.svd(pts_centered)
    axis_est = Vt[0]
    axis_est = axis_est / np.linalg.norm(axis_est)
    
    proj = cyl_vertices - np.outer(np.dot(pts_centered, axis_est), axis_est)
    dists = np.linalg.norm(proj - center_est, axis=1)
    radius_est = np.median(dists)
    
    z_vals = np.dot(pts_centered, axis_est)
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    height = z_max - z_min

    n_theta = int(np.ceil(2 * np.pi * radius_est / scan_size))
    n_z = int(np.ceil(height / scan_size))
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(z_min, z_max, n_z)
    
    lateral_pts = []
    lateral_normals = []
    # Determine orthogonal vectors to the estimated axis.
    if np.abs(np.dot(axis_est, np.array([0, 0, 1]))) > 0.99:
        v = np.cross(np.array([0, 1, 0]), axis_est)
    else:
        v = np.cross(np.array([0, 0, 1]), axis_est)
    v = v / np.linalg.norm(v)
    w = np.cross(axis_est, v)
    w = w / np.linalg.norm(w)
    
    for theta in thetas:
        for z in zs:
            pt = center_est + radius_est * (np.cos(theta) * v + np.sin(theta) * w) + z * axis_est
            normal = np.cos(theta) * v + np.sin(theta) * w
            lateral_pts.append(pt)
            lateral_normals.append(normal)
    
    return np.array(lateral_pts), np.array(lateral_normals)


# =============================================================================
# Planar Region Tiling
# =============================================================================
def tile_planar_region(region_pts, scan_size):
    """@brief Tile a planar region with a grid.

    Fits a plane to ``region_pts`` and generates a square grid of points and
    outward normals covering the entire region.

    @param region_pts Points belonging to the planar region.
    @param scan_size  Grid spacing.

    @return ``plane_pts`` and ``plane_normals`` arrays.
    """
    center = np.mean(region_pts, axis=0)
    U, S, Vt = np.linalg.svd(region_pts - center)
    plane_normal = Vt[-1]
    tangent1 = Vt[0]
    tangent2 = np.cross(plane_normal, tangent1)
    tangent2 /= np.linalg.norm(tangent2)
    pts_2d = np.column_stack((
        np.dot(region_pts - center, tangent1),
        np.dot(region_pts - center, tangent2)
    ))
    min_xy = pts_2d.min(axis=0)
    max_xy = pts_2d.max(axis=0)
    xs = np.arange(min_xy[0], max_xy[0] + scan_size, scan_size)
    ys = np.arange(min_xy[1], max_xy[1] + scan_size, scan_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_pts_2d = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    plane_pts = center + np.outer(grid_pts_2d[:, 0], tangent1) + np.outer(grid_pts_2d[:, 1], tangent2)
    plane_normals = np.tile(plane_normal, (plane_pts.shape[0], 1))
    return plane_pts, plane_normals


# =============================================================================
# Planar Face Region Growing and Detection
# =============================================================================
def region_grow_planar_faces(mesh, normal_threshold=np.deg2rad(5), min_region_size=50):
    """@brief Grow planar face regions based on normal similarity.

    Starting from each unvisited face, neighboring faces with similar
    orientation are iteratively added to form planar patches.

    @param mesh Mesh object.
    @param normal_threshold Maximum allowed angular difference in radians.
    @param min_region_size Minimum number of faces for a region.

    @return List of regions (each a list of face indices).
    """
    face_normals = mesh.face_normals
    n_faces = len(mesh.faces)
    visited = np.zeros(n_faces, dtype=bool)
    regions = []
    
    # Use mesh face adjacency if available; otherwise, build it.
    if hasattr(mesh, 'face_adjacency'):
        adjacency = mesh.face_adjacency
    else:
        from collections import defaultdict
        face_vertex_map = defaultdict(list)
        for i, face in enumerate(mesh.faces):
            for v in face:
                face_vertex_map[v].append(i)
        adjacency = []
        for i in range(n_faces):
            neighbors = set()
            for v in mesh.faces[i]:
                neighbors.update(face_vertex_map[v])
            neighbors.discard(i)
            for n in neighbors:
                adjacency.append([i, n])
        adjacency = np.array(adjacency)
    
    face_adj = {i: set() for i in range(n_faces)}
    for a, b in adjacency:
        face_adj[a].add(b)
        face_adj[b].add(a)
    
    for i in range(n_faces):
        if visited[i]:
            continue
        region = []
        stack = [i]
        while stack:
            curr = stack.pop()
            if visited[curr]:
                continue
            visited[curr] = True
            region.append(curr)
            for nbr in face_adj[curr]:
                if not visited[nbr]:
                    dot_val = np.clip(np.dot(face_normals[curr], face_normals[nbr]), -1, 1)
                    if np.arccos(dot_val) < normal_threshold:
                        stack.append(nbr)
        if len(region) >= min_region_size:
            regions.append(region)
    return regions


def detect_and_tile_planar_regions_new(mesh, normal_threshold=np.deg2rad(5), min_region_size=50, scan_size=25):
    """@brief Detect planar regions and tile them with a grid.

    Regions of nearly constant normal are extracted via region growing and each
    is covered by a grid of scan points with spacing ``scan_size``.

    @param mesh Mesh object.
    @param normal_threshold Tolerance for face normal differences.
    @param min_region_size Minimum faces required for a region.
    @param scan_size Grid spacing for tiling.

    @return ``planar_pts`` and ``planar_normals`` arrays.
    """
    regions = region_grow_planar_faces(mesh, normal_threshold, min_region_size)
    planar_pts_list = []
    planar_normals_list = []
    
    for region in regions:
        region_faces = mesh.faces[region]
        face_centroids = np.mean(mesh.vertices[region_faces], axis=1)
        clustering = DBSCAN(eps=scan_size * 0.5, min_samples=1).fit(face_centroids)
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            face_cluster_indices = np.where(clustering.labels_ == label)[0]
            faces_in_cluster = region_faces[face_cluster_indices]
            cluster_vertex_indices = np.unique(faces_in_cluster.flatten())
            cluster_pts = mesh.vertices[cluster_vertex_indices]
            if cluster_pts.shape[0] < min_region_size:
                continue
            center = np.mean(cluster_pts, axis=0)
            tol = scan_size * 0.5
            closest, dist, _ = mesh.nearest.on_surface(np.array([center]))
            if dist[0] > tol:
                continue
            face_indices_in_mesh = np.array(region)[face_cluster_indices]
            face_normals = mesh.face_normals[face_indices_in_mesh]
            plane_normal = np.mean(face_normals, axis=0)
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            pts, _ = tile_planar_region(cluster_pts, scan_size)
            normals = np.tile(plane_normal, (pts.shape[0], 1))
            planar_pts_list.append(pts)
            planar_normals_list.append(normals)
    
    if planar_pts_list:
        planar_pts = np.vstack(planar_pts_list)
        planar_normals = np.vstack(planar_normals_list)
    else:
        planar_pts = np.empty((0, 3))
        planar_normals = np.empty((0, 3))
    return planar_pts, planar_normals


# =============================================================================
# Poisson Disc Sampling and Redundancy Filtering
# =============================================================================
def poisson_disc_sampling(points, min_distance):
    """@brief Perform Poisson disc sampling on a point set.

    Randomly permutes the points and iteratively selects one while removing
    neighbors within ``min_distance`` to maintain a uniform distribution.

    @param points Array of points.
    @param min_distance Minimum allowed distance between points.

    @return Subset of the points after sampling.
    """
    if points.shape[0] == 0:
        return points
    tree = KDTree(points)
    selected = []
    removed = np.zeros(points.shape[0], dtype=bool)
    indices = np.random.permutation(points.shape[0])
    for idx in indices:
        if removed[idx]:
            continue
        selected.append(points[idx])
        neighbors = tree.query_ball_point(points[idx], min_distance)
        removed[neighbors] = True
    return np.array(selected)


def redundancy_filter(points, min_distance):
    """@brief Remove points that are closer than ``min_distance``.

    Uses a KD-tree to discard points that lie within ``min_distance`` of
    any previously kept point.

    @param points Array of points.
    @param min_distance Distance threshold.

    @return Filtered points.
    """
    if points.shape[0] == 0:
        return points
    tree = KDTree(points)
    keep_mask = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if not keep_mask[i]:
            continue
        neighbors = tree.query_ball_point(p, min_distance)
        for j in neighbors:
            if j != i:
                keep_mask[j] = False
    return points[keep_mask]


# =============================================================================
# Optimal Scan Points Generation (Hybrid Method)
# =============================================================================
def generate_optimal_scan_points_hybrid(stl_file, scan_size, N_candidates=30000, factor_non_cyl=0.5,
                                          debug_only_cyl=False, debug_single_row=False):
    """@brief Generate scan points using a hybrid approach.

    Combines cylindrical tiling, planar region detection and Poisson disc
    sampling to achieve good coverage of complex parts.

    @param stl_file Path to the STL file.
    @param scan_size Grid spacing for scan points.
    @param N_candidates Number of candidate points for initial sampling.
    @param factor_non_cyl Spacing factor for non‑cylindrical regions.
    @param debug_only_cyl If ``True``, return only cylindrical points.
    @param debug_single_row If ``True`` with ``debug_only_cyl``, return a single row.

    @return ``scan_points``, ``scan_normals`` and the loaded ``mesh``.
    """
    import trimesh
    from scipy.spatial import cKDTree, KDTree
    from sklearn.cluster import DBSCAN

    # Load the mesh.
    mesh = trimesh.load(stl_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    
    # Sample candidate points (used later in the hybrid approach).
    pts, face_indices = trimesh.sample.sample_surface(mesh, N_candidates)
    pts = np.array(pts)
    tree = cKDTree(mesh.vertices)
    dists, v_indices = tree.query(pts)
    normals_global = mesh.vertex_normals[v_indices]
    
    # Compute a cylindrical mask based on vertex curvature.
    cyl_mask = segment_cylindrical_region(mesh, curvature_threshold=0.1)
    
    # If debug_only_cyl is True, return only cylindrical points.
    if debug_only_cyl:
        # If additionally debug_single_row is True, return only one row of points.
        if debug_single_row:
            # Extract cylindrical vertices.
            cyl_vertices = mesh.vertices[cyl_mask]
            if cyl_vertices.shape[0] == 0:
                return np.empty((0, 3)), np.empty((0, 3)), mesh
            # Estimate the cylinder's center and axis.
            center_est = np.mean(cyl_vertices, axis=0)
            pts_centered = cyl_vertices - center_est
            U, S, Vt = np.linalg.svd(pts_centered)
            axis_est = Vt[0]
            axis_est = axis_est / np.linalg.norm(axis_est)
            # Estimate radius from projected distances.
            proj = cyl_vertices - np.outer(np.dot(pts_centered, axis_est), axis_est)
            dists_proj = np.linalg.norm(proj - center_est, axis=1)
            radius_est = np.median(dists_proj)
            # Determine the z-range along the cylinder axis.
            z_vals = np.dot(pts_centered, axis_est)
            z_min, z_max = np.min(z_vals), np.max(z_vals)
            n_z = int(np.ceil((z_max - z_min) / scan_size))
            zs = np.linspace(z_min, z_max, n_z)
            
            # Determine two orthogonal directions (v and w) perpendicular to the axis.
            if np.abs(np.dot(axis_est, np.array([0, 0, 1]))) > 0.99:
                v = np.cross(np.array([0, 1, 0]), axis_est)
            else:
                v = np.cross(np.array([0, 0, 1]), axis_est)
            v = v / np.linalg.norm(v)
            w = np.cross(axis_est, v)
            w = w / np.linalg.norm(w)
            # Choose a fixed theta (e.g., theta = 0) so that the part is not rotated.
            theta = 0.0
            
            lateral_pts = []
            lateral_normals = []
            for z in zs:
                pt = center_est + radius_est * (np.cos(theta) * v + np.sin(theta) * w) + z * axis_est
                normal = np.array([np.cos(theta) * v[0] + np.sin(theta) * w[0],
                                   np.cos(theta) * v[1] + np.sin(theta) * w[1],
                                   np.cos(theta) * v[2] + np.sin(theta) * w[2]])
                lateral_pts.append(pt)
                lateral_normals.append(normal)
            lateral_pts = np.array(lateral_pts)
            lateral_normals = np.array(lateral_normals)
            return lateral_pts, lateral_normals, mesh
        else:
            # Return the full cylindrical grid.
            cyl_pts, cyl_normals = tile_cylindrical_region_with_ends(mesh, scan_size)
            return cyl_pts, cyl_normals, mesh

    # Proceed with the full hybrid method if not in debug-only cylindrical mode.
    planar_regions = region_grow_planar_faces(mesh, normal_threshold=np.deg2rad(5), min_region_size=50)
    planar_faces = set()
    for region in planar_regions:
        planar_faces.update(region)
    candidate_in_planar = np.array([face_idx in planar_faces for face_idx in face_indices])
    
    non_feature_mask = ~(cyl_mask[v_indices] | candidate_in_planar)
    non_feature_pts = pts[non_feature_mask]
    if non_feature_pts.shape[0] > 0:
        min_dist = factor_non_cyl * (scan_size / np.sqrt(2))
        scan_pts_non_feature = poisson_disc_sampling(non_feature_pts, min_dist)
        tree_non_feature = KDTree(non_feature_pts)
        indices = []
        for pt in scan_pts_non_feature:
            _, idx = tree_non_feature.query(pt)
            indices.append(idx)
        scan_normals_non_feature = normals_global[non_feature_mask][indices]
    else:
        scan_pts_non_feature = np.empty((0, 3))
        scan_normals_non_feature = np.empty((0, 3))
    
    cyl_pts, cyl_normals = tile_cylindrical_region_with_ends(mesh, scan_size)
    planar_pts, planar_normals = detect_and_tile_planar_regions_new(
        mesh, normal_threshold=np.deg2rad(5), min_region_size=50, scan_size=scan_size
    )
    
    candidates = []
    normals_list = []
    if scan_pts_non_feature.shape[0] > 0:
        candidates.append(scan_pts_non_feature)
        normals_list.append(scan_normals_non_feature)
    if cyl_pts.shape[0] > 0:
        candidates.append(cyl_pts)
        normals_list.append(cyl_normals)
    if planar_pts.shape[0] > 0:
        candidates.append(planar_pts)
        normals_list.append(planar_normals)
    if candidates:
        scan_points = np.vstack(candidates)
        scan_normals = np.vstack(normals_list)
    else:
        scan_points = np.empty((0, 3))
        scan_normals = np.empty((0, 3))
    
    coverage_radius = scan_size / np.sqrt(2)
    if scan_points.shape[0] > 0:
        tree_scans = KDTree(scan_points)
        dists_to_scan, _ = tree_scans.query(pts)
        unscanned_mask = dists_to_scan >= coverage_radius
    else:
        unscanned_mask = np.ones(pts.shape[0], dtype=bool)
    
    unsampled_pts = pts[unscanned_mask]
    if unsampled_pts.shape[0] > 0:
        clustering = DBSCAN(eps=coverage_radius, min_samples=3).fit(unsampled_pts)
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)
        additional_points = []
        additional_normals = []
        mesh_tree = KDTree(mesh.vertices)
        for label in unique_labels:
            cluster_pts = unsampled_pts[labels == label]
            if cluster_pts.shape[0] >= 3:
                centroid = cluster_pts.mean(axis=0)
                additional_points.append(centroid)
                _, idx = mesh_tree.query(centroid)
                n = mesh.vertex_normals[idx]
                additional_normals.append(n / np.linalg.norm(n))
        if additional_points:
            additional_points = np.array(additional_points)
            additional_normals = np.array(additional_normals)
            scan_points = np.vstack([scan_points, additional_points])
            scan_normals = np.vstack([scan_normals, additional_normals])
    
    return scan_points, scan_normals, mesh



# =============================================================================
# Visualization of Detected Flat Surfaces
# =============================================================================
def visualize_detected_surfaces(mesh, scan_size=25, normal_threshold=np.deg2rad(5), min_region_size=50):
    """@brief Visualize detected planar surfaces using Plotly.

    Shows clusters of planar faces and their normals to assist with debugging
    of the planar region detection logic.

    @param mesh Mesh object.
    @param scan_size Grid spacing for tiling.
    @param normal_threshold Tolerance for grouping face normals.
    @param min_region_size Minimum faces required for a valid region.
    """
    surfaces = []
    normals_info = []
    n_skipped = 0
    n_total = 0

    regions = region_grow_planar_faces(mesh, normal_threshold, min_region_size)
    
    for region in regions:
        region_faces = mesh.faces[region]
        face_centroids = np.mean(mesh.vertices[region_faces], axis=1)
        clustering = DBSCAN(eps=scan_size * 0.5, min_samples=1).fit(face_centroids)
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            n_total += 1
            face_cluster_indices = np.where(clustering.labels_ == label)[0]
            faces_in_cluster = region_faces[face_cluster_indices]
            cluster_vertex_indices = np.unique(faces_in_cluster.flatten())
            cluster_pts = mesh.vertices[cluster_vertex_indices]
            if cluster_pts.shape[0] < min_region_size:
                continue
            center = np.mean(cluster_pts, axis=0)
            tol = scan_size * 0.5
            closest, dist, _ = mesh.nearest.on_surface(np.array([center]))
            if dist[0] > tol:
                n_skipped += 1
                continue
            U, S, Vt = np.linalg.svd(cluster_pts - center)
            plane_normal = Vt[-1]
            normals_info.append((center, plane_normal))
            pts, _ = tile_planar_region(cluster_pts, scan_size)
            surfaces.append(pts)
    
    print(f"Detected {len(surfaces)} flat surfaces (skipped {n_skipped} out of {n_total} clusters).")
    
    fig = go.Figure()
    # Plot the original mesh.
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color='lightgrey',
        opacity=0.5,
        name='Mesh'
    ))
    
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    
    for idx, pts in enumerate(surfaces):
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors[idx % len(colors)]),
            name=f'Surface {idx+1}'
        ))
    
    for idx, (center, plane_normal) in enumerate(normals_info):
        arrow_length = scan_size * 0.5
        arrow_end = center + plane_normal * arrow_length
        fig.add_trace(go.Scatter3d(
            x=[center[0], arrow_end[0]],
            y=[center[1], arrow_end[1]],
            z=[center[2], arrow_end[2]],
            mode='lines+markers',
            line=dict(color='black', width=4),
            marker=dict(size=4),
            name=f'Normal {idx+1}'
        ))
    
    fig.update_layout(title="Detected Flat Surfaces with Centroid Check",
                      scene=dict(aspectmode="data"))
    fig.show()


# =============================================================================
# Main Test Function
# =============================================================================
if __name__ == '__main__':
    # Replace the file path with the path to your STL file.
    stl_file = r"C:\Users\robbi\Documents\STL\TorpedoMockup.stl"
    scan_size = 50
    # Load the mesh and generate scan points.
    _, _, mesh = generate_optimal_scan_points_hybrid(stl_file, scan_size, N_candidates=30000, factor_non_cyl=0.8)
    # Visualize the detected flat surfaces.
    visualize_detected_surfaces(mesh, scan_size=scan_size, normal_threshold=np.deg2rad(5), min_region_size=50)
