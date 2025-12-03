import numpy as np
import json
from scipy.spatial import ConvexHull
from itertools import permutations

def generate_truncated_octahedron_vertices():
    # Permutations of (+/- 1, +/- 2, 0)
    base = [1, 2, 0]
    vertices = set()
    for p in permutations(base):
        for signs in product([1, -1], repeat=3):
            v = tuple(p[i] * signs[i] for i in range(3))
            vertices.add(v)
    
    # Filter duplicates (though set handles it, just to be safe logic)
    # The logic above generates 6 permutations * 8 signs = 48.
    # But 0 is invariant to sign, so we have overcounting.
    # Let's be explicit.
    
    unique_vertices = set()
    for x in [-1, 1]:
        for y in [-2, 2]:
            unique_vertices.add((x, y, 0))
            unique_vertices.add((x, 0, y))
            unique_vertices.add((0, x, y))
            unique_vertices.add((y, x, 0))
            unique_vertices.add((y, 0, x))
            unique_vertices.add((0, y, x))
            
    return np.array(list(unique_vertices))

from itertools import product

def get_truncated_octahedron_data():
    # 1. Generate Basic Vertices
    # Permutations of (+/- 1, +/- 2, 0)
    # (1, 2, 0)
    vs = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            vs.append([s1 * 1, s2 * 2, 0]) # +/- 1, +/- 2, 0
            vs.append([s1 * 2, s2 * 1, 0]) # +/- 2, +/- 1, 0
            vs.append([s1 * 1, 0, s2 * 2]) # +/- 1, 0, +/- 2
            vs.append([s1 * 2, 0, s2 * 1]) # +/- 2, 0, +/- 1
            vs.append([0, s1 * 1, s2 * 2]) # 0, +/- 1, +/- 2
            vs.append([0, s1 * 2, s2 * 1]) # 0, +/- 2, +/- 1
    vertices = np.array(vs, dtype=float)
    # Should be 24
    unique_vertices = np.unique(vertices, axis=0)
    
    # 2. Rotate to align a square face with z=0
    # A square face is formed by (2, 1, 0), (2, -1, 0), (2, 0, 1), (2, 0, -1).
    # Center is (2, 0, 0). Normal is (1, 0, 0).
    # We want normal to be (0, 0, -1) or (0, 0, 1).
    # Rotate -90 deg around Y axis: (x, y, z) -> (z, y, -x)
    # Then (2, 0, 0) -> (0, 0, -2).
    # Shift by +2 in Z -> (0, 0, 0).
    
    rotated_vertices = np.zeros_like(unique_vertices)
    rotated_vertices[:, 0] = unique_vertices[:, 2]
    rotated_vertices[:, 1] = unique_vertices[:, 1]
    rotated_vertices[:, 2] = -unique_vertices[:, 0]
    
    # Translate
    rotated_vertices[:, 2] += 2.0
    
    # 3. Make Irregular (Affine Transform)
    # Scale axes slightly to be irregular but preserve planarity.
    scale = np.array([1.2, 0.8, 1.5]) 
    # Note: Non-uniform scaling preserves planarity of faces.
    irregular_vertices = rotated_vertices * scale
    
    # 4. Get Topology (Edges, Faces) using ConvexHull
    hull = ConvexHull(irregular_vertices)
    
    # Extract Faces (Polygons)
    # We need to group coplanar simplices.
    # Equations: (a, b, c, d) such that ax+by+cz+d=0
    # hull.equations
    
    # Group simplices by normal
    tol = 1e-4
    face_indices = []
    
    # We will recover faces by looking at the equations
    # hull.equations gives [normal_x, normal_y, normal_z, offset]
    # For each unique equation, find vertices close to the plane.
    
    unique_equations = []
    for eq in hull.equations:
        # Check if we already have this equation (up to tolerance)
        is_new = True
        for u_eq in unique_equations:
            if np.allclose(eq, u_eq, atol=tol):
                is_new = False
                break
        if is_new:
            unique_equations.append(eq)
            
    # Now for each unique equation, identify vertices
    faces = []
    for eq in unique_equations:
        normal = eq[:3]
        d = eq[3]
        # dist = n.dot(v) + d
        dists = np.abs(irregular_vertices.dot(normal) + d)
        on_plane_indices = np.where(dists < tol)[0]
        
        # Order vertices to form a polygon
        # Project to 2D plane, compute centroid, sort by angle
        face_verts = irregular_vertices[on_plane_indices]
        if len(face_verts) < 3: continue
        
        # Local coordinates
        # Center
        center = np.mean(face_verts, axis=0)
        # Vectors from center
        vecs = face_verts - center
        # Normal vector for this face
        n = normal
        
        # Create two orthogonal tangent vectors
        # arbitrary vector not parallel to n
        arb = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(arb, n)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        
        t1 = np.cross(n, arb)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        
        # Project
        x_proj = vecs.dot(t1)
        y_proj = vecs.dot(t2)
        
        angles = np.arctan2(y_proj, x_proj)
        sorted_order = np.argsort(angles)
        
        ordered_indices = on_plane_indices[sorted_order]
        faces.append(ordered_indices.tolist())

    # Extract Edges from Faces (to avoid diagonals)
    edges_set = set()
    for f in faces:
        # f is a list of indices in order
        for i in range(len(f)):
            u, v = f[i], f[(i+1)%len(f)]
            if u > v: u, v = v, u
            edges_set.add((u, v))
            
    edges = list(edges_set)

    # 5. Identify Anchored Face
    # Find face with all vertices at z=0
    anchor_face = None
    for f in faces:
        z_vals = irregular_vertices[f, 2]
        if np.allclose(z_vals, 0, atol=1e-4):
            anchor_face = f
            break
            
    if anchor_face is None:
        # Fallback: find lowest face
        min_z = np.min([np.mean(irregular_vertices[f, 2]) for f in faces])
        for f in faces:
             if np.isclose(np.mean(irregular_vertices[f, 2]), min_z, atol=1e-4):
                 anchor_face = f
                 break

    # 6. Calculate Measured Edge Lengths
    measured_edge_lengths = []
    for (u, v) in edges:
        dist = np.linalg.norm(irregular_vertices[u] - irregular_vertices[v])
        measured_edge_lengths.append(float(dist))

    return {
        "vertices": irregular_vertices.tolist(),
        "edges": [[int(u), int(v)] for u, v in edges], # list of [u, v]
        "faces": [[int(x) for x in f] for f in faces], # list of lists
        "measured_edge_lengths": measured_edge_lengths,
        "anchor_face": [int(x) for x in anchor_face] if anchor_face else None
    }

if __name__ == "__main__":
    data = get_truncated_octahedron_data()
    with open("diamond_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(data['vertices'])} vertices")
    print(f"Generated {len(data['edges'])} edges")
    print(f"Generated {len(data['faces'])} faces")
    print(f"Anchor face: {data['anchor_face']}")

