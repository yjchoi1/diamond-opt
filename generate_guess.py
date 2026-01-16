import numpy as np
import networkx as nx
from sklearn.manifold import MDS

def get_initial_guess(data, config, true_vertices=None):
    """
    Generates the initial vertex positions for the optimization.
    
    Args:
        data (dict): The loaded JSON data containing edges, faces, anchors, etc.
        config (dict): The configuration dictionary.
        true_vertices (np.ndarray, optional): Ground truth vertices if available.
    
    Returns:
        np.ndarray: Initial vertices (N, 3)
    """
    
    anchor_indices = data['anchor_face']
    
    # Determine number of vertices
    if true_vertices is not None:
        n_vertices = len(true_vertices)
    elif 'vertices' in data:
        n_vertices = len(data['vertices'])
    else:
        all_edge_indices = [idx for edge in data['edges'] for idx in edge]
        n_vertices = max(all_edge_indices) + 1

    # 1. Check for explicit initial guess in data
    if 'initial_guess' in data:
        print("Initializing with Pre-computed Initial Guess from JSON")
        return np.array(data['initial_guess'])

    # 2. Select Strategy
    strategy = config.get("initial_guess", "auto")
    if not isinstance(strategy, str):
        raise ValueError("config['initial_guess'] must be a string indicating the strategy to use.")
    strategy = strategy.lower()
    valid_strategies = {"auto", "scale", "perturb", "mds", "blind", "polyhedron"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown initial guess strategy '{strategy}'. Valid options: {sorted(valid_strategies)}")

    # Determine automatic fallback if requested
    if strategy == "auto":
        if true_vertices is not None:
            strategy = "perturb"
        elif 'measured_edge_lengths' in data and len(data['measured_edge_lengths']) > 0:
            strategy = "mds"
        else:
            strategy = "blind"
    
    if strategy == "scale":
        if true_vertices is None:
            raise ValueError("Scale initialization requires true_vertices.")
        scale_factor = config.get("scale_factor", 1.0)
        print(f"Initializing with Scale Factor Strategy ({scale_factor})")
        return _scale_initialization(true_vertices, anchor_indices, scale_factor)
    
    if strategy == "perturb":
        if true_vertices is None:
            raise ValueError("Perturb initialization requires true_vertices.")
        perturbation = config.get("initial_perturbation", 0.1)
        print(f"Initializing with Perturb Strategy (σ={perturbation})")
        return _perturb_initialization(true_vertices, perturbation, anchor_indices, n_vertices)
    
    if strategy == "mds":
        print("Initializing with MDS (Multidimensional Scaling)")
        try:
            return _mds_initialization(data, n_vertices)
        except Exception as e:
            print(f"MDS failed ({e}), falling back to blind random guess.")
            strategy = "blind"

    if strategy == "polyhedron":
        print("Initializing with Canonical Polyhedron Strategy (Force-Directed)")
        try:
            return _polyhedron_initialization(data, n_vertices, config)
        except Exception as e:
            print(f"Polyhedron initialization failed ({e}), falling back to blind random guess.")
            strategy = "blind"
    
    # strategy == "blind"
    print("Initializing with Blind Random Guess")
    return _blind_random_initialization(data, n_vertices)


def _scale_initialization(true_vertices, anchor_indices, scale_factor):
    initial_vertices = true_vertices.copy()
    anchor_center = np.mean(initial_vertices[anchor_indices], axis=0)
    
    # Vector from anchor center
    vecs = initial_vertices - anchor_center
    
    # Scale
    scaled_vecs = vecs * scale_factor
    
    initial_vertices = anchor_center + scaled_vecs
    
    # Reset anchor vertices to exact true position
    initial_vertices[anchor_indices] = true_vertices[anchor_indices]
    return initial_vertices


def _perturb_initialization(true_vertices, perturbation_amount, anchor_indices, n_vertices):
    # Identify variable indices
    all_indices = set(range(n_vertices))
    fixed_indices_set = set(anchor_indices)
    variable_indices = sorted(list(all_indices - fixed_indices_set))
    
    perturbation = np.random.normal(0, perturbation_amount, true_vertices.shape)
    initial_vertices = true_vertices.copy()
    # Only perturb variable vertices
    initial_vertices[variable_indices] += perturbation[variable_indices]
    return initial_vertices


def _blind_random_initialization(data, n_vertices):
    anchor_indices = data['anchor_face']
    
    # Get anchor coordinates
    if 'anchor_coordinates' in data:
        anchor_coords = np.array(data['anchor_coordinates'])
    elif 'vertices' in data:
         anchor_coords = np.array(data['vertices'])[anchor_indices]
    else:
        # Fallback if somehow we don't have anchor coords
        raise ValueError("Anchor coordinates missing for blind initialization")

    initial_vertices = np.zeros((n_vertices, 3))
    
    # Set anchor vertices
    initial_vertices[anchor_indices] = anchor_coords
    
    # Initialize variable vertices randomly around the anchor center
    anchor_center = np.mean(anchor_coords, axis=0)
    
    # Identify variable indices
    all_indices = set(range(n_vertices))
    fixed_indices_set = set(anchor_indices)
    variable_indices = sorted(list(all_indices - fixed_indices_set))
    
    # Random offset
    scale = 2.0 
    random_offset = np.random.normal(0, scale, (n_vertices, 3)) 
    
    initial_vertices[variable_indices] = anchor_center + random_offset[variable_indices]
    
    return initial_vertices


def _mds_initialization(data, n_vertices):
    edges = data['edges']
    lengths = data['measured_edge_lengths']
    anchor_indices = data['anchor_face']
    
    if 'anchor_coordinates' in data:
        anchor_coords = np.array(data['anchor_coordinates'])
    elif 'vertices' in data:
         anchor_coords = np.array(data['vertices'])[anchor_indices]
    else:
        raise ValueError("Anchor coordinates missing for MDS alignment")

    # 1. Build Graph and Distance Matrix
    G = nx.Graph()
    for i, (u, v) in enumerate(edges):
        # If lengths are missing or mismatched, this might fail, so we assume correctness
        w = lengths[i] if i < len(lengths) else 1.0
        G.add_edge(u, v, weight=w)
    
    # Compute all-pairs shortest paths
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    
    # Map graph nodes to 0..N-1 indices if they aren't already
    # But here we assume nodes are 0..N-1 integers
    dist_matrix = np.zeros((n_vertices, n_vertices))
    
    # We need to handle disconnected components or nodes not in edges?
    # Assuming connected graph for now.
    
    for i in range(n_vertices):
        for j in range(n_vertices):
            if i == j:
                dist_matrix[i, j] = 0
            elif i in path_lengths and j in path_lengths[i]:
                dist_matrix[i, j] = path_lengths[i][j]
            else:
                 # Disconnected graph? Use a large number
                 dist_matrix[i, j] = 1000.0

    # 2. MDS
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dist_matrix)

    # 3. Align to Anchor Face (Procrustes / Kabsch)
    # Extract guess anchors
    guess_anchors = coords[anchor_indices]
    
    # Centroids
    guess_center = np.mean(guess_anchors, axis=0)
    target_center = np.mean(anchor_coords, axis=0)
    
    # Centered vectors
    guess_centered = guess_anchors - guess_center
    target_centered = anchor_coords - target_center
    
    # Kabsch
    H = guess_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    aligned_coords = (coords - guess_center) @ R + target_center
    
    return aligned_coords


def _polyhedron_initialization(data, n_vertices, config):
    anchor_indices = data['anchor_face']
    
    # Get anchor coordinates
    if 'anchor_coordinates' in data:
        anchor_coords = np.array(data['anchor_coordinates'])
    elif 'vertices' in data:
        anchor_coords = np.array(data['vertices'])[anchor_indices]
    else:
        raise ValueError("Anchor coordinates missing for polyhedron initialization")
        
    # Build graph with edges AND diagonals to rigidify structure
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    G.add_edges_from(data['edges'])
    if 'diagonals' in data:
         G.add_edges_from(data['diagonals'])
         
    # Prepare fixed positions for anchors
    fixed_pos = {}
    for i, idx in enumerate(anchor_indices):
        fixed_pos[idx] = anchor_coords[i]
        
    # Estimate spring constant 'k' based on average anchor edge length
    anchor_dists = []
    for i in range(len(anchor_indices)):
        p1 = fixed_pos[anchor_indices[i]]
        p2 = fixed_pos[anchor_indices[(i+1) % len(anchor_indices)]]
        anchor_dists.append(np.linalg.norm(p1 - p2))
    avg_anchor_edge = np.mean(anchor_dists) if anchor_dists else 1.0
    
    # Get size scaling from config
    polyhedron_scale = config.get("polyhedron_size", 1.0)
    k_val = avg_anchor_edge * polyhedron_scale

    # Determine "Up" direction from anchors
    # Assuming anchors form a base, we calculate normal
    anchor_center = np.mean(anchor_coords, axis=0)
    
    # Calculate normal of the anchor plane
    if len(anchor_indices) >= 3:
        v1 = anchor_coords[1] - anchor_coords[0]
        v2 = anchor_coords[2] - anchor_coords[0]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-6:
             normal = np.array([0.0, 0.0, 1.0])
        else:
             normal = normal / np.linalg.norm(normal)
    else:
        normal = np.array([0.0, 0.0, 1.0])
        
    # Force normal to point to +Z if the Z component is significant
    # This ensures "up" is consistent with world "up" if anchors are roughly horizontal
    if normal[2] < 0:
        normal = -normal
        
    # Initial positions for ALL nodes to help spring_layout
    init_pos = {}
    for i in range(n_vertices):
        if i in fixed_pos:
            init_pos[i] = fixed_pos[i]
        else:
            # Initialize ABOVE the anchor plane
            # Add noise in X/Y, but positive bias in Normal direction
            noise = np.random.normal(0, avg_anchor_edge * 0.5, 3)
            # Project noise onto plane to keep X/Y spread
            noise_plane = noise - np.dot(noise, normal) * normal
            
            # Add height component: ensure it's positive relative to normal
            # We bias the initialization to be definitely "above"
            height = abs(np.random.normal(avg_anchor_edge, avg_anchor_edge * 0.2)) + avg_anchor_edge * 0.5
            
            init_pos[i] = anchor_center + noise_plane + (normal * height * polyhedron_scale)

    # Run Spring Layout
    pos = nx.spring_layout(G, dim=3, k=k_val, pos=init_pos, fixed=anchor_indices, iterations=500, scale=k_val*n_vertices)
    
    # Convert to numpy array
    initial_vertices = np.zeros((n_vertices, 3))
    
    # Post-process to ensure "upward" orientation if spring layout flipped things
    non_anchor_indices = [i for i in range(n_vertices) if i not in anchor_indices]
    
    for i in range(n_vertices):
        initial_vertices[i] = pos[i]

    if non_anchor_indices:
        non_anchor_verts = initial_vertices[non_anchor_indices]
        centroid_non_anchor = np.mean(non_anchor_verts, axis=0)
        direction = centroid_non_anchor - anchor_center
        
        # Check dot product with the *forced positive* normal
        if np.dot(direction, normal) < 0:
            print("Reflecting initial guess to face upward...")
            for i in non_anchor_indices:
                v = initial_vertices[i]
                dist = np.dot(v - anchor_center, normal)
                # Reflect across the plane defined by anchor_center and normal
                initial_vertices[i] = v - 2 * dist * normal

    return initial_vertices
