import torch
import numpy as np
import json
import time
from scipy.optimize import minimize
import sys
import os
from generate_guess import get_initial_guess
import trimesh

# Add current directory to path to import visualizer
sys.path.append(os.getcwd())
from visualizer import plot_optimization_history, plot_3d_comparison, plot_3d_animation

# --- Hardcoded Inputs ---
INPUT_CONFIG = {
    "data_path": "diamond_data_experiment.json", # Updated to use blind data by default
    # "data_path": "diamond_data_blind2.json",
    # "data_path": "diamond_data.json",
    # "initial_guess": "perturb",  # Options: auto, scale, perturb, mds, blind, polyhedron
    "initial_guess": "polyhedron",  # Options: auto, scale, perturb, mds, blind, polyhedron
    "polyhedron_size": 1, # Used when initial_guess is set to polyhedron
    "initial_perturbation": 0.5,  # Used when initial_guess is set to perturb
    "scale_factor": 1.1, # Used when initial_guess is set to scale
    "optimization": {
        "method": "L-BFGS-B",
        "maxiter": 2000,
        "ftol": 1e-9
    },
    "weights": {
        "length": 1.0,  # 1
        "planarity": 0.01,  # 1000
        "convexity": 0.0,  # 100
        "diagonal": 0.0 # 1.0 Weight for diagonal constraints
    }
}
# ------------------------

class DiamondReconstruction:
    def __init__(self, config):
        self.config = config
        with open(config["data_path"], 'r') as f:
            self.data = json.load(f)
        
        # Check if true vertices exist
        if 'vertices' in self.data:
            self.true_vertices = np.array(self.data['vertices'])
            self.n_vertices = len(self.true_vertices)
        else:
            self.true_vertices = None
            # Infer N from edges if vertices not provided
            all_edge_indices = [idx for edge in self.data['edges'] for idx in edge]
            self.n_vertices = max(all_edge_indices) + 1

        self.edges = self.data['edges']
        self.faces = self.data['faces']
        self.measured_lengths = torch.tensor(self.data['measured_edge_lengths'], dtype=torch.float64)
        
        # Load Diagonal Data
        # Diagonals are critical for rigidifying non-triangular faces (e.g. squares, hexagons).
        # Without them, the shape can shear while preserving edge lengths.
        self.diagonals = self.data.get('diagonals', [])
        self.measured_diagonal_lengths = torch.tensor(self.data.get('measured_diagonal_lengths', []), dtype=torch.float64)
        
        self.anchor_indices = self.data['anchor_face']
        
        # Handle Anchor Coordinates
        if self.true_vertices is not None:
            anchor_coords_np = self.true_vertices[self.anchor_indices]
        else:
            if 'anchor_coordinates' in self.data:
                anchor_coords_np = np.array(self.data['anchor_coordinates'])
            else:
                raise ValueError("If 'vertices' are not provided, 'anchor_coordinates' must be present.")

        self.fixed_coords = torch.tensor(anchor_coords_np, dtype=torch.float64)

        # Identify variable indices (non-anchor)
        all_indices = set(range(self.n_vertices))
        self.fixed_indices_set = set(self.anchor_indices)
        self.variable_indices = sorted(list(all_indices - self.fixed_indices_set))
        
        # Initial Guess
        np.random.seed(42)
        initial_vertices = get_initial_guess(self.data, self.config, self.true_vertices)
        
        # Decision Variables (Variables x 3)
        self.x_vars = torch.tensor(
            initial_vertices[self.variable_indices], 
            dtype=torch.float64, 
            requires_grad=True
        )
        
        self.loss_history = []
        self.model_error_history = []
        self.trajectory = [initial_vertices.copy()] # Store full (N, 3) numpy arrays
        
    def get_full_vertices(self, x_current):
        # Reconstruct full (N, 3) tensor
        # x_current is (N_vars, 3) or flat
        if x_current.ndim == 1:
            x_current = x_current.reshape(-1, 3)
            
        full_verts = torch.zeros((self.n_vertices, 3), dtype=torch.float64, device=x_current.device)
        
        # Indices tensors
        var_idx = torch.tensor(self.variable_indices, device=x_current.device)
        fix_idx = torch.tensor(self.anchor_indices, device=x_current.device)
        
        # Scatter variables
        full_verts.index_copy_(0, var_idx, x_current)
        # Scatter fixed
        full_verts.index_copy_(0, fix_idx, self.fixed_coords)
        
        return full_verts

    def objective(self, x_flat):
        # x_flat is numpy array from scipy
        # Convert to torch
        if isinstance(x_flat, np.ndarray):
             t_x = torch.tensor(x_flat, dtype=torch.float64, requires_grad=True)
        else:
             t_x = x_flat
             
        # Reshape
        x_reshaped = t_x.reshape(-1, 3)
        vertices = self.get_full_vertices(x_reshaped)
        
        # 1. Edge Length Residuals
        # Gather edge endpoints
        u_indices = [e[0] for e in self.edges]
        v_indices = [e[1] for e in self.edges]
        
        p_u = vertices[u_indices]
        p_v = vertices[v_indices]
        
        current_lengths = torch.norm(p_u - p_v, dim=1)
        
        length_residuals = current_lengths - self.measured_lengths
        length_loss = torch.sum(length_residuals**2)
        
        # 2. Diagonal Length Constraints
        # Minimize the difference between current diagonal lengths and measured diagonal lengths.
        # This prevents shearing of non-triangular faces.
        diag_loss = 0.0
        if len(self.diagonals) > 0:
            u_diag = [d[0] for d in self.diagonals]
            v_diag = [d[1] for d in self.diagonals]
            
            p_u_d = vertices[u_diag]
            p_v_d = vertices[v_diag]
            
            current_diag_lengths = torch.norm(p_u_d - p_v_d, dim=1)
            diag_residuals = current_diag_lengths - self.measured_diagonal_lengths
            diag_loss = torch.sum(diag_residuals**2)
        
        # 3. Planarity Constraints (Penalty)
        # For each face, minimize the smallest eigenvalue of the covariance matrix
        planarity_loss = 0.0
        convexity_loss = 0.0
        centroid = torch.mean(vertices, dim=0)
        
        for face_idx in self.faces:
            # face_idx is list of vertex indices
            face_verts = vertices[face_idx] # (K, 3)
            
            # Centering
            center = torch.mean(face_verts, dim=0)
            centered = face_verts - center
            
            # Covariance matrix: C = (X^T X)
            # We want to find normal direction with least variance.
            cov = torch.matmul(centered.T, centered)
            
            # Eigenvalues
            # torch.linalg.eigvalsh for symmetric matrix
            eigs = torch.linalg.eigvalsh(cov)
            
            # Smallest eigenvalue should be 0 for planar
            min_eig = eigs[0] 
            planarity_loss += min_eig**2 
            
            # Convexity / Star-convexity check
            v0 = face_verts[0]
            v1 = face_verts[1]
            v2 = face_verts[2]
            normal = torch.linalg.cross(v1 - v0, v2 - v0)
            normal = normal / (torch.norm(normal) + 1e-8)
            
            # Check direction relative to centroid
            # Vector from centroid to face center
            dir_vec = center - centroid
            
            # Dot product should be positive (outward facing)
            dot_prod = torch.dot(normal, dir_vec)
            
            # If dot_prod < 0, it means the face is facing "inward" relative to centroid
            # Penalize this.
            convexity_loss += torch.relu(-dot_prod + 0.1) ** 2 # Margin of 0.1
            
        total_loss = (
            self.config["weights"]["length"] * length_loss + 
            self.config["weights"]["planarity"] * planarity_loss + 
            self.config["weights"]["convexity"] * convexity_loss +
            self.config["weights"]["diagonal"] * diag_loss
        )
        
        return total_loss

    def run_optimization(self):
        print("Starting optimization...")
        start_time = time.time()
        
        # Wrapper for Scipy
        def func(x):
            # Forward pass
            t_x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            loss = self.objective(t_x)
            loss.backward()
            
            grad = t_x.grad.numpy().astype(np.float64)
            val = loss.item()
            
            self.loss_history.append(val)
            
            # Calculate Model Error (MSE of vertices) if true vertices exist
            with torch.no_grad():
                if self.true_vertices is not None:
                    x_reshaped = t_x.reshape(-1, 3)
                    current_verts = self.get_full_vertices(x_reshaped).cpu().numpy()
                    # Mean Squared Error between estimated and true vertices
                    mse = np.mean((current_verts - self.true_vertices)**2)
                    self.model_error_history.append(mse)
                else:
                     self.model_error_history.append(0.0) # Placeholder
            
            return val, grad

        # Callback to store trajectory
        def callback(x):
             # Reconstruct full vertices
             t_x = torch.tensor(x, dtype=torch.float64)
             verts = self.get_full_vertices(t_x.reshape(-1, 3)).detach().numpy()
             self.trajectory.append(verts)

        # Initial variables flattened
        x0 = self.x_vars.detach().numpy().flatten()
        
        res = minimize(
            func, 
            x0, 
            method=self.config["optimization"]["method"],
            jac=True,
            callback=callback,
            options={
                'disp': True, 
                'maxiter': self.config["optimization"]["maxiter"], 
                'ftol': self.config["optimization"]["ftol"]
            }
        )
        
        print(f"Optimization finished in {time.time() - start_time:.2f}s")
        print(f"Final Loss: {res.fun}")
        print(f"Message: {res.message}")
        
        # Final vertices
        final_x = torch.tensor(res.x, dtype=torch.float64)
        final_vertices = self.get_full_vertices(final_x.reshape(-1, 3)).detach().numpy()
        
        return final_vertices, self.loss_history, self.model_error_history, self.trajectory

    def verify_results(self, final_vertices):
        # Check Planarity
        max_planarity_error = 0.0
        for face in self.faces:
            pts = final_vertices[face]
            center = np.mean(pts, axis=0)
            centered = pts - center
            cov = centered.T @ centered
            eigs = np.linalg.eigvalsh(cov)
            min_eig = eigs[0]
            if min_eig > max_planarity_error:
                max_planarity_error = min_eig
        
        print(f"Max Planarity Violation (Eigenvalue): {max_planarity_error:.2e}")
        
        # Check Edge Lengths
        max_length_error = 0.0
        for i, (u, v) in enumerate(self.edges):
            dist = np.linalg.norm(final_vertices[u] - final_vertices[v])
            measured = self.measured_lengths[i].item()
            err = abs(dist - measured)
            if err > max_length_error:
                max_length_error = err
        
        print(f"Max Edge Length Error: {max_length_error:.2e}")

    def calculate_volume(self, vertices):
        """Calculates the volume of the polyhedron using the divergence theorem.

        Assumes faces are consistently oriented (e.g., all outward or all inward).
        Uses the formula: V = 1/6 * sum(dot(v0, cross(v1, v2))) for triangulated faces.

        Args:
            vertices: A numpy array of shape (N, 3) containing vertex coordinates.

        Returns:
            float: The calculated volume of the polyhedron.
        """
        # Analytic/divergence-theorem-style volume using fan triangulation
        volume = 0.0
        triangulated_faces = []  # store triangles as index triplets for trimesh
        for face in self.faces:
            # Triangulate face: (v0, vi, vi+1)
            # Assuming convex/star-shaped faces with respect to v0
            v0_idx = face[0]
            v0 = vertices[v0_idx]
            for i in range(1, len(face) - 1):
                v1_idx = face[i]
                v2_idx = face[i+1]
                v1 = vertices[v1_idx]
                v2 = vertices[v2_idx]

                # Signed volume of tetrahedron with origin
                # V = 1/6 * det(v0, v1, v2) = 1/6 * dot(v0, cross(v1, v2))
                cross_prod = np.cross(v1, v2)
                tet_vol = np.dot(v0, cross_prod) / 6.0
                volume += tet_vol

                triangulated_faces.append([v0_idx, v1_idx, v2_idx])

        volume = abs(volume)

        tri_faces_arr = np.array(triangulated_faces, dtype=int)
        mesh = trimesh.Trimesh(vertices=vertices, faces=tri_faces_arr, process=True)
        trimesh_volume = float(mesh.volume)

        # Diagnostics about mesh quality and orientation
        print(
            f"trimesh: watertight={mesh.is_watertight}, "
            f"winding_consistent={mesh.is_winding_consistent}"
        )

        # Try fixing normals / winding and recomputing volume
        mesh_fixed = mesh.copy()
        mesh_fixed.fix_normals()
        trimesh_volume_fixed = float(mesh_fixed.volume)

        print(
            f"Volume (analytic): {volume:.6f}, "
            f"Volume (trimesh): {trimesh_volume:.6f}, "
            f"Volume (trimesh, fixed_normals): {trimesh_volume_fixed:.6f}"
        )
        
        return volume

    def calculate_opposing_distances(self, vertices):
        """Calculates distances between centroids of opposing faces."""
        distances = []
        face_centroids = []
        face_normals = []

        for face in self.faces:
            face_verts = vertices[face]
            centroid = np.mean(face_verts, axis=0)
            face_centroids.append(centroid)
            
            # Normal using first 3 vertices (assuming consistent winding/convexity)
            v0, v1, v2 = face_verts[0], face_verts[1], face_verts[2]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_val = np.linalg.norm(normal)
            normal = normal / norm_val if norm_val > 1e-8 else np.array([0.0, 0.0, 1.0])
            face_normals.append(normal)

        processed_pairs = set()
        for i in range(len(self.faces)):
            n_i = face_normals[i]
            c_i = face_centroids[i]
            
            # Find best opposing face
            best_match_idx = -1
            min_dot = 1.0
            
            for j in range(len(self.faces)):
                if i == j: continue
                dot_prod = np.dot(n_i, face_normals[j])
                if dot_prod < min_dot:
                    min_dot = dot_prod
                    best_match_idx = j
            
            # If sufficiently opposing (anti-parallel)
            if min_dot < -0.9 and best_match_idx != -1:
                j = best_match_idx
                pair = tuple(sorted((i, j)))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    c_j = face_centroids[j]
                    dist = float(np.linalg.norm(c_i - c_j))
                    distances.append({
                        "face_indices": [int(pair[0]), int(pair[1])],
                        "distance": dist,
                        "normal_alignment": float(min_dot)
                    })
                    print(f"Opposing faces {pair}: Distance={dist:.2f}, Alignment={min_dot:.4f}")
                    
        return distances

    def save_reconstructed_data(self, vertices, output_path):
        """Save the reconstructed shape data in the same format as the input JSON.
        
        Args:
            vertices: A numpy array of shape (N, 3) containing reconstructed vertex coordinates.
            output_path: Path to save the JSON file.
        """
        # Convert vertices to list format
        vertices_list = vertices.tolist()
        
        # Calculate actual edge lengths from reconstructed vertices
        actual_edge_lengths = []
        for u, v in self.edges:
            dist = np.linalg.norm(vertices[u] - vertices[v])
            actual_edge_lengths.append(float(dist))

        # Calculate opposing distances
        opposing_distances = self.calculate_opposing_distances(vertices)
        
        # Prepare output data
        output_data = {
            "vertices": vertices_list,
            "edges": self.edges,
            "faces": self.faces,
            "measured_edge_lengths": actual_edge_lengths,
            "anchor_face": self.anchor_indices,
            "opposing_face_distances": opposing_distances
        }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Reconstructed shape saved to {output_path}")

    def save_reconstructed_obj(self, vertices, output_path):
        """Save the reconstructed shape as an OBJ file.
        
        Args:
            vertices: A numpy array of shape (N, 3) containing reconstructed vertex coordinates.
            output_path: Path to save the OBJ file.
        """
        # Triangulate polygon faces (OBJ supports polygons, but trimesh requires triangles)
        tri_faces = []
        for face in self.faces:
            if len(face) < 3:
                continue
            if len(face) == 3:
                tri_faces.append(face)
                continue
            v0 = face[0]
            for i in range(1, len(face) - 1):
                tri_faces.append([v0, face[i], face[i + 1]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=tri_faces, process=False)
        mesh.export(output_path)
        print(f"Reconstructed OBJ saved to {output_path}")


if __name__ == "__main__":
    # Generate data if not exists (though we already did)
    if not os.path.exists(INPUT_CONFIG["data_path"]):
        print(f"Error: {INPUT_CONFIG['data_path']} not found.")
        sys.exit(1)
        
    reconstruction = DiamondReconstruction(INPUT_CONFIG)
    final_verts, history, model_errors, trajectory = reconstruction.run_optimization()
    
    reconstruction.verify_results(final_verts)
    
    volume = reconstruction.calculate_volume(final_verts)
    print(f"Reconstructed Volume: {volume:.2f}")
    
    # Save reconstructed shape data
    input_filename = os.path.splitext(os.path.basename(INPUT_CONFIG["data_path"]))[0]
    output_filename = f"{input_filename}_reconstructed.json"
    reconstruction.save_reconstructed_data(final_verts, output_filename)
    
    # Save reconstructed OBJ mesh
    obj_filename = f"{input_filename}_reconstructed.obj"
    reconstruction.save_reconstructed_obj(final_verts, obj_filename)
    
    # Visualization
    # Pass None for model_errors if we don't have true vertices (or handle inside visualizer)
    # The visualizer handles model_errors=None, but we passed a list of zeros if true_vertices is None.
    # Better to pass None explicitly if true_vertices is None to avoid plotting a flat zero line which is misleading.
    
    plot_model_errors = model_errors if reconstruction.true_vertices is not None else None
    
    plot_optimization_history(history, model_errors=plot_model_errors)
    
    plot_3d_comparison(
        final_verts, 
        reconstruction.true_vertices, 
        reconstruction.faces,
        filename="diamond_reconstruction_static.html"
    )
    
    plot_3d_animation(
        trajectory,
        reconstruction.true_vertices,
        reconstruction.faces,
        filename="diamond_reconstruction_animation.html"
    )
