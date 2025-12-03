import torch
import numpy as np
import json
import time
from scipy.optimize import minimize
import sys
import os

# Add current directory to path to import visualizer
sys.path.append(os.getcwd())
from visualizer import plot_optimization_history, plot_3d_comparison, plot_3d_animation

# --- Hardcoded Inputs ---
INPUT_CONFIG = {
    "data_path": "diamond_data.json",
    "use_scale_initialization": False,
    "initial_perturbation": 0.2,  # If use_scale_initialization is False. We just perturb the vertices, but fix the anchor vertices.
    "scale_factor": 1.1, # If use_scale_initialization is True. We scale the vertices, but fix the anchor vertices.
    "optimization": {
        "method": "L-BFGS-B",
        "maxiter": 2000,
        "ftol": 1e-9
    },
    "weights": {
        "length": 1.0,  # 1
        "planarity": 100.0,  # 1000
        "convexity": 100.0  # 100
    }
}
# ------------------------

class DiamondReconstruction:
    def __init__(self, config):
        self.config = config
        with open(config["data_path"], 'r') as f:
            self.data = json.load(f)
        
        self.true_vertices = np.array(self.data['vertices'])
        self.edges = self.data['edges']
        self.faces = self.data['faces']
        self.measured_lengths = torch.tensor(self.data['measured_edge_lengths'], dtype=torch.float64)
        self.anchor_indices = self.data['anchor_face']
        
        # Identify variable indices (non-anchor)
        self.n_vertices = len(self.true_vertices)
        all_indices = set(range(self.n_vertices))
        self.fixed_indices_set = set(self.anchor_indices)
        self.variable_indices = sorted(list(all_indices - self.fixed_indices_set))
        
        # Initial Guess: Perturb the true vertices (except anchor)
        # to create a non-trivial starting point.
        # This simulates starting from a generic shape.
        np.random.seed(42)
        
        if config.get("use_scale_initialization", False):
             # Option 2: Scale the shape (preserving convexity)
             # We scale around the centroid of the anchor face (or global centroid)
             # Let's scale relative to the anchor face center to keep anchor fixed-ish
             # Actually, anchor face is fixed.
             # We should scale the Z-height or overall size relative to anchor.
             print(f"Initializing with Scale Factor: {config['scale_factor']}")
             initial_vertices = self.true_vertices.copy()
             
             anchor_center = np.mean(initial_vertices[self.anchor_indices], axis=0)
             
             # Vector from anchor center
             vecs = initial_vertices - anchor_center
             
             # Scale
             scaled_vecs = vecs * config["scale_factor"]
             
             initial_vertices = anchor_center + scaled_vecs
             
             # Reset anchor vertices to exact true position (since they are fixed)
             # Although scaling relative to anchor center should preserve them if they are on a plane?
             # If scaling is uniform, they expand.
             # So we must force anchor vertices back to true positions (which the optimizer treats as fixed anyway)
             initial_vertices[self.anchor_indices] = self.true_vertices[self.anchor_indices]
             
        else:
            # Option 1: Random Perturbation
            print("Initializing with Random Perturbation")
            perturbation = np.random.normal(0, config["initial_perturbation"], self.true_vertices.shape)
            initial_vertices = self.true_vertices.copy()
            # Only perturb variable vertices
            initial_vertices[self.variable_indices] += perturbation[self.variable_indices]
        
        # Prepare fixed coordinates
        self.fixed_coords = torch.tensor(self.true_vertices[self.anchor_indices], dtype=torch.float64)
        
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
        
        # 2. Planarity Constraints (Penalty)
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
            # Normal Estimation (Planar fit normal or simple cross product)
            # Since we enforce planarity, cross product of first two edges is sufficient approx
            # But to be robust, use the eigenvector corresponding to min_eig (which is the normal)
            # However, eigenvectors don't have consistent sign direction.
            # So we use cross product of edges: (v1-v0) x (v2-v0)
            # We assume face vertices are ordered consistently (CCW or CW)
            
            v0 = face_verts[0]
            v1 = face_verts[1]
            v2 = face_verts[2]
            normal = torch.linalg.cross(v1 - v0, v2 - v0)
            normal = normal / (torch.norm(normal) + 1e-8)
            
            # Check direction relative to centroid
            # Vector from centroid to face center
            dir_vec = center - centroid
            
            # Dot product should be positive (outward facing)
            # We need to verify if the original data has outward facing normals.
            # Assuming standard ConvexHull output, they are outward.
            dot_prod = torch.dot(normal, dir_vec)
            
            # If dot_prod < 0, it means the face is facing "inward" relative to centroid
            # Penalize this.
            # We use ReLU to only penalize negative values
            convexity_loss += torch.relu(-dot_prod + 0.1) ** 2 # Margin of 0.1
            
        total_loss = (
            self.config["weights"]["length"] * length_loss + 
            self.config["weights"]["planarity"] * planarity_loss + 
            self.config["weights"]["convexity"] * convexity_loss
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
            
            # Calculate Model Error (MSE of vertices)
            # We need full vertices
            with torch.no_grad():
                x_reshaped = t_x.reshape(-1, 3)
                current_verts = self.get_full_vertices(x_reshaped).cpu().numpy()
                # Mean Squared Error between estimated and true vertices
                mse = np.mean((current_verts - self.true_vertices)**2)
                self.model_error_history.append(mse)
            
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


if __name__ == "__main__":
    # Generate data if not exists (though we already did)
    if not os.path.exists("diamond_data.json"):
        print("Error: diamond_data.json not found.")
        sys.exit(1)
        
    reconstruction = DiamondReconstruction(INPUT_CONFIG)
    final_verts, history, model_errors, trajectory = reconstruction.run_optimization()
    
    reconstruction.verify_results(final_verts)
    
    # Visualization
    plot_optimization_history(history, model_errors=model_errors)
    # plot_3d_comparison is redundant if we have animation, but kept for single view
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

