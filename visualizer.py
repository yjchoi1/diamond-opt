import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_optimization_history(losses, model_errors=None, filename="optimization_history.png"):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Data error (edge length error)', color=color)
    ax1.plot(losses, color=color, label="Data loss")
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    if model_errors is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Model error (vertex location error)', color=color)  # we already handled the x-label with ax1
        ax2.plot(model_errors, color=color, label="Model error")
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Optimization History")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(filename)
    print(f"Saved optimization history to {filename}")

def plot_3d_comparison(optimized_vertices, true_vertices, faces, filename="3d_comparison.html"):
    """
    Visualizes the optimized shape vs true shape using Plotly.
    """
    fig = go.Figure()

    # Helper to create mesh3d traces
    def create_mesh(verts, name, color, opacity):
        # Triangulate faces for mesh3d (Plotly needs triangles for mesh, or we can use i, j, k)
        # But Faces are polygons (4 or 6 vertices).
        # We can just plot the vertices and edges (wireframe) or use Mesh3D with i,j,k.
        # For simplicity, let's plot edges (Wireframe) + Markers.
        
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        
        # Markers
        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color=color),
            name=f"{name} Vertices"
        )
        
        lines = []
        for face in faces:
            # Close loop
            f_loop = face + [face[0]]
            x_loop = x[f_loop]
            y_loop = y[f_loop]
            z_loop = z[f_loop]
            
            lines.append(go.Scatter3d(
                x=x_loop, y=y_loop, z=z_loop,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False
            ))
            
        return [scatter] + lines

    # Plot True Shape
    if true_vertices is not None:
        true_traces = create_mesh(np.array(true_vertices), "True", "blue", 0.3)
        for t in true_traces:
            fig.add_trace(t)

    # Plot Optimized Shape
    opt_traces = create_mesh(np.array(optimized_vertices), "Optimized", "red", 0.5)
    for t in opt_traces:
        fig.add_trace(t)

    fig.update_layout(
        title="True vs Optimized Shape",
        scene=dict(
            aspectmode='data'
        )
    )
    
    fig.write_html(filename)
    print(f"Saved 3D comparison to {filename}")

def plot_3d_animation(trajectory, true_vertices, faces, filename="3d_animation.html"):
    """
    Visualizes the optimization progress using Plotly frames.
    trajectory: list of (N, 3) arrays
    """
    fig = go.Figure()

    # 1. Add Fixed Traces (True Shape)
    if true_vertices is not None:
        true_x, true_y, true_z = true_vertices[:, 0], true_vertices[:, 1], true_vertices[:, 2]
        # Nodes
        fig.add_trace(go.Scatter3d(
            x=true_x, y=true_y, z=true_z,
            mode='markers', marker=dict(size=4, color='blue', opacity=0.3),
            name="True Vertices"
        ))
        # Edges
        for face in faces:
            f_loop = face + [face[0]]
            fig.add_trace(go.Scatter3d(
                x=true_x[f_loop], y=true_y[f_loop], z=true_z[f_loop],
                mode='lines', line=dict(color='blue', width=2), opacity=0.3,
                showlegend=False
            ))

    # 2. Add Initial Frame (First step of trajectory)
    initial_verts = trajectory[0]
    
    # We need a single trace for vertices and multiple for edges?
    # Or one trace for all edges using None to break lines?
    # Plotly requires updating traces.
    # Simplest is to update one Scatter3d for vertices and one Scatter3d for all edges combined.
    
    def get_edge_coords(verts, faces):
        x_lines, y_lines, z_lines = [], [], []
        for face in faces:
            f_loop = face + [face[0]]
            x_lines.extend(verts[f_loop, 0].tolist())
            y_lines.extend(verts[f_loop, 1].tolist())
            z_lines.extend(verts[f_loop, 2].tolist())
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)
        return x_lines, y_lines, z_lines

    ix, iy, iz = get_edge_coords(initial_verts, faces)
    
    # Trace for Optimized Edges
    fig.add_trace(go.Scatter3d(
        x=ix, y=iy, z=iz,
        mode='lines', line=dict(color='red', width=4),
        name="Optimized Edges"
    ))
    
    # Trace for Optimized Vertices
    fig.add_trace(go.Scatter3d(
        x=initial_verts[:, 0], y=initial_verts[:, 1], z=initial_verts[:, 2],
        mode='markers', marker=dict(size=5, color='red'),
        name="Optimized Vertices"
    ))

    # 3. Create Frames
    frames = []
    # Downsample trajectory if too long
    step = max(1, len(trajectory) // 50)
    for i in range(0, len(trajectory), step):
        verts = trajectory[i]
        ex, ey, ez = get_edge_coords(verts, faces)
        
        frames.append(go.Frame(
            data=[
                # Update Edges (Trace index: len(true_traces) + 0) -> Need to be careful with indices.
                # Indices in fig.data: 
                # 0: True Verts
                # 1..N: True Edges (Wait, I added multiple traces for True edges)
                # This is messy. Let's consolidate True Edges too.
                
                # Let's Refactor:
                # Trace 0: True Verts
                # Trace 1: True Edges (Combined)
                # Trace 2: Opt Edges (Combined)
                # Trace 3: Opt Verts
                go.Scatter3d(x=ex, y=ey, z=ez), # Updates Trace 2? No, need to target specific traces.
                go.Scatter3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2]) # Updates Trace 3
            ],
            name=f"frame{i}"
        ))
    
    # Let's restart figure construction to be clean about indices
    fig = go.Figure()
    
    # Trace 0: True Verts
    if true_vertices is not None:
        fig.add_trace(go.Scatter3d(
            x=true_vertices[:,0], y=true_vertices[:,1], z=true_vertices[:,2],
            mode='markers', marker=dict(size=4, color='blue', opacity=0.3),
            name="True Vertices"
        ))
        
    # Trace 1: True Edges
    if true_vertices is not None:
        tx, ty, tz = get_edge_coords(np.array(true_vertices), faces)
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode='lines', line=dict(color='blue', width=2), opacity=0.3,
            name="True Wireframe"
        ))

    # Trace 2: Opt Edges
    ix, iy, iz = get_edge_coords(trajectory[0], faces)
    fig.add_trace(go.Scatter3d(
        x=ix, y=iy, z=iz,
        mode='lines', line=dict(color='red', width=4),
        name="Current Shape"
    ))
    
    # Trace 3: Opt Verts
    fig.add_trace(go.Scatter3d(
        x=trajectory[0][:,0], y=trajectory[0][:,1], z=trajectory[0][:,2],
        mode='markers', marker=dict(size=5, color='red'),
        name="Current Vertices"
    ))

    # Frames
    # We update traces 2 and 3.
    frames = []
    step = max(1, len(trajectory) // 100) # 100 frames max
    for k, i in enumerate(range(0, len(trajectory), step)):
        verts = trajectory[i]
        ex, ey, ez = get_edge_coords(verts, faces)
        
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=ex, y=ey, z=ez), # Matches Trace 2 (first in list effectively?)
                go.Scatter3d(x=verts[:,0], y=verts[:,1], z=verts[:,2]) # Matches Trace 3
            ],
            traces=[2, 3], # Explicitly target traces 2 and 3
            name=f"fr{k}"
        ))
        
    fig.frames = frames

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    fig.update_layout(
        title="Optimization Progress",
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(frames)
            ],
        }],
        scene=dict(aspectmode='data')
    )
    
    fig.write_html(filename)
    print(f"Saved animation to {filename}")


