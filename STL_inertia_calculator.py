import trimesh
import numpy as np

stl_path = 'F8_lightning.stl' 
total_mass = 153775 

mesh = trimesh.load(stl_path, force='mesh')
mesh.density = total_mass / mesh.volume

center_of_mass = mesh.center_mass
inertia_tensor = mesh.moment_inertia

print(f"Center of Mass (CoM): {np.round(center_of_mass, 5)} m")
print("Inertia Tensor (relative to the CoM):")
print(np.round(inertia_tensor, 6))
