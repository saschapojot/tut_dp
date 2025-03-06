import numpy as np
from ase import Atoms
from ase.visualize import view

# Load data
data_dir="./examples/water/data/data_3/"
data_folder=data_dir+"/set.000/"


types = np.loadtxt(data_dir+'type.raw', dtype=int)
type_map = np.loadtxt(data_dir+'type_map.raw', dtype=str)
coords = np.load(data_folder+'coord.npy')  # Shape: (n_frames, n_atoms, 3)
boxes = np.load(data_folder+'box.npy')     # Shape: (n_frames, 9)

# Fix coordinate dimensions
frame_idx = 0
n_atoms = len(types)
positions = coords[frame_idx].reshape(n_atoms, 3)  # Critical reshape

# Create ASE atoms
symbols = [type_map[t] for t in types]
atoms = Atoms(symbols=symbols,
              positions=positions,
              cell=np.load(data_folder+'box.npy')[frame_idx].reshape(3,3),
              pbc=True)

view(atoms)