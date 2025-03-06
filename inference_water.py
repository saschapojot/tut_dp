import numpy as np
import matplotlib.pyplot as plt
from deepmd.infer import DeepPot

#this script loads  data in water example
# and makes inference
inDataDir="./examples/water/data/data_3/set.000/"

box_file=inDataDir+"/box.npy"

coord_file=inDataDir+"/coord.npy"

energy_file=inDataDir+"/energy.npy"

force_file=inDataDir+"/force.npy"
############################
box_data=np.load(box_file)

coord_data=np.load(coord_file)
energy_data=np.load(energy_file)
force_data=np.load(force_file)


###################inference
ind=37
cell = box_data[ind,:]


coord = np.array([
    0.0, 0.0, 0.0,    # O (atom 1)
    0.95, 0.0, 0.0,   # H (atom 2)
    -0.32, 0.90, 0.0  # H (atom 3)
], dtype=np.float32)+2.0

# Reshape for inference (1 frame, 3 atoms)
coord = coord.reshape(1, -1)  # Shape: [1, 9]
atype = [0, 1, 1]  # O, H, H

inModelFile="./examples/water/se_e2_a/compressed_model_water.pth"

model=DeepPot(inModelFile)


# Predict
energy, force, virial = model.eval(coord, cell, atype)

print(f"Energy: {energy[0]} eV")
print(f"Forces (eV/Å):\n{force.reshape(-1, 3)}")