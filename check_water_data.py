import numpy as np
import matplotlib.pyplot as plt
#this script loads and displays data in water example

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


ind=5

one_row_box=box_data[ind,:]

basis=one_row_box.reshape((3,3)).T

basis_inv=np.linalg.inv(basis)

one_row_coord=coord_data[ind,:]

one_row_coord_arr=one_row_coord.reshape((-1,3))

one_row_coord_arr_T=one_row_coord_arr.T

frac_coord=basis_inv@one_row_coord_arr_T

# Find out-of-range indices
rows, cols = np.where((frac_coord < 0) | (frac_coord > 1))
print("Row indices:", rows)  # Output: [1]
print("Column indices:", cols)  # Output: [1]

# Get values
print(frac_coord[:,5])  # Output: [1.2]

