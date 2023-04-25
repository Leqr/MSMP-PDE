
#%%
from equations.PDEs import *
from common.utils import HDF5Dataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

# Load datasets
pde = "AD"
exp = "RPU"
train_string = f'data/{pde}_train_{exp}.h5'
valid_string = f'data/{pde}_valid_{exp}.h5'
test_string = f'data/{pde}_test_{exp}.h5'
super_resolution=[250,200]
base_resolution=[250,100]

train_dataset = HDF5Dataset(train_string, pde=pde, mode='train', base_resolution=base_resolution, super_resolution=super_resolution)
train_loader = DataLoader(train_dataset,
                            batch_size=1024,
                            shuffle=True)

valid_dataset = HDF5Dataset(valid_string, pde=pde, mode='valid', base_resolution=base_resolution, super_resolution=super_resolution)
valid_loader = DataLoader(valid_dataset,
                            batch_size=124,
                            shuffle=False)

test_dataset = HDF5Dataset(test_string, pde=pde, mode='test', base_resolution=base_resolution, super_resolution=super_resolution)
test_loader = DataLoader(test_dataset,
                            batch_size=124,
                            shuffle=False)
#%%
try:
    os.remove(f'data/{pde}_train_{exp}_I.h5')
except:
    print(".")
try:
    os.remove(f'data/{pde}_valid_{exp}_I.h5')
except:
    print(".")
try:
    os.remove(f'data/{pde}_test_{exp}_I.h5')
except:
    print(".")

#%%
x0 = 0
xL = 16

new_train_string = f'data/{pde}_train_{exp}_I.h5'
new_valid_string = f'data/{pde}_valid_{exp}_I.h5'
new_test_string = f'data/{pde}_test_{exp}_I.h5'

mode = ["train","valid","test"]
array = [new_train_string, new_valid_string, new_test_string]
array_dataset = [train_dataset, valid_dataset, test_dataset]
data_store = None
#go through test, train, valid
for m, s, dataset in zip(mode, array, array_dataset):   
    print(f"***** Mode : {m} *****")
    h5f = h5py.File(s, 'a')
    new_dataset = h5f.create_group(m)

    t = {}
    x = {}
    h5f_u = {}
    h5f_a = {}
    h5f_b = {}

    #go through the keys --> resolutions + params --> discard params
    for key in dataset.data:
        if key == "a":
            h5f_a[key] = new_dataset.create_dataset(key, dataset.data[key].shape, dtype=float)
            h5f_a[key] = np.array(dataset.data[key])
        if key == "b":
            h5f_b[key] = new_dataset.create_dataset(key, dataset.data[key].shape, dtype=float)
            h5f_b[key] = np.array(dataset.data[key])
        #discard params
        if len(key.split("-")) > 1:
            print(f"Interpolating {key}")
            nx = key.split("-")[-1]
            x_struct = np.linspace(x0,xL,int(nx))
            x_rand = dataset.data[key].attrs['x']
            t[key] = torch.linspace(dataset.tmin,dataset.tmax, dataset.nt)
            x[key] = x_struct
            h5f_u[key] = new_dataset.create_dataset(key, dataset.data[key].shape, dtype=float)
            h5f[m][key].attrs['dt'] = dataset.dt
            h5f[m][key].attrs['dx'] = dataset.dx
            h5f[m][key].attrs['nt'] = dataset.nt
            h5f[m][key].attrs['nx'] = nx
            h5f[m][key].attrs['tmin'] = dataset.tmin
            h5f[m][key].attrs['tmax'] = dataset.tmax
            h5f[m][key].attrs['x'] = x[key]
            #[nb, dim, nt, nx]
            for b,batch_data in enumerate(dataset.data[key]):
                #[dim, nt, nx]
                for d,dim_data in enumerate(batch_data):
                    #[nt, nx]
                    for ti,time_data in enumerate(dim_data):
                        #[nx]
                        f = interpolate.interp1d(x_rand, time_data)
                        new_time_data = f(x_struct)
                        h5f_u[key][b,d,ti,:] = new_time_data
    h5f.close()  

# %%

