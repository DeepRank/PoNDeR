import numpy as np
import h5py

hf1 = h5py.File('FINAL_FullDual.h5','r')
hf2 = h5py.File('FINAL_FullSingle.h5', 'w')

def transferGroup(groupName):
    hf2.create_group(groupName)

    group1 = hf1[groupName]
    group2 = hf2[groupName]

    keys = list(group1.keys())
    for key in keys:
        # Read out from old file
        subgroup1 = group1.get(key)
        pcA = np.array(subgroup1.get('A'))
        pcB = np.array(subgroup1.get('B'))
        pcA = np.c_[pcA, np.zeros_like(pcA)] # Pad right
        pcB = np.c_[np.zeros_like(pcB), pcB] # Pad left
        # Convert
        pc  = np.r_[pcA, pcB]
        # Copy everything over to new file
        ds = group2.create_dataset(key, data = pc)
        ds.attrs['irmsd'] = subgroup1.attrs['irmsd']
        ds.attrs['lrmsd'] = subgroup1.attrs['lrmsd']
        ds.attrs['fnat']  = subgroup1.attrs['fnat']
        ds.attrs['dockQ'] = subgroup1.attrs['dockQ']
        print(key, 'done')

transferGroup('train')
transferGroup('test')
transferGroup('holdout')

hf1.close()
hf2.close()