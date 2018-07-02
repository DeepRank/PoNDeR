import pkg_resources
import argparse
import os
import numpy as np
import torch
import sys
import h5py
import random

from deeprank.features import AtomicFeature
from deeprank.tools import StructureSimilarity

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='Absolute path to data')

arg = parser.parse_args(['--root_dir', '/home/lukas/DR_DATA/'])

""" # Force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features', '') + '/forcefield/'
param_charge = FF + 'protein-allhdg5-4_new.top'
param_vdw = FF + 'protein-allhdg5-4_new.param'
patch_file = FF + 'patch.top'

hf = h5py.File('pointclouds.h5', 'w')

g_test = hf.create_group('test')
g_train = hf.create_group('train')
g_holdout = hf.create_group('holdout')

def getSubgroup(native_name):
    rand = random.random()
    if rand < 0.8:
        subgroup = g_train.create_group(native_name[:4])
    elif rand < 0.9:
        subgroup = g_test.create_group(native_name[:4])
    else:
        subgroup = g_holdout.create_group(native_name[:4])
    return subgroup

for native_name in sorted(os.listdir(arg.root_dir+'natives/')):
    if native_name.endswith(".pdb"):
        decoy_dir = arg.root_dir+'decoys/'+native_name[:4]
        if os.path.isdir(decoy_dir):
            subgroup = getSubgroup(native_name)
            print('Putting', native_name[:4], 'in', subgroup.name)
            for decoy_name in sorted(os.listdir(decoy_dir)):
                # Declare the feature calculator instance
                atFeat = AtomicFeature(decoy_dir+'/'+decoy_name, param_charge=param_charge, param_vdw=param_vdw, patch_file=patch_file)

                # Assign parameters
                atFeat.assign_parameters()
                
                try:
                    # Compute the pair interactions
                    atFeat.evaluate_pair_interaction()

                    # Get the contact atoms
                    indA, indB = atFeat.sqldb.get_contact_atoms()

                    # Create "point cloud"
                    pcA = np.array(atFeat.sqldb.get('x,y,z,eps,sig,charge', rowID=indA))
                    pcB = np.array(atFeat.sqldb.get('x,y,z,eps,sig,charge', rowID=indB))
                    pcA = np.c_[pcA, np.zeros_like(pcA)]
                    pcB = np.c_[pcB, np.zeros_like(pcB)]
                    pc = np.r_[pcA, pcB].astype(np.float32)

                    # Get iRMSD
                    sim = StructureSimilarity(decoy_dir+'/'+decoy_name, arg.root_dir+'natives/'+native_name)
                    irmsd = sim.compute_irmsd_fast(method='svd', izone='izone/'+native_name[:4]+'.izone')

                    # Save file
                    ds = subgroup.create_dataset(decoy_name[:-4], data = pc)
                    ds.attrs['irmsd'] = irmsd
                    print(decoy_name[:-4], 'done')
                except KeyboardInterrupt:
                    hf.close()
                #except:
                    #print(decoy_name[:-4], 'did not contain contact atoms')
                break """
    

hq = h5py.File('pointclouds.h5', 'r')
train = hq.get('train/1AVX')
ds=train.get('1AVX_100w')
print(ds.dtype)
print(np.array(ds))