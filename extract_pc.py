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

'''
KNOWN ERROR

Decoy folders can't contain any files that aren't pdb's.
This can be implemented through a check or just running following command while in the folder:
    find . -type -f ! -name "*.pdb" -delete
'''

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/home/lukas/DR_DATA/', help='Path to data')
parser.add_argument('--decoy_dir', type=str, default='decoys/', help='Relative path to decoys')
parser.add_argument('--decoy_subdir', type=str, default='', help='Subfolder within specific decoy folder (e.g. water/)')
parser.add_argument('--native_dir', type=str, default='natives/', help='Relative path to natives')
parser.add_argument('--dual', dest='dual', default=False, action='store_true',help='Store pointclouds of different proteins separately')
arg = parser.parse_args()

# Force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features', '') + '/forcefield/'
param_charge = FF + 'protein-allhdg5-4_new.top'
param_vdw = FF + 'protein-allhdg5-4_new.param'
patch_file = FF + 'patch.top'

# Prepare HDF5 file & groups
if arg.dual:
    filename = 'dualPointclouds.h5'
else:
    filename = 'pointclouds.h5'

hf = h5py.File(filename, 'w')

g_test = hf.create_group('test')
g_train = hf.create_group('train')
g_holdout = hf.create_group('holdout')

# Random distribution of protein pairs among groups
def getGroup(native_name):
    rand = random.random()
    if rand < 0.7:
        group = g_train
    elif rand < 0.85:
        group = g_test
    else:
        group = g_holdout
    return group

# Start converting
for native_name in sorted(os.listdir(arg.root_dir+arg.native_dir)):
    decoy_dir = arg.root_dir+arg.decoy_dir+native_name[:4]+'/'+arg.decoy_subdir
    if os.path.isdir(decoy_dir):
        group = getGroup(native_name)
        print('Putting', native_name[:4], 'in', group.name)
        for decoy_name in sorted(os.listdir(decoy_dir)):
            # Declare the feature calculator instance
            atFeat = AtomicFeature(decoy_dir+'/'+decoy_name, param_charge=param_charge, param_vdw=param_vdw, patch_file=patch_file)

            # Assign parameters
            atFeat.assign_parameters()
            
            # Compute the pair interactions
            atFeat.evaluate_pair_interaction()

            # Get contact atoms, append features
                # x, y, z   -> Coordinates
                # occ       -> Occupancy
                # temp      -> Temperature factor (uncertainty)
                # eps       -> 
                # sig       -> Sigma
                # charge    ->
            indA, indB = atFeat.sqldb.get_contact_atoms(cutoff=6.5)

            if len(indA)==0: # If no contact atoms found
                print('    ', decoy_name[:-4], 'did not contain contact atoms')
            else: 
                pcA = np.array(atFeat.sqldb.get('x,y,z,eps,sig,charge', rowID=indA)).astype(np.float32)
                pcB = np.array(atFeat.sqldb.get('x,y,z,eps,sig,charge', rowID=indB)).astype(np.float32)

                if not arg.dual:
                    pcA = np.c_[pcA, np.zeros_like(pcA)]
                    pcB = np.c_[np.zeros_like(pcB), pcB]
                    pc = np.r_[pcA, pcB].astype(np.float32)

                # Get metrics
                sim = StructureSimilarity(decoy_dir+'/'+decoy_name, arg.root_dir+arg.native_dir+native_name)
                irmsd = sim.compute_irmsd_fast(method='svd')
                lrmsd = sim.compute_lrmsd_fast(method='svd')
                fnat = sim.compute_Fnat_fast()
                dockQ = sim.compute_DockQScore(fnat,lrmsd,irmsd)

                # Save file
                if arg.dual:
                    subgroup = group.create_group(decoy_name[:-4])
                    dsA = subgroup.create_dataset('A', data = pcA)
                    dsB = subgroup.create_dataset('B', data = pcB)
                    subgroup.attrs['irmsd'] = irmsd
                    subgroup.attrs['lrmsd'] = lrmsd
                    subgroup.attrs['fnat']  = fnat
                    subgroup.attrs['dockQ'] = dockQ
                else:
                    ds = group.create_dataset(decoy_name[:-4], data = pc)
                    ds.attrs['irmsd'] = irmsd
                    ds.attrs['lrmsd'] = lrmsd
                    ds.attrs['fnat']  = fnat
                    ds.attrs['dockQ'] = dockQ
                print('    ',decoy_name[:-4], 'done')
    else:
        print(decoy_dir, 'not found')
hf.close()