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

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='Absolute path to data')

arg = parser.parse_args(['--root_dir', '/home/lukas/DR_DATA/'])

# Force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features', '') + '/forcefield/'
param_charge = FF + 'protein-allhdg5-4_new.top'
param_vdw = FF + 'protein-allhdg5-4_new.param'
patch_file = FF + 'patch.top'

# Prepare folders
os.makedirs('lzone', exist_ok=True)
os.makedirs('izone', exist_ok=True)
os.makedirs('refpairs', exist_ok=True)

# Prepare HDF5 file & groups
hf = h5py.File('pointclouds.h5', 'w')

g_test = hf.create_group('test')
g_train = hf.create_group('train')
g_holdout = hf.create_group('holdout')

# Random distribution of protein pairs among groups
def getGroup(native_name):
    rand = random.random()
    if rand < 0.8:
        group = g_train
    elif rand < 0.9:
        group = g_test
    else:
        group = g_holdout
    return group
i = 0

# Start converting
for native_name in sorted(os.listdir(arg.root_dir+'natives/')):
    decoy_dir = arg.root_dir+'decoys/'+native_name[:4]
    if os.path.isdir(decoy_dir):
        group = getGroup(native_name)
        i+=1
        print('Putting', native_name[:4], 'in', group.name)
        for decoy_name in os.listdir(decoy_dir):
            # Declare the feature calculator instance
            atFeat = AtomicFeature(decoy_dir+'/'+decoy_name, param_charge=param_charge, param_vdw=param_vdw, patch_file=patch_file)

            # Assign parameters
            atFeat.assign_parameters()
            
            try:
                # Compute the pair interactions
                atFeat.evaluate_pair_interaction()

                # Get contact pairs, append features
                    # x, y, z   -> Coordinates
                    # occ       -> Occupancy
                    # temp      -> Temperature factor (uncertainty)
                    # eps       -> 
                    # sig       -> Sigma (?)
                    # charge    ->
                pc_pairs = []
                index = atFeat.sqldb.get_contact_atoms(return_contact_pairs=True)
                for key,val in index.items():
                    pc1 = atFeat.sqldb.get('x,y,z,eps,sig,charge',rowID=key)[0]
                    pc2 = atFeat.sqldb.get('x,y,z,eps,sig,charge',rowID=val)
                    a = np.array(pc1[0:3], dtype=np.float32)
                    
                    for p in pc2:
                        b = np.array(p[0:3], dtype=np.float32)
                        dist = np.linalg.norm(a-b) # Euclidian distance
                        pc_pairs.append(pc1+p+dist)
                        

                # List of atom pair parameters to array
                pc = np.vstack(pc_pairs).astype(np.float32) 

                # Get metrics
                sim = StructureSimilarity(decoy_dir+'/'+decoy_name, arg.root_dir+'natives/'+native_name)
                irmsd = sim.compute_irmsd_fast(method='svd', izone='izone/'+native_name[:4]+'.izone')
                lrmsd = sim.compute_lrmsd_fast(method='svd',lzone='lzone/'+native_name[:4]+'.lzone')
                fnat = sim.compute_Fnat_fast(ref_pairs='refpairs/'+native_name[:4]+'.refpairs')
                dockQ = sim.compute_DockQScore(fnat,lrmsd,irmsd)

                # Save file
                ds = group.create_dataset(decoy_name[:-4], data = pc)
                ds.attrs['irmsd'] = irmsd
                ds.attrs['lrmsd'] = lrmsd
                ds.attrs['fnat'] = fnat
                ds.attrs['dockQ'] = dockQ
                print('    ',decoy_name[:-4], 'done')
            except KeyboardInterrupt:
                hf.close()
                sys.exit()
            except:
                print(decoy_name[:-4], 'did not contain contact atoms')
    if i == 10: break
hf.close()