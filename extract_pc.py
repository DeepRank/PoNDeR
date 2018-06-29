import pkg_resources
import argparse
import os
import numpy as np
import pickle
import torch
import sys

from deeprank.features import AtomicFeature
from deeprank.tools import StructureSimilarity

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from multiprocessing.dummy import Pool as ThreadPool 

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', default=True, action='store_false', help='Train data if True, test data if False')
parser.add_argument('--root_dir', type=str, help='Absolute path to data')

arg = parser.parse_args(['--root_dir','/home/lukas/DR_DATA/'])

if arg.train:
    subfolder='train/'
else:
    subfolder='test/'

pool = ThreadPool(4) 

# Force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features','') + '/forcefield/'
param_charge = FF + 'protein-allhdg5-4_new.top'
param_vdw    = FF + 'protein-allhdg5-4_new.param'
patch_file   = FF + 'patch.top'

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

def convertFolder(native_name):
    if native_name.endswith(".pdb"):
            decoy_dir = arg.root_dir+'decoys/'+native_name[:4]
            if os.path.isdir(decoy_dir):
                for decoy_name in os.listdir(decoy_dir):
                    pc_file = arg.root_dir + 'pointclouds/' + subfolder + decoy_name[:-4] + '.pickle'
                    if os.path.isfile(pc_file):
                        print(decoy_name[:-4],'skipped')
                    else:
                        try:
                            # Declare the feature calculator instance
                            atFeat = AtomicFeature(decoy_dir+'/'+decoy_name, param_charge = param_charge, param_vdw = param_vdw, patch_file = patch_file)

                            # Assign parameters
                            atFeat.assign_parameters()
                            
                            # Compute the pair interactions
                            atFeat.evaluate_pair_interaction()
                            
                            # Get the contact atoms
                            indA, indB = atFeat.sqldb.get_contact_atoms()#extend_to_residue=True ?
                            
                            # Create "point cloud"
                            pc = np.array(atFeat.sqldb.get('x,y,z,eps,sig,charge,chainID',rowID=indA+indB))  
                        
                            # Get iRMSD
                            sim = StructureSimilarity(decoy_dir+'/'+decoy_name,arg.root_dir+'natives/'+native_name)
                            irmsd = sim.compute_irmsd_fast(method='svd',izone=native_name[:4]+'.izone')
                            
                            # Convert encoding
                            integer_encoded = label_encoder.fit_transform(pc[:,6])
                            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                            pc=np.hstack((pc[:,:6],onehot_encoded))
                            pc=np.array(pc).astype(np.float32) # Convert to float because, unfortunately, the sql returns everything as strings
                            
                            # Save file
                            with open(pc_file, "wb") as f:
                                pickle.dump([irmsd,pc], f)
                            print(decoy_name[:-4],'done')
                        except KeyboardInterrupt:
                            sys.exit()
                        except:
                            print(decoy_name[:-4],'did not contain contact atoms')

pool.map(convertFolder, sorted(os.listdir(arg.root_dir+'natives/')))