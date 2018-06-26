import pkg_resources
import argparse
import os
import numpy as np
import pickle

from deeprank.features import AtomicFeature
from deeprank.tools import StructureSimilarity

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--decoy_path', type=str, help='Path (absolute) to decoy folders')
parser.add_argument('--native_dir',type=str, help='Path (absolute) to native files')

arg = parser.parse_args(['--decoy_path','/home/lukas/DR_DATA/decoys','--native_dir','/home/lukas/DR_DATA/natives'])

# force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features','') + '/forcefield/'
param_charge = FF + 'protein-allhdg5-4_new.top'
param_vdw    = FF + 'protein-allhdg5-4_new.param'
patch_file   = FF + 'patch.top'

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

for native_name in os.listdir(arg.native_dir):
    if native_name.endswith(".pdb"):
        decoy_dir = arg.decoy_path+'/'+native_name[:4]
        if os.path.isdir(decoy_dir):
            irmsds = []
            pcs = []
            for decoy_name in os.listdir(decoy_dir):
                
                # Declare the feature calculator instance
                atFeat = AtomicFeature(decoy_dir+'/'+decoy_name, param_charge = param_charge, param_vdw = param_vdw, patch_file = patch_file)

                # Assign parameters
                atFeat.assign_parameters()
                
                # Compute the pair interactions
                atFeat.evaluate_pair_interaction()
                
                # Get the contact atoms
                indA, indB = atFeat.sqldb.get_contact_atoms()
                
                # Create "point cloud"
                pc = np.array(atFeat.sqldb.get('x,y,z,eps,sig,charge,chainID',rowID=indA+indB))
            
                # Get iRMSD
                sim = StructureSimilarity(decoy_dir+'/'+decoy_name,arg.native_dir+'/'+native_name)
                irmsd = sim.compute_irmsd_fast(method='svd',izone=native_name[:4]+'.izone')

                integer_encoded = label_encoder.fit_transform(pc[:,6])
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                pc=np.hstack((pc[:,:6],onehot_encoded))
                irmsds.append(irmsd)
                pcs.append(pc)
    pc_file = '/home/lukas'+native_name+'.pickle'
    with open(pc_file, "wb") as f:
        pickle.dump([irmsds,pcs], f)