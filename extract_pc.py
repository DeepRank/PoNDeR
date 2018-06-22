import pkg_resources
import argparse
import os

from deeprank.features import AtomicFeature
from deeprank.tools import StructureSimilarity

parser = argparse.ArgumentParser()
parser.add_argument('--decoy_path', type=str, help='Path (absolute) to decoy files')
parser.add_argument('--native_path',type=int, help='Path (absolute) to native files')

arg = parser.parse_args(['--decoy_path','/home/lukas/DR_DATA/decoys','--native_path','/home/lukas/DR_DATA/natives'])

# force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features','') + '/forcefield/'

param_charge = FF + 'protein-allhdg5-4_new.top'
param_vdw    = FF + 'protein-allhdg5-4_new.param'
patch_file   = FF + 'patch.top'

for native_name in os.listdir(arg.native_path):
    if native_name.endswith(".pdb"):
        for decoy_name in os.listdir(arg.decoy_path+'/'+native_name[:4]):
            # declare the feature calculator instance
            atfeat = AtomicFeature(decoy_name, param_charge = param_charge, param_vdw = param_vdw, patch_file = patch_file)

            # assign parameters
            atfeat.assign_parameters()

            # only compute the pair interactions here
            atfeat.evaluate_pair_interaction()

            # get the contact atoms
            indA, indB = atfeat.sqldb.get_contact_atoms()

            # create the "point cloud"
            pc = atfeat.sqldb.get('x,y,z,chainID,name,eps,sig,charge',rowID=indA+indB)

            # get the class from the irmsd
            sim = StructureSimilarity(decoy_name,native_name)
            irmsd = sim.compute_irmsd_fast(method='svd',izone='1AK4.izone')
            binclass = not irmsd < 4.0