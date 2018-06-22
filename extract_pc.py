from deeprank.features import AtomicFeature
from deeprank.tools import StructureSimilarity
import pkg_resources

# name of the pdb
decoy = '1AK4_100w.pdb'
ref = '1AK4.pdb'

# force field provided with deeprank
FF = pkg_resources.resource_filename('deeprank.features','') + '/forcefield/'

# declare the feature calculator instance
atfeat = AtomicFeature(decoy,
                       param_charge = FF + 'protein-allhdg5-4_new.top',
                       param_vdw    = FF + 'protein-allhdg5-4_new.param',
                       patch_file   = FF + 'patch.top')

# assign parameters
atfeat.assign_parameters()

# only compute the pair interactions here
atfeat.evaluate_pair_interaction()

# get the contact atoms
indA, indB = atfeat.sqldb.get_contact_atoms()

# create the "point cloud"
pc = atfeat.sqldb.get('x,y,z,chainID,name,eps,sig,charge',rowID=indA+indB)


# get the class from the irmsd
sim = StructureSimilarity(decoy,ref)
irmsd = sim.compute_irmsd_fast(method='svd',izone='1AK4.izone')
binclass = not irmsd < 4.0
