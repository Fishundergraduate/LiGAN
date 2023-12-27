import imp
import sys, os, re, time
import pandas as pd
from openbabel import openbabel as ob
import torch
from torch import Tensor, nn, tensor, utils

import molgrid
from . import atom_types, atom_structs, atom_grids
from .atom_types import AtomTyper
from .interpolation import TransformInterpolation
# import packages
# general tools
import numpy as np
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from graphsite import Graphsite
from torchdata import datapipes
from glob import glob
from rdkit import RDLogger, Chem

from torch_geometric.utils import smiles, dense_to_sparse
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import torch
import periodictable as pt
# RDKitのエラーメッセージを無効にする
RDLogger.DisableLog('rdApp.*')
gs = Graphsite()

def __split_train_test(train_ratio=0.9):
    return lambda _: int(np.random.binomial(1,1-train_ratio))

def makePocketDatapipe(datarootDir:str):
    """
    datarootDir: path to data
    data
        |--pocket-data
        |--protein-data-part1
        |--protein-data-part2
    """
    __pock_pipe = datapipes.iter.FileLister(datarootDir+"/pocket-data",recursive=False)
    __pock_pipe = __pock_pipe.filter(filter_fn=lambda x: x.endswith("00.mol2"))
    __pock_pipe = __pock_pipe.map(lambda x:{
        'pockID' : x.split("/")[-1][:5],#Key index
        'pockPath': x
    })
    __prot_pipe_1 = datapipes.iter.FileLister(datarootDir+"/protein-data-part1",recursive=False)
    __prot_pipe_2 = datapipes.iter.FileLister(datarootDir+"/protein-data-part2",recursive=False)
    __prot_pipe = __prot_pipe_1.concat(__prot_pipe_2)
    __prot_pipe = __prot_pipe.filter(filter_fn=lambda x: x.endswith("pops"))
    __prot_pipe = __prot_pipe.map(lambda x:{
        'pockID__prot': x.split("/")[-2],
        'profilePath': ".".join(x.split(".")[:-1])+".profile", # change extension to .profile
        'popsPath':x
    })
    __lig__pipe = datapipes.iter.FileLister(datarootDir+"/pocket-data",recursive=False)
    __lig__pipe = __lig__pipe.filter(filter_fn=lambda x: x.endswith("00.sdf"))
    __lig__pipe = __lig__pipe.map(lambda x:{
        'pockID' : x.split("/")[-1][:4],#Key index
        'ligPath': x
    })
    __lig__pipe = __lig__pipe.map(lambda x:{
        'pockID' : x['pockID'],
        'ligSMI': convert_to_smiles(x['ligPath'])
    })
    __lig__pipe = __lig__pipe.map(lambda x:{
        'pockID__lig' : x['pockID'],
        'ligGraph': create_pytorch_geometric_graph_data_list_from_smiles_and_labels_single(x['ligSMI'])
    })
    

    #TODO sdf -> molGraph

    pipe = __pock_pipe.zip_with_iter(
        __prot_pipe,
        key_fn=lambda x: x['pockID'],
        ref_key_fn=lambda x:x['pockID__prot'],
        merge_fn=lambda x,y:dict(x, **y),
        keep_key=False
    )

    pipe = pipe.zip_with_iter(
        __lig__pipe,
        key_fn=lambda x: x['pockID'],#dict
        ref_key_fn=lambda x:x['pockID__lig'],
        merge_fn=lambda x,y:dict(x, **y),
        keep_key=False
    )
    pipe = pipe.map(lambda x: {
        "proteinGraph":gs(mol_path=x['pockPath'],
                                 profile_path=x['profilePath'],
                                 pop_path=x['popsPath']),
        "LigandGraph":x['ligGraph']})
    return pipe

def getDataLoader(datarootDir:str,batch_size:int,train_ratio=0.8):
    __mySplitter=__split_train_test(train_ratio=train_ratio)
    __train_pipe, __test_pipe = makePocketDatapipe(datarootDir=datarootDir).enumerate().demux(num_instances=2, classifier_fn=__mySplitter)
    return DataLoader(__train_pipe,batch_size=batch_size,shuffle=False), DataLoader(__test_pipe, batch_size)

class biDataset(utils.data.Dataset):
    def __init__(self, datarootDir:str, device:str="cuda"):
        mol2_path = glob(datarootDir+"/pocket/*.mol2")
        pockID = [os.path.splitext(os.path.basename(x))[0] for x in mol2_path]
        dataA = pd.DataFrame({"pockID":pockID,"mol2_path":mol2_path})
        
        lig_path = glob(datarootDir+"/ligand/*.sdf")
        lig_smi = [convert_to_smiles(x) for x in lig_path] # Nullable List
        ligID = [os.path.splitext(os.path.basename(x))[0] for x in lig_path]
        dataB = pd.DataFrame({"ligID":ligID,"lig_smi":lig_smi})


        pops_path = glob(datarootDir+"/protein/*.pops")
        popsID = [os.path.splitext(os.path.basename(x))[0] for x in pops_path]
        dataC = pd.DataFrame({"popsID":popsID,"pops_path":pops_path})
        
        profile_path = glob(datarootDir+"/protein/*.profile")
        profileID = [os.path.splitext(os.path.basename(x))[0] for x in profile_path]
        dataD= pd.DataFrame({"profileID":profileID,"profile_path":profile_path})
        # DataFrameを結合
        # DataFrameを結合
        merged_data = pd.merge(dataA, dataB, left_on='pockID', right_on='ligID')
        merged_data = pd.merge(merged_data, dataC, left_on='pockID', right_on='popsID')
        merged_data = pd.merge(merged_data, dataD, left_on='pockID', right_on='profileID')
        merged_data = merged_data.dropna()
        merged_data = merged_data.drop(columns=['ligID','popsID','profileID'])
        self.data = merged_data
        self.device = device
        self.smiles_graph , error_list= self.smiles2graphlist(self.data['lig_smi'])
        self.data = self.data[~self.data['lig_smi'].isin(error_list)]
        self.prot_graph, error_list = self.prot2graphlist(self.data['mol2_path'],self.data['profile_path'],self.data['pops_path'])
        self.data = self.data[~self.data['mol2_path'].isin(error_list)]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        lig_graph = self.smiles_graph[idx]
        prot_graph = self.prot_graph[idx]

        return lig_graph, prot_graph 

    def smiles2graphlist(self, smiles_list):
        """
        This function takes a list of SMILES strings as input.
        It then attempts to convert each SMILES string into a graph.
        If the conversion is successful, the graph is added to the success_list.
        If the conversion fails (for example, if an invalid SMILES string is provided), the SMILES string is added to the error_list.
        The function ultimately returns two lists: success_list and error_list.
        """
        success_list = list()
        error_list = list()
        for smi in smiles_list:
            try:
                d = smiles.from_smiles(smi)
                d = Data(
                    x=d.x.to(torch.float32),
                    edge_index=d.edge_index.to(torch.long),
                    edge_attr=d.edge_attr.to(torch.float32),
                    smiles = d.smiles)
                success_list.append(d)
            except:
                error_list.append(smi)
        return success_list, error_list
    
    def prot2graphlist(self, mol2_path_list, profile_path_list, pops_path_list):
        """
        This function takes a list of paths to mol2 files, a list of paths to profile files, and a list of paths to pops files as input.
        It then attempts to convert each mol2 file, profile file, and pops file into a graph.
        If the conversion is successful, the graph is added to the success_list.
        If the conversion fails (for example, if an invalid mol2 file, profile file, or pops file is provided), the paths to the mol2 file, profile file, and pops file are added to the error_list.
        The function ultimately returns two lists: success_list and error_list.
        """
        success_list = list()
        error_list = list()
        for mol2_path, profile_path, pops_path in zip(mol2_path_list, profile_path_list, pops_path_list):
            try:
                d = gs(mol_path=mol2_path,profile_path=profile_path,pop_path=pops_path)
                d = Data(
                    x=torch.from_numpy(d[0]).to(torch.float32),
                    edge_index=torch.from_numpy(d[1]).to(torch.long),
                    edge_attr=torch.from_numpy(d[2]).to(torch.float32),
                    y=torch.tensor([0.0]).to(torch.float32))
                success_list.append(d)
            except:
                error_list.append([mol2_path])
        return success_list, error_list

@staticmethod
def convert_to_original_data(data):
    return Data(
        x=data.x.to(torch.long),
        edge_index=data.edge_index.to(torch.long),
        edge_attr=data.edge_attr.to(torch.long),
    )


@staticmethod
def convert_to_edge_attr(adj_matrix:Tensor):
    assert adj_matrix.shape[-1] == 3, "adj_matrix should be 3D tensor"
    edge_bondtype = dense_to_sparse(adj_matrix[:,:,0])
    edge_stereomer = dense_to_sparse(adj_matrix[:,:,1])
    edge_conjugated = dense_to_sparse(adj_matrix[:,:,2])

    edge_attr_stereo = torch.zeros_like(edge_bondtype[1])
    edge_attr_conjugated = torch.zeros_like(edge_bondtype[1])

    if edge_stereomer[0].shape[-1] > 0:
        for i in range(edge_stereomer[0].shape[-1]):
            mask = (edge_bondtype[0][0] == edge_stereomer[0][0][i]) & (edge_bondtype[0][1] == edge_stereomer[0][1][i])
            mask_list = mask.nonzero(as_tuple=True)[0].tolist()
            for idx in mask_list:
                edge_attr_stereo[idx] = edge_stereomer[1][i]
    if edge_conjugated[0].shape[-1] > 0:
        for i in range(edge_conjugated[0].shape[-1]):
            mask = (edge_bondtype[0][0] == edge_conjugated[0][0][i]) & (edge_bondtype[0][1] == edge_conjugated[0][1][i])
            mask_list = mask.nonzero(as_tuple=True)[0].tolist()
            for idx in mask_list:
                edge_attr_conjugated[idx] = edge_conjugated[1][i]
    edge_attr = torch.stack([
        edge_bondtype[1],
        edge_attr_stereo,
        edge_attr_conjugated
    ])
    return edge_bondtype[0], edge_attr.T




class MolDataset(utils.data.IterableDataset):

    def __init__(
        self, rec_typer, lig_typer, data_file, data_root, verbose=False
    ):
        super().__init__()

        # what is this unknown column?
        #  it's positive for low_rmsd, negative for ~low_rmsd,
        #  but otherwise same absolute distributions...
        data_cols = [
            'low_rmsd',
            'true_aff',
            'xtal_rmsd',
            'rec_src',
            'lig_src',
            'vina_aff'
        ]
        self.data = pd.read_csv(
            data_file, sep=' ', names=data_cols, index_col=False
        )
        self.root_dir = data_root

        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat('pdb')
        self.read_pdb = ob_conv.ReadFile

        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat('sdf')
        self.read_sdf = ob_conv.ReadFile

        self.mol_cache = dict()
        self.verbose = verbose

        self.rec_typer = rec_typer
        self.lig_typer = lig_typer

    def read_mol(self, mol_src, pdb=False):

        mol_file = os.path.join(self.root_dir, mol_src)
        if self.verbose:
            print('Reading ' + mol_file)

        assert os.path.isfile(mol_file), 'file does not exist'

        mol = ob.OBMol()
        if pdb:
            assert self.read_pdb(mol, mol_file), 'failed to read mol'
        else:
            assert self.read_sdf(mol, mol_file), 'failed to read mol'

        mol.AddHydrogens()
        assert mol.NumAtoms() > 0, 'mol has zero atoms'

        mol.SetTitle(mol_src)
        return mol

    def get_rec_mol(self, mol_src):
        if mol_src not in self.mol_cache:
            self.mol_cache[mol_src] = self.read_mol(mol_src, pdb=True)
        return self.mol_cache[mol_src]

    def get_lig_mol(self, mol_src):
        if mol_src not in self.mol_cache:
            self.mol_cache[mol_src] = self.read_mol(mol_src, pdb=False)
        return self.mol_cache[mol_src]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        rec_mol = self.get_rec_mol(example.rec_src)
        lig_mol = self.get_lig_mol(example.lig_src)
        return rec_mol, lig_mol

    def __iter__(self):
        for rec_src, lig_src in zip(self.data.rec_src, self.data.lig_src):
            rec_mol = self.get_rec_mol(rec_src)
            lig_mol = self.get_lig_mol(lig_src)
            yield rec_mol, lig_mol


class AtomGridData(object):

    def __init__(
        self,
        data_file,
        data_root,
        batch_size,
        rec_typer,
        lig_typer,
        use_rec_elems=True,
        resolution=0.5,
        dimension=None,
        grid_size=None,
        shuffle=False,
        random_rotation=False,
        random_translation=0.0,
        diff_cond_transform=False,
        diff_cond_structs=False,
        n_samples=1,
        rec_molcache=None,
        lig_molcache=None,
        cache_structs=True,
        device='cuda',
        debug=False,
    ):
        super().__init__()

        assert (dimension or grid_size) and not (dimension and grid_size), \
            'must specify one of either dimension or grid_size'
        if grid_size:
            dimension = atom_grids.size_to_dimension(grid_size, resolution)
        
        # create receptor and ligand atom typers
        self.lig_typer = AtomTyper.get_typer(
            *lig_typer.split('-'), rec=False, device=device
        )
        self.rec_typer = AtomTyper.get_typer(
            *rec_typer.split('-'), rec=use_rec_elems, device=device
        )

        atom_typers = [self.rec_typer, self.lig_typer]
        if diff_cond_structs: # duplicate atom typers
            atom_typers *= 2

        # create example provider
        self.ex_provider = molgrid.ExampleProvider(
            *atom_typers,
            data_root=data_root,
            recmolcache=rec_molcache or '',
            ligmolcache=lig_molcache or '',
            cache_structs=cache_structs,
            shuffle=shuffle,
            num_copies=n_samples,
        )

        # create molgrid maker
        self.grid_maker = molgrid.GridMaker(
            resolution=resolution,
            dimension=dimension,
            gaussian_radius_multiple=-1.5,
        )
        self.batch_size = batch_size

        # transformation settings
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.diff_cond_transform = diff_cond_transform
        self.diff_cond_structs = diff_cond_structs
        self.debug = debug
        self.device = device

        # transform interpolation state
        self.cond_interp = TransformInterpolation(n_samples=n_samples)

        # load data from file
        self.ex_provider.populate(data_file)

    @classmethod
    def from_param(cls, param):

        return cls(
            data_root=param.root_folder,
            batch_size=param.batch_size,
            rec_typer=param.recmap,
            lig_typer=param.ligmap,
            resolution=param.resolution,
            grid_size=atom_grids.dimension_to_size(
                param.dimension, param.resolution
            ),
            shuffle=param.shuffle,
            random_rotation=param.random_rotation,
            random_translation=param.random_translate,
            rec_molcache=param.recmolcache,
            lig_molcache=param.ligmolcache,
        )

    @property
    def root_dir(self):
        return self.ex_provider.settings().data_root

    @property
    def n_rec_channels(self):
        return self.rec_typer.num_types() if self.rec_typer else 0

    @property
    def n_lig_channels(self):
        return self.lig_typer.num_types() if self.lig_typer else 0
 
    @property
    def n_channels(self):
        return self.n_rec_channels + self.n_lig_channels

    @property
    def resolution(self):
        return self.grid_maker.get_resolution()

    @property
    def dimension(self):
        return self.grid_maker.get_dimension()

    @property
    def grid_size(self):
        return atom_grids.dimension_to_size(self.dimension, self.resolution)

    def __len__(self):
        return self.ex_provider.size()

    def forward(self, interpolate=False, spherical=False):
        assert len(self) > 0, 'data is empty'

        # get next batch of structures
        examples = self.ex_provider.next_batch(self.batch_size)
        labels = torch.zeros(self.batch_size, device=self.device)
        examples.extract_label(0, labels)

        # create lists for examples, structs and transforms
        batch_list = lambda: [None] * self.batch_size

        input_examples = batch_list()
        input_rec_structs = batch_list()
        input_lig_structs = batch_list()
        input_transforms = batch_list()

        cond_examples = batch_list()
        cond_rec_structs = batch_list()
        cond_lig_structs = batch_list()
        cond_transforms = batch_list()

        # create output tensors for atomic density grids
        input_grids = torch.zeros(
            self.batch_size,
            self.n_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device=self.device,
        )
        cond_grids = torch.zeros(
            self.batch_size,
            self.n_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device=self.device,
        )

        # split examples, create structs and transforms
        for i, ex in enumerate(examples):

            if self.diff_cond_structs:

                # different input and conditional molecules
                input_rec_coord_set, input_lig_coord_set, \
                    cond_rec_coord_set, cond_lig_coord_set = ex.coord_sets

                # split example into inputs and conditions
                input_ex = molgrid.Example()
                input_ex.coord_sets.append(input_rec_coord_set)
                input_ex.coord_sets.append(input_lig_coord_set)

                cond_ex = molgrid.Example()
                cond_ex.coord_sets.append(cond_rec_coord_set)
                cond_ex.coord_sets.append(cond_lig_coord_set)

            else: # same conditional molecules as input
                input_rec_coord_set, input_lig_coord_set = ex.coord_sets
                cond_rec_coord_set, cond_lig_coord_set = ex.coord_sets
                input_ex = cond_ex = ex

            # store split examples for gridding
            input_examples[i] = input_ex
            cond_examples[i] = cond_ex

            # convert coord sets to atom structs
            input_rec_structs[i] = atom_structs.AtomStruct.from_coord_set(
                input_rec_coord_set,
                typer=self.rec_typer,
                data_root=self.root_dir,
                device=self.device
            )
            input_lig_structs[i] = atom_structs.AtomStruct.from_coord_set(
                input_lig_coord_set,
                typer=self.lig_typer,
                data_root=self.root_dir,
                device=self.device
            )
            if self.diff_cond_structs:
                cond_rec_structs[i] = atom_structs.AtomStruct.from_coord_set(
                    cond_rec_coord_set,
                    typer=self.rec_typer,
                    data_root=self.root_dir,
                    device=self.device
                )
                cond_lig_structs[i] = atom_structs.AtomStruct.from_coord_set(
                    cond_lig_coord_set,
                    typer=self.lig_typer,
                    data_root=self.root_dir,
                    device=self.device
                )
            else: # same structs as input
                cond_rec_structs[i] = input_rec_structs[i]
                cond_lig_structs[i] = input_lig_structs[i]

            # create input transform
            input_transforms[i] = molgrid.Transform(
                center=input_lig_coord_set.center(),
                random_translate=self.random_translation,
                random_rotation=self.random_rotation,
            )
            if self.diff_cond_transform:

                # create conditional transform
                cond_transforms[i] = molgrid.Transform(
                    center=cond_lig_coord_set.center(),
                    random_translate=self.random_translation,
                    random_rotation=self.random_rotation,
                )
            else: # same transform as input
                cond_transforms[i] = input_transforms[i]
        
        if interpolate: # interpolate conditional transforms
            # i.e. location and orientation of conditional grid
            if not self.cond_interp.is_initialized:
                self.cond_interp.initialize(cond_examples[0])
            cond_transforms = self.cond_interp(
                transforms=cond_transforms,
                spherical=spherical,
            )

        # create density grids
        for i in range(self.batch_size):

            # create input density grid
            self.grid_maker.forward(
                input_examples[i],
                input_transforms[i],
                input_grids[i]
            )
            if (
                self.diff_cond_transform or self.diff_cond_structs or interpolate
            ):
                # create conditional density grid
                self.grid_maker.forward(
                    cond_examples[i],
                    cond_transforms[i],
                    cond_grids[i]
                )
            else: # same density grid as input
                cond_grids[i] = input_grids[i]

        input_structs = (input_rec_structs, input_lig_structs)
        cond_structs = (cond_rec_structs, cond_lig_structs)
        transforms = (input_transforms, cond_transforms)
        return (
            input_grids, cond_grids,
            input_structs, cond_structs,
            transforms, labels
        )

    def split_channels(self, grids):
        '''
        Split receptor and ligand grid channels.
        '''
        return torch.split(
            grids, [self.n_rec_channels, self.n_lig_channels], dim=1
        )

    def find_real_mol(self, mol_src, ext):
        return find_real_mol(mol_src, self.root_dir, ext)


def find_real_mol(mol_src, data_root, ext):

    m = re.match(r'(.+)_(\d+)((\..*)+)', mol_src)
    if m:
        mol_name = m.group(1)
        pose_idx = int(m.group(2))
    else:
        m = re.match(r'(.+)((\..*)+)', mol_src)
        mol_name = m.group(1)
        pose_idx = 0

    mol_file = os.path.join(data_root, mol_name + ext)
    return mol_file, mol_name, pose_idx
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels_single(x_smiles, y=0.0):
    """
    Inputs:
    
    x_smiles = smiles SMILES strings
    y = y numerial labels for the SMILES strings (such as associated pKi values)
    TODO: y should be QED value from rdkit
    Outputs:
    
    data_list = G torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    if x_smiles is None:
        return None
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(x_smiles)
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
        
    X = torch.tensor(X, dtype = torch.float)
    
    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim = 0)
    
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    
    EF = torch.tensor(EF, dtype = torch.float)
    
    # construct label tensor
    y_tensor = torch.tensor(np.array([y]), dtype = torch.float)
    
    # construct Pytorch Geometric data object and append to data list
    return Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor)

def convert_to_smiles(input_sdf)->str:
    # Create a molecule object from the input file
    suppl = Chem.SDMolSupplier(input_sdf)
    mols = [x for x in suppl if x is not None]
    return Chem.MolToSmiles(mols[0]) if len(mols)!=0 else None

def extract_options_from_sdf(sdf_file_path)->list[float]:
    # SDFファイルを読み込む
    supplier = Chem.SDMolSupplier(sdf_file_path)

    options_list = []

    # 各分子の情報を抽出
    for mol in supplier:
        if mol is not None:
            # 分子の全てのプロパティを取得
            options = {}
            for prop_name in mol.GetPropNames():
                prop_value = mol.GetProp(prop_name)
                options[prop_name] = prop_value

            options_list.append(options)

    return options_list

def pdb_to_graph(pdb_file, threshold=5.0):
    """
        Returns node_feature and edge_index
        #TODO: atom_feature should be extended
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    atoms = list(model.get_atoms())
    coords = np.array([atom.get_coord() for atom in atoms])
    features = np.array([pt.elements.symbol(atom.element).number for atom in atoms])
    distance_matrix = cdist(coords, coords)
    edge_index = np.argwhere(distance_matrix <= threshold)
    return features, edge_index
    data = Data(x=torch.tensor(features, dtype=torch.float).view(-1, 1), edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous())
    return data