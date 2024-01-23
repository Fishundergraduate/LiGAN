import sys, os, re, time, gzip, itertools
from collections import defaultdict
from pathlib import Path

from sklearn import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 32)
pd.set_option('display.max_rows', 120)
import torch
from rdkit import Chem

import ligan
from ligan import models, data
from ligan.atom_grids import AtomGrid
from ligan.atom_structs import AtomStruct
from ligan import molecules as mols

from torch_geometric.data import DataLoader, Data, Batch
import glob
from torch_geometric.utils import smiles as gsmiles
from torch_geometric.utils import dense_to_sparse
from torch import Tensor
from torch.nn import functional as F
MB = 1024 ** 2

    
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
class MoleculeGenerator(object):
    '''
    Base class for generating 2D molecules
    using a generative model, atom fitting,
    and bond adding algorithms.
    '''
    # subclasses override these class attributes
    gen_model_type = None
    has_complex_input = False

    def __init__(
        self,
        out_prefix,
        n_samples,
        fit_atoms,
        data_kws={},
        gen_model_kws={},
        prior_model_kws={},
        atom_fitting_kws={},
        bond_adding_kws={},
        output_kws={},
        device='cuda',
        verbose=False,
        debug=False,
    ):
        super().__init__()
        self.device = device

    
        print('Loading data')
        self.init_data(device=device, **data_kws)
        self.tmp_n_channels_in = 3
        self.tmp_n_channels_cond = 11
        self.tmp_n_channels_out = 3
        self.batch_size = data_kws['batch_size']
        
        if self.gen_model_type:
            print('Initializing generative model')
            self.init_gen_model(
                device=device, n_samples=n_samples, **gen_model_kws
            )
            self.prior_model = None
        else:
            assert False, 'Must specify generative model type'

        print('Initializing output writer')
        self.out_writer = OutputWriter(
            out_prefix=out_prefix,
            n_samples=n_samples,
            verbose=verbose,
            **output_kws,
        )

    def init_data(self, device, **data_kws):
        #self.train_dataLoader, self.test_dataLoader = data.getDataLoader(train_file,batch_size=data_kws['batch_size'], train_ratio=data_kws['train_ratio'])
        #self.train_dataLoader, self.test_dataLoader = data.getDataLoader(train_file,batch_size=10**6, train_ratio=data_kws['train_ratio'])
        """ self.train_data = \
            data.AtomGridData(device=device, data_file=train_file, **data_kws)
        self.test_data = \
            data.AtomGridData(device=device, data_file=test_file, **data_kws) """
        if data_kws["train_data"] is not None:
            self.train_data = torch.load(data_kws["train_data"])
            self.test_data = torch.load(data_kws["test_data"])
        else:
            assert False, "pickle file is not provided"
            """ __ds = data.biDataset(train_file, device=device)
            train_size = int(len(__ds)*0.8)
            test_size = len(__ds) - train_size
            self.train_data , self.test_data= utils.data.random_split(__ds,[train_size,test_size]) """
        self.train_data = DataLoader(self.train_data,
                                     batch_size=data_kws['batch_size'],
                                     num_workers=0)
        self.test_data = DataLoader(self.test_data,
                                    batch_size=data_kws['batch_size'],
                                    num_workers=0)
        mol2_path_list = glob.glob(data_kws["target_recept_dir"]+"/*.mol2")
        profile_path_list = glob.glob(data_kws["target_recept_dir"]+"/*.profile")
        pops_path_list = glob.glob(data_kws["target_recept_dir"]+"/*.pops")
        mol2_path_list = [i for i in mol2_path_list if os.path.basename(i).split('.')[0][:4] not in [os.path.basename(j).split('.')[0] for j in profile_path_list]]
        profile_path_list = np.repeat(profile_path_list, len(mol2_path_list)).tolist()
        pops_path_list = np.repeat(pops_path_list, len(mol2_path_list)).tolist()
        self.recept_list, _ = self.train_data.dataset.dataset.prot2graphlist(mol2_path_list=mol2_path_list, profile_path_list=profile_path_list, pops_path_list=pops_path_list)

    def init_gen_model(
        self,
        device,
        caffe_init=False,
        state=None,
        **gen_model_kws
    ):
        self.gen_model = self.gen_model_type(
            n_channels_in=self.n_channels_in,
            n_channels_cond=self.n_channels_cond,
            n_channels_out=self.n_channels_out,
            device=device,
            **gen_model_kws
        )
        if caffe_init:
            self.gen_model.apply(ligan.models.caffe_init_weights)

        if state:
            print('Loading generative model state')
            state_dict = torch.load(state, map_location=device)
            state_dict.pop('log_recon_var', None)
            self.gen_model.load_state_dict(state_dict)

    @property
    def n_channels_in(self):
        if False:
            if self.gen_model_type.has_input_encoder:
                data = self.data
                if self.has_complex_input:
                    return data.n_rec_channels + data.n_lig_channels
                else:
                    return data.n_lig_channels
        else:
            return self.tmp_n_channels_in

    @property
    def n_channels_cond(self):
        if False:
            if self.gen_model_type.has_conditional_encoder:
                return self.data.n_rec_channels
        else:
            return self.tmp_n_channels_cond

    @property
    def n_channels_out(self):
        if False:
            return self.data.n_lig_channels
        else:
            return self.tmp_n_channels_out

    def forward(
        self,
        prior,
        stage2,
        interpolate=False,
        spherical=False,
        **kwargs
    ):
        print('Getting next batch of data')

        smiles_list = []
        if self.gen_model:

            with torch.no_grad():
                #TODO: impl skip_conneciton is False or True
                for protein in self.recept_list:
                    protein = protein.to(self.device)
                    for num_node in tqdm(range(3,100)):
                        in_latent_vec = torch.randn(num_node, 128).to(self.device)#,3
                        cond_latent_vec, skip_cond = self.gen_model.conditional_encoder(protein)
                        cond = cond_latent_vec.x.mean(dim=0,keepdim=True).repeat(num_node, 1)#.unsqueeze(2).expand(-1,-1,3)
                        latent_vec = torch.cat([in_latent_vec, cond], dim=1)
                        
                        adj_matrix = self.gen_model.attr_decoder(Batch.from_data_list([Data(
                            x = (latent_vec @ latent_vec.T).unsqueeze(2).expand(-1,-1,3), num_nodes = num_node,
                            device=latent_vec.device,
                            batch=torch.tensor(0,dtype=torch.int,device=latent_vec.device))]))
                        edge_index, edge_attr = convert_to_edge_attr(adj_matrix.x)
                        batch_cond = []
                        for feat in skip_cond:
                            single_cond_feat = []
                            for i in range(1):
                                __myx = F.pad(feat.x.mean(dim=0,keepdim=True).repeat(num_node, 1), (0, 0, 0, 0), "constant", 0)
                                single_cond_feat.append(Data(x=__myx, batch=torch.tensor(0, dtype=torch.int, device=latent_vec.device), num_nodes = num_node, device = feat.x.device))
                            batch_cond.append(Batch.from_data_list(single_cond_feat))
                        output = self.gen_model.decoder(Batch.from_data_list([Data(x=latent_vec, num_nodes=num_node, device=latent_vec.device, batch=torch.tensor(0, device=latent_vec.device))]), batch_cond)
                        try:
                            out_smiles = gsmiles.to_smiles(Data(output.x, edge_index, edge_attr, device=self.device), kekulize=True)
                            smiles_list.append(out_smiles)
                        except:
                            pass
        return smiles_list

    def generate(
        self,
        n_examples,
        n_samples,
        prior=False,
        stage2=False,
        var_factor=1.0,
        post_factor=1.0,
        z_score=None,
        truncate=None,
        interpolate=False,
        spherical=False,
        fit_atoms=True,
        add_bonds=True,
        uff_minimize=True,
        gnina_minimize=True,
        fit_to_real=False,
        add_to_real=False,
        minimize_real=True,
        verbose=True,
    ):
        '''
        Generate atomic density grids from generative
        model for each example in data, fit atomic
        structures, and add bonds to make molecules.
        '''
        batch_size = self.batch_size

        print('Starting to generate grids')
        for example_idx, sample_idx in itertools.product(
            range(n_examples), range(n_samples)
        ):
            # keep track of position in current batch
            full_idx = example_idx*n_samples + sample_idx
            batch_idx = full_idx % batch_size
            #print(example_idx, sample_idx, full_idx, batch_idx)

            need_real_input_mol = (sample_idx == 0)
            need_real_cond_mol = False
            need_next_batch = (batch_idx == 0)

            if need_next_batch: # forward next batch

                #if gnina_minimize: # copy to gpu
                #    self.gen_model.to('cuda')
                out_smiles = self.forward(
                    prior=prior,
                    stage2=stage2,
                    var_factor=var_factor,
                    post_factor=post_factor,
                    z_score=z_score,
                    truncate=truncate,
                    interpolate=interpolate,
                    spherical=spherical,
                )

        return self.out_writer.metrics

class CVAEGenerator(MoleculeGenerator):
    gen_model_type = ligan.models.CVAE
    has_complex_input = True


class OutputWriter(object):
    '''
    A data structure for receiving and sorting AtomGrids and
    AtomStructs from a generative model or atom fitting algorithm,
    computing metrics, and writing files to disk as necessary.
    '''
    def __init__(
        self,
        out_prefix,
        n_samples,
        output_mols=True,
        output_structs=False,
        output_grids=False,
        output_latents=False,
        output_visited=False,
        output_conv=False,
        batch_metrics=False,
        verbose=False
    ):
        out_dir, out_prefix = os.path.split(out_prefix)
        self.out_prefix = out_prefix

        self.output_structs = output_structs
        self.output_mols = output_mols
        self.output_latents = output_latents
        self.output_visited = output_visited
        self.output_conv = output_conv
        self.n_samples = n_samples
        self.batch_metrics = batch_metrics

        # accumulate metrics in dataframe
        self.metric_file = os.path.join(out_dir, f'{out_prefix}.gen_metrics')
        columns = [
            'example_idx',
            'input_rec_name',
            'input_lig_name',
            'cond_rec_name',
            'cond_lig_name',
            'sample_idx'
        ]
        self.metrics = pd.DataFrame(columns=columns).set_index(columns)

        # write a pymol script when finished
        self.pymol_file = os.path.join(out_dir, f'{out_prefix}.pymol')
        self.dx_prefixes = []
        self.sdf_files = []

        self.verbose = verbose

        # keep sdf files open so that all samples of a given
        #   struct or molecule can be written to one file
        self.open_files = dict()

        # create directories for output files
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        if output_latents:
            self.latent_dir = out_dir / 'latents'
            self.latent_dir.mkdir(exist_ok=True)

        if output_grids:
            self.grid_dir = out_dir / 'grids'
            self.grid_dir.mkdir(exist_ok=True)

        if output_structs:
            self.struct_dir = out_dir / 'structs'
            self.struct_dir.mkdir(exist_ok=True)

        if output_mols:
            self.mol_dir = out_dir / 'molecules'
            self.mol_dir.mkdir(exist_ok=True)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def close_files(self):
        '''
        Close all open files that the output
        writer currently has a reference to
        and delete the references.
        '''
        for f, out in list(self.open_files.items()):
            out.close()
            del self.open_files[f]

    def write_sdf(self, sdf_file, mol, sample_idx, is_real):
        '''
        Append molecule or atom sturct to sdf_file.

        NOTE this method assumes that samples will be
        produced in sequential order (i.e. not async)
        because it opens the file on first sample_idx
        and closes it on the last one.
        '''
        if sdf_file not in self.open_files:
            self.open_files[sdf_file] = gzip.open(sdf_file, 'wt')
        out = self.open_files[sdf_file]

        if sample_idx == 0 or not is_real:
            self.print(f'Writing {sdf_file} sample {sample_idx}')
                
            if isinstance(mol, AtomStruct):
                struct = mol
                if self.output_visited and 'visited_structs' in struct.info:
                    visited_structs = struct.info['visited_structs']
                    rd_mols = [s.to_rd_mol() for s in visited_structs]
                else:
                    rd_mols = [struct.to_rd_mol()]

            else: # molecule
                if self.output_visited and 'visited_mols' in mol.info:
                    rd_mols = mol.info['visited_mols']
                else:
                    rd_mols = [mol]

            try:
                mols.write_rd_mols_to_sdf_file(
                    out, rd_mols, str(sample_idx), kekulize=False
                )
            except ValueError:
                print(sdf_file, sample_idx, is_real)
                raise

        if sample_idx == 0:
            self.sdf_files.append(sdf_file)
        
        if sample_idx + 1 == self.n_samples or is_real:
            out.close()
            del self.open_files[sdf_file]

    def write_atom_types(self, types_file, atom_types):

        self.print('Writing ' + str(types_file))
        write_atom_types_to_file(types_file, atom_types)

    def write_dx(self, dx_prefix, grid):

        self.print(f'Writing {dx_prefix} .dx files')
        grid.to_dx(dx_prefix)
        self.dx_prefixes.append(dx_prefix)

    def write_latent(self, latent_file, latent_vec):

        self.print('Writing ' + str(latent_file))
        write_latent_vec_to_file(latent_file, latent_vec)

    def write(self, example_info, sample_idx, grid_type, grid):
        '''
        Write output files for grid and compute metrics in
        data frame, if all necessary data is present.
        '''
        out_prefix = self.out_prefix
        example_idx, input_rec_name, input_lig_name, cond_rec_name, cond_lig_name = example_info
        grid_prefix = f'{out_prefix}_{example_idx}_{grid_type}'
        i = str(sample_idx)

        assert grid_type in {
            'rec', 'lig', 'cond_rec', 'cond_lig',
            'lig_gen', 'lig_fit', 'lig_gen_fit'
        }
        is_lig_grid = grid_type.startswith('lig')
        is_gen_grid = grid_type.endswith('_gen')
        is_fit_grid = grid_type.endswith('_fit')

        is_real_grid = not (is_gen_grid or is_fit_grid)
        is_first_real_grid = (is_real_grid and sample_idx == 0)
        has_struct = (is_real_grid or is_fit_grid)
        has_conv_grid = not is_fit_grid # and is_lig_grid ?

        # write atomic structs and/or molecules
        if has_struct:

            # get struct that created this grid (via molgrid.GridMaker)
            #   note that depending on the grid_type, this can either be
            #   from atom fitting OR from typing a real molecule
            struct = grid.info['src_struct']

            # the real (source) molecule and atom types don't change
            #   between different samples, so only write them once

            # and we don't apply bond adding to the receptor struct,
            #   so only ligand structs have add_mol and uff_mol

            # write real molecule
            if self.output_mols and is_first_real_grid:

                sdf_file = self.mol_dir / (grid_prefix + '_src.sdf.gz')
                src_mol = struct.info['src_mol']
                self.write_sdf(sdf_file, src_mol, sample_idx, is_real=True)

                if 'pkt_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_pkt.sdf.gz')
                    pkt_mol = src_mol.info['pkt_mol']
                    self.write_sdf(sdf_file, pkt_mol, sample_idx, is_real=True)

                if 'uff_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_uff.sdf.gz')
                    uff_mol = src_mol.info['uff_mol']
                    self.write_sdf(sdf_file, uff_mol, sample_idx, is_real=True)

                if 'gni_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_gna.sdf.gz')
                    gni_mol = src_mol.info['gni_mol']
                    self.write_sdf(sdf_file, gni_mol, sample_idx, is_real=True)

            # write typed atomic structure (real or fit)
            if self.output_structs and (is_first_real_grid or is_fit_grid):

                sdf_file = self.struct_dir / (grid_prefix + '.sdf.gz')
                self.write_sdf(sdf_file, struct, sample_idx, is_real_grid)

                # write atom type channels
                types_base = grid_prefix + '_' + i + '.atom_types'
                types_file = self.struct_dir / types_base
                self.write_atom_types(types_file, struct.atom_types)

            # write bond-added molecule (real or fit ligand)
            if self.output_mols and 'add_mol' in struct.info:
                sdf_file = self.mol_dir / (grid_prefix + '_add.sdf.gz')
                add_mol = struct.info['add_mol']
                self.write_sdf(sdf_file, add_mol, sample_idx, is_real_grid)

                if 'pkt_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_pkt.sdf.gz')
                    pkt_mol = add_mol.info['pkt_mol']
                    self.write_sdf(sdf_file, pkt_mol, sample_idx, is_real_grid)

                if 'uff_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix + '_add_uff.sdf.gz')
                    uff_mol = add_mol.info['uff_mol']
                    self.write_sdf(sdf_file, uff_mol, sample_idx, is_real_grid)

                if 'gni_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_gna.sdf.gz')
                    gni_mol = add_mol.info['gni_mol']
                    self.write_sdf(sdf_file, gni_mol, sample_idx, is_real_grid)

                if 'cond_pkt_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_cond_pkt.sdf.gz')
                    pkt_mol = add_mol.info['pkt_mol']
                    self.write_sdf(sdf_file, pkt_mol, sample_idx, is_real_grid)

                if 'cond_uff_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix + '_add_cond_uff.sdf.gz')
                    uff_mol = add_mol.info['uff_mol']
                    self.write_sdf(sdf_file, uff_mol, sample_idx, is_real_grid)

                if 'cond_gni_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_cond_gna.sdf.gz')
                    gni_mol = add_mol.info['gni_mol']
                    self.write_sdf(sdf_file, gni_mol, sample_idx, is_real_grid)

        # write latent vectors
        if self.output_latents and is_gen_grid:

            latent_file = self.latent_dir / (grid_prefix + '_' + i + '.latent')
            self.write_latent(latent_file, grid.info['src_latent'])

        
    def compute_metrics(self, example_info, sample_idx=None):
        '''
        Compute metrics for density grids, typed atomic structures,
        and molecules for a given ligand in metrics data frame.
        '''
        has_rec = ('rec' in self.grid_types)
        has_cond_rec = ('cond_rec' in self.grid_types)
        has_lig_gen = ('lig_gen' in self.grid_types)
        has_lig_fit = ('lig_fit' in self.grid_types)
        has_lig_gen_fit = ('lig_gen_fit' in self.grid_types)

        if sample_idx is None:
            sample_idxs = range(self.n_samples)
        else:
            sample_idxs = [sample_idx]

        # TODO don't compute metrics twice w/ diff_cond_transform
        #   the only thing we really need is the lig l2 loss
        
        lig_grids = self.grids[example_info]

        if self.batch_metrics: # compute mean grids and type counts

            def get_mean_type_counts(struct_batch):
                n = len(struct_batch)
                type_counts = sum([s.type_counts for s in struct_batch]) / n
                elem_counts = sum([s.elem_counts for s in struct_batch]) / n
                prop_counts = sum([s.prop_counts for s in struct_batch]) / n
                return type_counts, elem_counts, prop_counts

            lig_grid_batch = [lig_grids[i]['lig'].values for i in sample_idxs]
            lig_grid_mean = sum(lig_grid_batch) / self.n_samples

            lig_struct_batch = [lig_grids[i]['lig'].info['src_struct'] for i in sample_idxs]
            lig_mean_counts = get_mean_type_counts(lig_struct_batch)

            if has_cond_rec:
                cond_lig_grid_batch = \
                    [lig_grids[i]['cond_lig'].values for i in sample_idxs]
                cond_lig_grid_mean = sum(cond_lig_grid_batch) / self.n_samples

                cond_lig_struct_batch = [
                    lig_grids[i]['cond_lig'].info['src_struct'] \
                        for i in sample_idxs
                ]
                cond_lig_mean_counts = \
                    get_mean_type_counts(cond_lig_struct_batch)

            if has_lig_fit:
                lig_fit_struct_batch = [lig_grids[i]['lig_fit'].info['src_struct'] for i in sample_idxs]
                lig_fit_mean_counts = get_mean_type_counts(lig_fit_struct_batch)

            if has_lig_gen:
                lig_gen_grid_mean = sum(lig_grids[i]['lig_gen'].values for i in sample_idxs) / self.n_samples
                lig_latent_mean = sum(lig_grids[i]['lig_gen'].info['src_latent'] for i in sample_idxs) / self.n_samples

                if has_lig_gen_fit:
                    lig_gen_fit_struct_batch = [lig_grids[i]['lig_gen_fit'].info['src_struct'] for i in sample_idxs]
                    lig_gen_fit_mean_counts = get_mean_type_counts(lig_gen_fit_struct_batch)
        else:
            lig_grid_mean = None
            cond_lig_grid_mean = None
            lig_mean_counts = None
            cond_lig_mean_counts = None
            lig_fit_mean_counts = None
            lig_gen_grid_mean = None
            lig_latent_mean = None
            lig_gen_fit_mean_counts = None

        for sample_idx in sample_idxs:
            idx = example_info + (sample_idx,)

            rec_grid = lig_grids[sample_idx]['rec'] if has_rec else None
            lig_grid = lig_grids[sample_idx]['lig']
            self.compute_grid_metrics(idx,
                grid_type='lig',
                grid=lig_grid,
                mean_grid=lig_grid_mean,
                cond_grid=rec_grid,
            )

            lig_struct = lig_grid.info['src_struct']
            self.compute_struct_metrics(idx,
                struct_type='lig',
                struct=lig_struct,
                mean_counts=lig_mean_counts,
            )

            lig_mol = lig_struct.info['src_mol']
            self.compute_mol_metrics(idx,
                mol_type='lig', mol=lig_mol
            )

            if 'add_mol' in lig_struct.info:

                lig_add_mol = lig_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_add', mol=lig_add_mol, ref_mol=lig_mol
                )

            if has_cond_rec:
                cond_rec_grid = lig_grids[sample_idx]['cond_rec']
                cond_lig_grid = lig_grids[sample_idx]['cond_lig']
                self.compute_grid_metrics(idx,
                    grid_type='cond_lig',
                    grid=cond_lig_grid,
                    mean_grid=cond_lig_grid_mean,
                    cond_grid=cond_rec_grid,
                )

                cond_lig_struct = cond_lig_grid.info['src_struct']
                self.compute_struct_metrics(idx,
                    struct_type='cond_lig',
                    struct=cond_lig_struct,
                    mean_counts=cond_lig_mean_counts,
                )

                cond_lig_mol = cond_lig_struct.info['src_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig', mol=cond_lig_mol
                )

            if has_lig_gen:

                lig_gen_grid = lig_grids[sample_idx]['lig_gen']
                self.compute_grid_metrics(idx,
                    grid_type='lig_gen',
                    grid=lig_gen_grid,
                    ref_grid=lig_grid,
                    mean_grid=lig_gen_grid_mean,
                    cond_grid=rec_grid
                )

                lig_latent = lig_gen_grid.info['src_latent']
                self.compute_latent_metrics(idx,
                    latent_type='lig',
                    latent=lig_latent,
                    mean_latent=lig_latent_mean
                )

                if has_cond_rec:

                    self.compute_grid_metrics(idx,
                        grid_type='lig_gen_cond',
                        grid=lig_gen_grid,
                        ref_grid=cond_lig_grid,
                        cond_grid=cond_rec_grid,
                        ref_only=True
                    )

            if has_lig_fit:

                lig_fit_grid = lig_grids[sample_idx]['lig_fit']
                self.compute_grid_metrics(idx,
                    grid_type='lig_fit',
                    grid=lig_fit_grid,
                    ref_grid=lig_grid,
                    cond_grid=rec_grid,
                )

                lig_fit_struct = lig_fit_grid.info['src_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_fit',
                    struct=lig_fit_struct,
                    ref_struct=lig_struct,
                    mean_counts=lig_fit_mean_counts,
                )

                lig_fit_add_mol = lig_fit_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_fit_add',
                    mol=lig_fit_add_mol,
                    ref_mol=lig_mol,
                )

                lig_fit_add_struct = lig_fit_add_mol.info['type_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_fit_add',
                    struct=lig_fit_add_struct,
                    ref_struct=lig_fit_struct,
                )

            if has_lig_gen_fit:

                lig_gen_fit_grid = lig_grids[sample_idx]['lig_gen_fit']
                self.compute_grid_metrics(idx,
                    grid_type='lig_gen_fit',
                    grid=lig_gen_fit_grid,
                    ref_grid=lig_gen_grid,
                    cond_grid=rec_grid,
                )

                lig_gen_fit_struct = lig_gen_fit_grid.info['src_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_gen_fit',
                    struct=lig_gen_fit_struct,
                    ref_struct=lig_struct,
                    mean_counts=lig_gen_fit_mean_counts,
                )

                lig_gen_fit_add_mol = lig_gen_fit_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_gen_fit_add',
                    mol=lig_gen_fit_add_mol,
                    ref_mol=lig_mol,
                )

                lig_gen_fit_add_struct = lig_gen_fit_add_mol.info['type_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_gen_fit_add',
                    struct=lig_gen_fit_add_struct,
                    ref_struct=lig_gen_fit_struct,
                )

                if has_cond_rec:

                    self.compute_struct_metrics(idx,
                        struct_type='lig_gen_fit_cond',
                        struct=lig_gen_fit_struct,
                        ref_struct=cond_lig_struct,
                        ref_only=True,
                    )

                    self.compute_mol_metrics(idx,
                        mol_type='lig_gen_fit_add_cond',
                        mol=lig_gen_fit_add_mol,
                        ref_mol=cond_lig_mol,
                        ref_only=True,
                        use_cond_min=True
                    )

        self.print(self.metrics.loc[example_info].loc[sample_idxs].transpose())

    def compute_grid_metrics(
        self,
        idx,
        grid_type,
        grid,
        ref_grid=None,
        mean_grid=None,
        cond_grid=None,
        ref_only=False,
    ):
        m = self.metrics

        if not ref_only:

            # density magnitude
            m.loc[idx, grid_type+'_grid_norm'] = grid.values.norm().item()
            m.loc[idx, grid_type+'_grid_elem_norm'] = \
                grid.elem_values.norm().item()
            m.loc[idx, grid_type+'_grid_prop_norm'] = \
                grid.prop_values.norm().item()

            if mean_grid is not None:

                # density variance
                # (divide by n_samples (+1) for sample (population) variance)
                m.loc[idx, grid_type+'_grid_variance'] = (
                    (grid.values - mean_grid)**2
                ).sum().item()

        if ref_grid is not None:

            # density L2 loss
            m.loc[idx, grid_type+'_L2_loss'] = (
                (ref_grid.values - grid.values)**2
            ).sum().item() / 2

            m.loc[idx, grid_type+'_elem_L2_loss'] = (
                (ref_grid.elem_values - grid.elem_values)**2
            ).sum().item() / 2

            m.loc[idx, grid_type+'_prop_L2_loss'] = (
                (ref_grid.prop_values - grid.prop_values)**2
            ).sum().item() / 2

            # shape similarity
            ref_shape = (ref_grid.values.sum(dim=0) > 0)
            shape = (grid.values.sum(dim=0) > 0)
            m.loc[idx, grid_type+'_shape_sim'] = (
                (ref_shape & shape).sum() / (ref_shape | shape).sum()
            ).item()

        if cond_grid is not None:

            # density product
            m.loc[idx, grid_type+'_rec_prod'] = (
                cond_grid.values.sum(dim=0) * 
                grid.values.sum(dim=0).clamp(0)
            ).sum().item()

            m.loc[idx, grid_type+'_rec_elem_prod'] = (
                cond_grid.elem_values.sum(dim=0) * 
                grid.elem_values.sum(dim=0).clamp(0)
            ).sum().item()

            m.loc[idx, grid_type+'_rec_prop_prod'] = (
                cond_grid.prop_values.sum(dim=0) * 
                grid.prop_values.sum(dim=0).clamp(0)
            ).sum().item()


    def compute_latent_metrics(
        self, idx, latent_type, latent, mean_latent=None
    ):
        m = self.metrics

        # latent vector magnitude
        m.loc[idx, latent_type+'_latent_norm'] = latent.norm().item()

        if mean_latent is not None:

            # latent vector variance
            variance = (
                (latent - mean_latent)**2
            ).sum().item()
        else:
            variance = np.nan

        m.loc[idx, latent_type+'_latent_variance'] = variance

    def compute_struct_metrics(
        self,
        idx,
        struct_type,
        struct,
        ref_struct=None,
        mean_counts=None,
        ref_only=False,
    ):
        m = self.metrics

        if not ref_only:

            m.loc[idx, struct_type+'_n_atoms'] = struct.n_atoms
            m.loc[idx, struct_type+'_radius'] = (
                struct.radius if struct.n_atoms > 0 else np.nan
            )

            if mean_counts is not None:

                mean_type_counts, mean_elem_counts, mean_prop_counts = \
                    mean_counts

                m.loc[idx, struct_type+'_type_variance'] = (
                    (struct.type_counts - mean_type_counts)**2
                ).sum().item()

                m.loc[idx, struct_type+'_elem_variance'] = (
                    (struct.elem_counts - mean_elem_counts)**2
                ).sum().item()

                m.loc[idx, struct_type+'_prop_variance'] = (
                    (struct.prop_counts - mean_prop_counts)**2
                ).sum().item()

        if ref_struct is not None:

            # difference in num atoms
            m.loc[idx, struct_type+'_n_atoms_diff'] = (
                ref_struct.n_atoms - struct.n_atoms
            )

            # overall type count difference
            m.loc[idx, struct_type+'_type_diff'] = (
                ref_struct.type_counts - struct.type_counts
            ).norm(p=1).item()

            # element type count difference
            m.loc[idx, struct_type+'_elem_diff'] = (
                ref_struct.elem_counts - struct.elem_counts
            ).norm(p=1).item()

            # property type count difference
            m.loc[idx, struct_type+'_prop_diff'] = (
                ref_struct.prop_counts - struct.prop_counts
            ).norm(p=1).item()

            # minimum atom-only RMSD (ignores properties)
            rmsd = ligan.metrics.compute_struct_rmsd(ref_struct, struct)
            m.loc[idx, struct_type+'_RMSD'] = rmsd

        if not ref_only:
            if struct_type.endswith('_fit'):

                # fit time and number of visited structures
                m.loc[idx, struct_type+'_time'] = struct.info['time']
                m.loc[idx, struct_type+'_n_visited'] = len(
                    struct.info['visited_structs']
                )

                # accuracy of estimated type counts, whether or not
                # they were actually used to constrain atom fitting
                est_type = struct_type[:-4] + '_est'
                m.loc[idx, est_type+'_type_diff'] = struct.info.get(
                    'est_type_diff', np.nan
                )
                m.loc[idx, est_type+'_exact_types'] = (
                    m.loc[idx, est_type+'_type_diff'] == 0
                )

    def compute_mol_metrics(
        self,
        idx,
        mol_type,
        mol,
        ref_mol=None,
        ref_only=False,
        use_cond_min=False,
    ):
        m = self.metrics

        # check molecular validity
        valid, reason = mol.validate()

        if not ref_only:
            m.loc[idx, mol_type+'_n_atoms'] = mol.n_atoms
            m.loc[idx, mol_type+'_n_frags'] = mol.n_frags
            m.loc[idx, mol_type+'_valid'] = valid
            m.loc[idx, mol_type+'_reason'] = reason

            # other molecular descriptors
            m.loc[idx, mol_type+'_MW'] = mols.get_rd_mol_weight(mol)
            m.loc[idx, mol_type+'_logP'] = mols.get_rd_mol_logP(mol)
            m.loc[idx, mol_type+'_QED'] = mols.get_rd_mol_QED(mol)
            if valid:
                m.loc[idx, mol_type+'_SAS'] = mols.get_rd_mol_SAS(mol)
                m.loc[idx, mol_type+'_NPS'] = mols.get_rd_mol_NPS(mol)
            else:
                m.loc[idx, mol_type+'_SAS'] = np.nan
                m.loc[idx, mol_type+'_NPS'] = np.nan

        # convert to SMILES string
        smi = mol.to_smi()
        if not ref_only:
            m.loc[idx, mol_type+'_SMILES'] = smi

        if ref_mol: # compare to ref_mol

            # difference in num atoms
            m.loc[idx, mol_type+'_n_atoms_diff'] = (
                ref_mol.n_atoms - mol.n_atoms
            )

            ref_valid, ref_reason = ref_mol.validate()

            # get reference SMILES strings
            ref_smi = ref_mol.to_smi()
            m.loc[idx, mol_type+'_SMILES_match'] = (smi == ref_smi)

            if valid and ref_valid: # fingerprint similarity

                m.loc[idx, mol_type+'_ob_sim'] = \
                    mols.get_ob_smi_similarity(ref_smi, smi)
                m.loc[idx, mol_type+'_rdkit_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'rdkit')
                m.loc[idx, mol_type+'_morgan_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'morgan')
                m.loc[idx, mol_type+'_maccs_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'maccs')
            else:
                m.loc[idx, mol_type+'_ob_sim'] = np.nan
                m.loc[idx, mol_type+'_rdkit_sim'] = np.nan
                m.loc[idx, mol_type+'_morgan_sim'] = np.nan
                m.loc[idx, mol_type+'_maccs_sim'] = np.nan

        if 'uff_mol' not in mol.info:
            return

        # UFF energy minimization
        if use_cond_min and 'cond_uff_mol' in mol.info: # handle diff_cond_transform with no diff_cond_structs
            uff_mol = mol.info['cond_uff_mol']
        else:
            uff_mol = mol.info['uff_mol']
        uff_init = uff_mol.info['E_init']
        uff_min = uff_mol.info['E_min']
        uff_rmsd = uff_mol.info['min_rmsd']

        m.loc[idx, mol_type+'_UFF_init'] = uff_init
        m.loc[idx, mol_type+'_UFF_min'] = uff_min
        m.loc[idx, mol_type+'_UFF_rmsd'] = uff_rmsd
        m.loc[idx, mol_type+'_UFF_error'] = uff_mol.info['min_error']
        m.loc[idx, mol_type+'_UFF_time'] = uff_mol.info['min_time']

        # compare energy to ref mol, before and after minimizing
        """ if ref_mol:
            ref_uff_mol = ref_mol.info['uff_mol']
            ref_uff_init = ref_uff_mol.info['E_init']
            ref_uff_min = ref_uff_mol.info['E_min']
            ref_uff_rmsd = ref_uff_mol.info['min_rmsd']
            m.loc[idx, mol_type+'_UFF_init_diff'] = uff_init - ref_uff_init
            m.loc[idx, mol_type+'_UFF_min_diff'] = uff_min - ref_uff_min
            m.loc[idx, mol_type+'_UFF_rmsd_diff'] = uff_rmsd - ref_uff_rmsd

        if 'gni_mol' not in mol.info:
            return

        # gnina energy minimization
        if use_cond_min and 'cond_gni_mol' in mol.info:
            gni_mol = mol.info['cond_gni_mol']
        else:
            gni_mol = mol.info['gni_mol']
        vina_aff = gni_mol.info.get('minimizedAffinity', np.nan)
        vina_rmsd = gni_mol.info.get('minimizedRMSD', np.nan)
        cnn_pose = gni_mol.info.get('CNNscore', np.nan)
        cnn_aff = gni_mol.info.get('CNNaffinity', np.nan)

        m.loc[idx, mol_type+'_vina_aff'] = vina_aff
        m.loc[idx, mol_type+'_vina_rmsd'] = vina_rmsd
        m.loc[idx, mol_type+'_cnn_pose'] = cnn_pose
        m.loc[idx, mol_type+'_cnn_aff'] = cnn_aff
        m.loc[idx, mol_type+'_gnina_error'] = gni_mol.info['error']

        # compare gnina metrics to ref mol
        if ref_mol:
            ref_gni_mol = ref_mol.info['gni_mol']
            try:
                ref_vina_aff = ref_gni_mol.info['minimizedAffinity']
            except KeyError:
                print(ref_gni_mol.info)
                raise
            ref_vina_rmsd = ref_gni_mol.info['minimizedRMSD']
            ref_cnn_pose = ref_gni_mol.info['CNNscore']
            ref_cnn_aff = ref_gni_mol.info['CNNaffinity']

            m.loc[idx, mol_type+'_vina_aff_diff'] = vina_aff - ref_vina_aff
            m.loc[idx, mol_type+'_vina_rmsd_diff'] = vina_rmsd - ref_vina_rmsd
            m.loc[idx, mol_type+'_cnn_pose_diff'] = cnn_pose - ref_cnn_pose
            m.loc[idx, mol_type+'_cnn_aff_diff'] = cnn_aff - ref_cnn_aff """


def read_rec_from_pdb_file(pdb_file):

    rec_mol = mols.Molecule.from_pdb(pdb_file, sanitize=False)
    try:
        Chem.SanitizeMol(rec_mol)
    except Chem.MolSanitizeException:
        pass
    return rec_mol


def read_lig_from_sdf_file(sdf_file, use_ob=True):
    '''
    Try to find the real molecule in data_root using the
    source path in the data file, without file extension.
    '''
    if use_ob: # read and add Hs with OpenBabel, then convert to RDkit
        lig_mol = mols.read_ob_mols_from_file(sdf_file, 'sdf')[0]
        lig_mol.AddHydrogens()
        lig_mol = mols.Molecule.from_ob_mol(lig_mol)

    else: # read and add Hs with RDKit (need to sanitize before add Hs)
        lig_mol = mols.Molecule.from_sdf(sdf_file, sanitize=False, idx=0)

    try: # need to do this to get ring info, etc.
        lig_mol.sanitize()
    except Chem.MolSanitizeException:
        pass

    if not use_ob: # add Hs with rdkit (after sanitize)
        lig_mol = lig_mol.add_hs()

    return lig_mol


def write_atom_types_to_file(types_file, atom_types):
    with open(types_file, 'w') as f:
        f.write('\n'.join(str(a) for a in atom_types))


def write_latent_vec_to_file(latent_file, latent_vec):

    with open(latent_file, 'w') as f:
        for value in latent_vec:
            f.write(str(value.item()) + '\n')


def write_pymol_script(
    pymol_file, out_prefix, dx_prefixes, sdf_files
):
    '''
    Write a pymol script that loads all .dx files with a given
    prefix into a single group, then loads a set of sdf_files
    and translates them to the origin, if centers are provided.
    '''
    with open(pymol_file, 'w') as f:

        for dx_prefix in dx_prefixes: # load density grids
            try:
                m = re.match(
                    '^grids/({}_.*)$'.format(re.escape(out_prefix)),
                    str(dx_prefix)
                )
            except AttributeError:
                print(dx_prefix, file=sys.stderr)
                raise
            group_name = m.group(1) + '_grids'
            dx_pattern = '{}_*.dx'.format(dx_prefix)
            f.write('load_group {}, {}\n'.format(dx_pattern, group_name))

        for sdf_file in sdf_files: # load structs/molecules
            try:
                m = re.match(
                    r'^.*(molecules|structs)/({}_.*)\.sdf(\.gz)?$'.format(
                        re.escape(out_prefix)
                    ),
                    str(sdf_file)
                )
                obj_name = m.group(2)
            except AttributeError:
                print(sdf_file, file=sys.stderr)
                raise
            f.write('load {}, {}\n'.format(sdf_file, obj_name))

        f.write('util.cbam *rec_src\n')
        f.write('util.cbag *lig_src\n')
        f.write('util.cbac *lig_gen_fit_add\n')
