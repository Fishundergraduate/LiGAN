## This file is used to make affinity csv file from pdbbind data for using train data as pKi value.
import pandas as pd


from torchdata import datapipes
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
import tqdm

import numpy as np
import os
import glob 

def makePocketDatapipe(datarootDir:str):
    __pock_pipe = datapipes.iter.FileLister(datarootDir+"/pocket-data",recursive=True)
    __pock_pipe = __pock_pipe.filter(filter_fn=lambda x: x.endswith("mol2"))
    __pock_pipe = __pock_pipe.map(lambda x:{
        'pockID' : x.split("/")[-1][:5],#Key index
        'pockPath': x
    })
    __lig__pipe = datapipes.iter.FileLister(datarootDir+"/pocket-data",recursive=True)
    __lig__pipe = __lig__pipe.filter(filter_fn=lambda x: x.endswith("sdf"))
    __lig__pipe = __lig__pipe.map(lambda x:{
        'pockID' : x.split("/")[-1][:4],#Key index
        'ligPath': x
    })
    pipe = __pock_pipe.zip_with_iter(
    __lig__pipe,
    key_fn=lambda x: x['pockID'],
    ref_key_fn=lambda x:x['pockID'],
    merge_fn=lambda x,y:(x,y),keep_key=True
    )
    return DataLoader(pipe,batch_size=1,shuffle=False,num_workers=0)

from openbabel import openbabel, pybel

def convert_to_3d_pdbqt(input_file, output_file):
    # Create a molecule object from the input file
    mol = next(pybel.readfile(input_file.split('.')[-1], input_file))

    # Add hydrogens to the molecule
    mol.OBMol.AddHydrogens()

    # Convert the molecule to 3D
    mol.make3D()

    # Convert the molecule to pdbqt format
    pdbqt = mol.write('pdbqt')

    # Write the pdbqt data to the output file
    with open(output_file, 'w') as f:
        f.write(pdbqt)

import glob
from tqdm import tqdm

def calculate_box_size(pdb_file, mol2_file):
    # Load the PDB file
    ppdb = PandasPdb().read_pdb(pdb_file)
    # Load the LPC file
    plpc = PandasMol2().read_mol2(mol2_file)
    # Extract the protein and ligand
    protein = ppdb.df['ATOM']
    ligand = plpc.df
    # Calculate the min and max coordinates for protein and ligand
    min_coords_protein = protein[['x_coord', 'y_coord', 'z_coord']].min().values
    max_coords_protein = protein[['x_coord', 'y_coord', 'z_coord']].max().values
    min_coords_ligand = ligand[['x', 'y', 'z']].min().values
    max_coords_ligand = ligand[['x', 'y', 'z']].max().values
    # Calculate the overall min and max coordinates
    min_coords = np.minimum(min_coords_protein, min_coords_ligand)
    max_coords = np.maximum(max_coords_protein, max_coords_ligand)
    # Calculate the box size
    box_size = max_coords - min_coords
    return box_size


def generate_vina_gpu_config(no_ext_file, output_dir):
    # Load the PDB file
    pmol2 = PandasMol2().read_mol2(no_ext_file+".mol2")

    if pmol2.df is None:
        raise ValueError(f"Failed to read {no_ext_file}.mol2")

    # Extract the ligand
    ligand = pmol2.df[['x', 'y', 'z']].values

    # Calculate the center of the docking box
    center = ligand.mean(axis=0)
    box_size = calculate_box_size(no_ext_file+".pdb", no_ext_file+".mol2")
    # Create the Vina-GPU config file
    vina_gpu_config = f"""receptor = {output_dir}/receptor.pdbqt
ligand = {output_dir}/ligand.pdbqt

center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}

size_x = {box_size[0]}
size_y = {box_size[1]}
size_z = {box_size[2]}

exhaustiveness = 8
num_modes = 20
energy_range = 3
thread = 8192"""

    # Write the Vina-GPU config file
    with open(os.path.join(output_dir, 'vina_gpu_config.txt'), 'w') as f:
        f.write(vina_gpu_config)
    # Convert the pdb and sdf files to 3D pdbqt format
    convert_to_3d_pdbqt(no_ext_file+".sdf", output_dir+'/ligand.pdbqt')
    full_body = glob.glob("/".join(no_ext_file.split("/")[:-3])+f"/protein-data*/{no_ext_file.split('/')[-1][:-2]}/{no_ext_file.split('/')[-1][:-2]}.pdb")[0]
    convert_to_3d_pdbqt(full_body, output_dir+f'/receptor.pdbqt')

if __name__=='__main__':
    #lder = makePocketDatapipe("/mnt/d/Documents_2023/data")
    lder = glob.glob("/mnt/d/Documents_2023/data/pocket-data/*")
    df = pd.DataFrame(lder,columns=['path'])
    df["dock_score"] = 0.0
    for k,e in tqdm(enumerate(lder)):
        #print(k,e)
        if k in [0,1,2,3,4,5,6,7,8,9]:
            continue
        no_ext = e+"/"+e.split("/")[-1]
        generate_vina_gpu_config(no_ext,"./data/workspace")
        break