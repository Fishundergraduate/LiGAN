from glob import glob
from argparse import ArgumentParser
import os
import pandas as pd
from multiprocessing import Pool
import multiprocessing

def get_prof_pops_are_exist(path_to_pocket:str)->bool:
    return os.path.exists(path_to_pocket+"/../../protein-data-part1/"+path_to_pocket.split("/")[-1][:-2]) or os.path.exists(path_to_pocket+"/../../protein-data-part2/"+path_to_pocket.split("/")[-1][:-2])
parser = ArgumentParser()

#parser.add_argument("dir","path to pocket-data")
#args = parser.parse_args()
#path_to_pocket = args.dir
path_to_pocket = "/mnt/d/Documents_2023/pocket-data"

paths = glob(f"{path_to_pocket}/*")

df = pd.DataFrame(paths,columns=["path_pocket"])

with Pool(multiprocessing.cpu_count()) as pool:
    df["path_protein"] = pool.map(get_prof_pops_are_exist, df["path_pocket"])

os.getcwd()