import os
import sys
import glob
from pathlib import Path
import tables

def split_samples(root_folder="./airsim"):

        
        cells = []
        for item in Path(root_folder).glob("*/*/*.h5"):
            if ('output.h5' in str(item)) and ("2021-02-03T035302" not in str(item)) and ("2021-02-04T104447" not in str(item)):
                cells.append(item)

        cells = sorted(cells)

        nblocks = 10
        sequence = []
        i = 0
        for cell in cells:
            print(cell)
            f = tables.open_file(cell, "r")
            labels = f.get_node("/labels")[:]
            X_tmp = f.get_node("/videos")[:].squeeze()
    
            f.close()
            cell_path, _ = os.path.split(cell)
            split_path = os.path.join(cell_path,"split")
            if not os.path.exists(split_path):
                os.mkdir(split_path)

            for j in range(labels.shape[0]):
                x = X_tmp[j,:,:,:,:].squeeze()
                tensor_path = os.path.join(split_path,f'{j}.h5')
                f = tables.open_file(tensor_path, 'w')
                atom1 = tables.Atom.from_dtype(x.dtype)
                ds1 = f.create_carray(f.root, 'videos', atom1, x.shape)
                ds1[:] = x
                f.close()
            
            del X_tmp
                
                

if __name__ == "__main__":
    
    split_samples(root_folder="/network/tmp1/bakhtias/airsim")