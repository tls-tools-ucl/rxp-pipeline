import numpy as np
import sys, os
from tqdm import tqdm

with open('scan_positions.csv', 'w') as fh:

    for dat in tqdm(sys.argv[1:]):
        mat = np.loadtxt(dat)
        fh.write('{} {} {} {}\n'.format(os.path.split(dat)[1][:-4], mat[0, 3], mat[1, 3], mat[2, 3]))
