import os
import sys
import glob
import multiprocessing
import json
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np

import ply_io
import pdal


def tile_index(ply, args):

    if args.verbose:
        with args.Lock:
            print(f'processing: {ply}')

    reader = {"type":f"readers{os.path.splitext(ply)[1]}",
              "filename":ply}
    stats =  {"type":"filters.stats",
              "dimensions":"X,Y"}
    JSON = json.dumps([reader, stats])
    pipeline = pdal.Pipeline(JSON)
    pipeline.execute()
    JSON = pipeline.metadata
    X = JSON['metadata']['filters.stats']['statistic'][0]['average']
    Y = JSON['metadata']['filters.stats']['statistic'][1]['average']
    T = int(os.path.split(ply)[1].split('.')[0])
    
    with args.Lock:
        
        with open(args.tile_index, 'a') as fh:
            fh.write(f'{T} {X} {Y}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--pc', type=str, nargs='*', required=True, help='input tiles')
    parser.add_argument('-t','--tile-index', default='tile_index.dat', help='tile index file')
    parser.add_argument('--num-prcs', type=int, default=10, help='number of cores to use')
    parser.add_argument('--verbose', action='store_true', help='print something')
    args = parser.parse_args()

    m = multiprocessing.Manager()
    args.Lock = m.Lock()
    pool = multiprocessing.Pool(args.num_prcs)
    pool.starmap_async(tile_index, [(ply, args) for ply in args.pc])
    pool.close()
    pool.join() 
