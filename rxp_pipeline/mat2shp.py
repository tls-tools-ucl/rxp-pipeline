import os
import glob
import argparse
import numpy as np
import geopandas as gp
from shapely.geometry import Point

parser = argparse.ArgumentParser()
parser.add_argument('-m','--mdir', required=True, type=str, help='directory where matrix files are stored')
parser.add_argument('-o','--oshp', required=True, type=str, help='output shapefile')
parser.add_argument('--global-matrix', required=False, type=str, default=False, help='global rotation matrix')
parser.add_argument('--verbose', action='store_true', help='print something')
args = parser.parse_args()

df = gp.GeoDataFrame(columns=['x', 'y', 'z'])

for i, dat in enumerate(glob.glob(os.path.join(args.mdir, '*.dat')) +
                        glob.glob(os.path.join(args.mdir, '*.DAT'))):
    df.loc[i, :] = np.loadtxt(dat)[:3, 3]

if args.global_matrix:
    df.loc[:, 'a'] = 1
    df[['x', 'y', 'z']] = np.dot(np.loadtxt(args.global_matrix), df[['x', 'y', 'z', 'a']].values.T).T[:, :3]

geometry = [Point(r.x, r.y) for r in df.itertuples()]
df.geometry = geometry

X = df.unary_union.minimum_rotated_rectangle
gp.GeoDataFrame(data=['A'], geometry=[X], columns=['A']).to_file(args.oshp)

if args.verbose: print(f'shapefile saved to: {args.oshp}')

