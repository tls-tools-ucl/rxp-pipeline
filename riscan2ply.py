from datetime import datetime
start = datetime.now()

import sys
import os
import glob
import multiprocessing
import json
import argparse

import pandas as pd
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import pyproj

from lxml import etree as et

import ply_io
import pdal

def tile_data(scan_pos, args):


    try:
        if args.verbose:
            with args.Lock:
                print('rdbx -> xyz:', scan_pos['rdbx'])

        sp = int(scan_pos['name'].replace('ScanPos', ''))    
        cmds = []
 
        # pdal commands as dictionaries
        read_in = {"type":"readers.rdb",
                   "filename": scan_pos['rdbx']}
        cmds.append(read_in)
    
        dev_filter = {"type":"filters.range", 
                      "limits":"Deviation[0:{}]".format(args.deviation)}
        cmds.append(dev_filter)    

        refl_filter = {"type":"filters.range", 
                      "limits":"Reflectance[{}:{}]".format(*args.reflectance)}
        cmds.append(refl_filter)
    
        if args.preserve_projection:
            pass
        elif args.target_epsg:
            transform = {"type":"filters.reprojection",
                         "in_srs":f"EPSG:{args.pop_epsg}",
                         "out_srs":f"EPSG:{args.target_epsg}"}
            cmds.append(transform)
        else: 
            transform = {"type":"filters.transformation",
                         "matrix":' '.join(np.linalg.inv(args.pop).flatten().astype(str))}
            cmds.append(transform)

        tile = {"type":"filters.splitter",
                "length":f"{args.tile}",
                "origin_x":"0",
                "origin_y":"0"}
        cmds.append(tile)

        # link commmands and pass to pdal
        JSON = json.dumps(cmds)

        pipeline = pdal.Pipeline(JSON)
        pipeline.execute()
    
        # iterate over tiled arrays
        for arr in pipeline.arrays:
    
            arr = pd.DataFrame(arr)
            arr = arr.rename(columns={'X':'x', 'Y':'y', 'Z':'z'})
            #arr.columns = ['x', 'y', 'z', 'InternalTime', 'ReturnNumber', 'NumberOfReturns',
            #               'amp', 'refl', 'EchoRange', 'dev', 'BackgroundRadiation', 
            #               'IsPpsLocked', 'EdgeOfFlightLine']
            arr.loc[:, 'sp'] = sp
            arr = arr[['x', 'y', 'z', 'Reflectance', 'Deviation', 'ReturnNumber', 'NumberOfReturns', 'sp']] # save only relevant fields
            
            # remove points outside bbox
            arr = arr.loc[(arr.x.between(args.bbox[0], args.bbox[2])) & 
                          (arr.y.between(args.bbox[1], args.bbox[3]))]
            if len(arr) == 0: continue
    
            # identify tile number
            X, Y = (arr[['x', 'y']].min() // args.tile * args.tile).astype(int)
            tile = args.tiles.loc[(args.tiles.x == X) & (args.tiles.y == Y)] 
            if len(tile) == 0: continue   
            tile_n = str(tile.tile.item()).zfill(args.n)
 
            # save to xyz file
            with args.Lock:
                if args.store_tmp_with_sp:
                    with open(os.path.join(args.odir, f'{args.plot_code}{tile_n}.{sp}.xyz'), 'ab') as fh:
                        fh.write(arr.to_records(index=False).tobytes())
                else:
                    with open(os.path.join(args.odir, f'{args.plot_code}{tile_n}.xyz'), 'ab') as fh:
                        fh.write(arr.to_records(index=False).tobytes())

    except Exception as err:
        print('!!!!', scan_pos, err, '!!!!') 

def xyz2ply_w_sp(xyz, args):

    xyz = str(xyz).zfill(args.n)
    L = glob.glob(os.path.join(args.odir, f'{xyz}.*.xyz'))

    if len(L) > 0:

        tmp = pd.DataFrame()

        if args.verbose:
            with args.Lock: print(f'xyz -> ply: {xyz}')

        for fn in glob.glob(os.path.join(args.odir, f'{xyz}.*.xyz')):
            open_file = open(fn, encoding='ISO-8859-1')
            tmp = pd.concat([tmp, pd.DataFrame(np.fromfile(open_file, 
                                                           dtype='float64,float64,float64,float32,float32,uint8,uint8,int64'))])
            os.unlink(fn)

        if len(tmp) > 0:
            tmp.columns = ['x', 'y', 'z', 'refl', 'dev', 'ReturnNumber', 'NumberOfReturns', 'sp']
            ply_io.write_ply(f'{xyz}.ply', tmp)
    
def xyz2ply(xyz_path, args):

    if args.verbose:
        with args.Lock:
            print('xyz -> ply:', xyz_path)
    
    open_file = open(xyz_path, encoding='ISO-8859-1')
    tmp = pd.DataFrame(np.fromfile(open_file, dtype='float64,float64,float64,float32,float32,uint8,uint8,int64'))
    tmp.columns = ['x', 'y', 'z', 'refl', 'dev', 'ReturnNumber', 'NumberOfReturns', 'sp']
    ply_io.write_ply(xyz_path.replace('.xyz', '.ply'), tmp)
    os.unlink(xyz_path)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--riproject', '-r', required=True, type=str, help='path to point cloud')
    #parser.add_argument('--matrix-dir', '-m',  type=str, default='', help='path to rotation matrices')
    parser.add_argument('--plot-code', type=str, default='', help='plot suffix')
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    parser.add_argument('--not-registered', action='store_true', help='include scans that are not registered')
    parser.add_argument('--deviation', type=float, default=15, help='deviation filter')
    parser.add_argument('--reflectance', type=float, nargs=2, default=[-999, 999], help='reflectance filter')
    parser.add_argument('--tile', type=float, default=10, help='length of tile')
    parser.add_argument('--num-prcs', type=int, default=10, help='number of cores to use')
    #parser.add_argument('--prefix', type=str, default='ScanPos', help='file name prefix, deafult:ScanPos')
    parser.add_argument('--buffer', type=float, default=10., help='size of buffer')
    parser.add_argument('--bbox', type=int, nargs=4, default=[], help='bounding box format xmin ymin xmax ymax')
    parser.add_argument('--bbox-only', action='store_true', help='generate bounding box only, do not process tiles')
    parser.add_argument('--bounding-geometry', type=str, default=False, help='a bounding geometry')
    parser.add_argument('--convex-hull', action='store_true', help='fits a convex hull geometry around scan positions')
    parser.add_argument('--rotate-bbox', action='store_true', help='rotate bounding geometry to best fit scan positions')
    parser.add_argument('--save-bounding-geometry', type=str, default=False, help='file where to save bounding geometry')
    parser.add_argument('--save-matrix', type=str, default=False, help='file where to save bounding geometry')
    parser.add_argument('--save-scan-positions', type=str, default=False, help='file where to save scan positions as .shp')
    #parser.add_argument('--global-matrix', type=str, default=False, help='path to global rotation matrix')
    parser.add_argument('--preserve-projection', action='store_true', default=False, help='keep data in geocentric projection')
    parser.add_argument('--target-epsg', type=int, default=False, help='number of cores to use')
    parser.add_argument('--pos', default=[], nargs='*', help='process using specific scan positions')
    #parser.add_argument('--test', action='store_true', help='test using the .mon.rxp')
    parser.add_argument('--store-tmp-with-sp', action='store_true', help='spits out individual tmp files for tiles _and_ scan position')
    parser.add_argument('--verbose', action='store_true', help='print something')

    args = parser.parse_args()
    args.riproject = os.path.abspath(args.riproject) 

    if args.save_matrix and not os.path.isdir(args.save_matrix):
        raise Exception(f'{args.save_matrix} is not a directory')

    if args.buffer == 0: args.buffer = False

    # read in project.rsp
    rsp = os.path.join(args.riproject, 'project.rsp')
    tree = et.parse(rsp)
    root = tree.getroot()
    scanpositions = root.find('scanpositions')

    # pop matrix
    pop = root.find('pop').find('matrix').text
    args.pop = np.array([float(v) for v in pop.split()]).reshape(4, 4)

    # project epsg
    args.pop_epsg = int(root.find('project_epsg').text.split('::')[1])

    # read in scanpos data
    args.ScanPos = {}
    not_registered = []
    for sp in scanpositions.iterchildren():
        name = sp.get('name')#
        if name in args.ScanPos.keys(): 
            raise Exception(f'more than one scan for position: {name}')
        if sp.find('singlescans').find('scan') is not None:
            scan = sp.find('singlescans').find('scan').get('name')
            regi = bool(int(sp.find('registered').text))
            if not regi and not args.not_registered: 
                not_registered.append(name)
                continue
            matrix = np.array([float(v) for v in sp.find('sop').find('matrix').text.split()]).reshape(4, 4)
            try:
                rdbx = glob.glob(os.path.join(args.riproject, 'project.rdb', 'SCANS', name, 'SINGLESCANS', scan, scan + '.rdbx'))[0]
            except Exception as err:
                print(os.path.join(args.riproject, 'project.rdb', 'SCANS', name, 'SINGLESCANS', scan, scan + '.rdbx'))        
            args.ScanPos[name] = {'name':name, 'scan':scan, 'regi':regi, 'mat':np.dot(args.pop, matrix), 'rdbx':rdbx}

    if args.verbose: print('Scan positions not registered:', not_registered)

    # read rotation matrix into array
    xyz = np.zeros((len(args.ScanPos), 3))
    for i, m in enumerate(args.ScanPos.keys()):
        xyz[i, :] = args.ScanPos[m]['mat'][:3, 3]
    
    geometry = [Point(r[0], r[1], r[2]) for r in xyz] 
    xyz = gp.GeoDataFrame(data=xyz, columns=['x', 'y', 'z'], geometry=geometry)
    xyz = xyz.set_crs(epsg=args.pop_epsg)
    xyz['ScanPos'] = args.ScanPos.keys()

    if args.preserve_projection:
        pass
    elif args.target_epsg:
        if pyproj.CRS.from_epsg(args.target_epsg).is_geographic:
            raise Exception(f'target epsg:{target_epsg} is a geographic projection and is currently not supported')
        xyz = xyz.to_crs(epsg=args.target_epsg)
        for k, v in args.ScanPos.items():
            coords = [xyz.loc[xyz.ScanPos == k].geometry.x.item(), 
                      xyz.loc[xyz.ScanPos == k].geometry.y.item(), 
                      xyz.loc[xyz.ScanPos == k].geometry.z.item()]
            args.ScanPos[k]['mat'][:3, 3] = coords
    else:
        xyz['a'] = 1
        xyz[['x', 'y', 'z', 'a']] = np.dot(np.linalg.inv(args.pop), xyz[['x', 'y', 'z', 'a']].T).T
        geometry = [Point(r.x, r.y, r.z) for r in xyz.itertuples()]
        xyz.geometry = geometry
        xyz.crs = None 
        for k, v in args.ScanPos.items():
            coords = [xyz.loc[xyz.ScanPos == k].geometry.x.item(), 
                      xyz.loc[xyz.ScanPos == k].geometry.y.item(), 
                      xyz.loc[xyz.ScanPos == k].geometry.z.item()]
            args.ScanPos[k]['mat'][:3, 3] = coords

    # bbox [xmin, ymin, xmax, ymax]
    if args.bounding_geometry and len(args.bbox) > 0:
        raise Exception('a bounding geometry and bounding box have been specified')
    if args.bbox:
        geometry = Polygon(((args.bbox[0], args.bbox[1]), (args.bbox[0], args.bbox[3]), 
                            (args.bbox[2], args.bbox[3]), (args.bbox[2], args.bbox[1]), 
                            (args.bbox[0], args.bbox[1])))
        extent = gp.GeoDataFrame([0], geometry=[geometry], columns=['id'])
    elif args.bounding_geometry:
        extent = gp.read_file(args.bounding_geometry)#.buffer(args.buffer, join_style='mitre')
    elif args.rotate_bbox:
        extent = gp.GeoDataFrame([0], columns=['id'], geometry=[xyz.unary_union.minimum_rotated_rectangle])
    elif args.convex_hull:
        extent = gp.GeoDataFrame([0], columns=['id'], geometry=[xyz.unary_union.convex_hull])
    else:
        extent = gp.GeoDataFrame([0], columns=['id'], geometry=[xyz.unary_union.envelope])
    if args.save_bounding_geometry: 
        extent.to_file(args.save_bounding_geometry, crs=args.pop_epsg if args.preserve_projection else 
                                                        args.target_epsg if args.target_epsg else None)
        if args.verbose: print(f'bounding geometry saved to {args.save_bounding_geometry}') 
    if args.buffer: 
        extent.geometry = extent.buffer(args.tile + args.buffer, join_style='mitre' if not args.convex_hull else 1)    
        if args.save_bounding_geometry: 
            extent.to_file(args.save_bounding_geometry.replace('.shp', '.buffer.shp'), 
                                                               crs=args.pop_epsg if args.preserve_projection else
                                                               args.target_epsg if args.target_epsg else None)
    args.bbox = (extent.exterior.bounds.values[0] // args.tile) * args.tile
    if args.verbose: print('bounding box:', args.bbox)

    # create tile db
    X, Y = np.meshgrid(np.arange(args.bbox[0], args.bbox[2], args.tile),
                       np.arange(args.bbox[1], args.bbox[3], args.tile))
    XY = np.vstack([X.flatten(), Y.flatten()]).T.astype(int)
    args.tiles = gp.GeoDataFrame(data=XY, columns=['x', 'y'], geometry=[Point(r[0], r[1]) for r in XY])
    args.tiles = gp.sjoin(args.tiles, extent, how='inner')
    
    args.tiles.loc[:, 'tile'] = range(len(args.tiles))
    args.tiles = args.tiles[['x', 'y', 'tile', 'geometry']]
    args.n = 4 #len(str(len(args.tiles)))

    if len(args.pos) > 0:
        args.pos = [f'ScanPos{p.zfill(3)}' for p in args.pos]
        if args.verbose: print('processing only:', args.pos)
        args.ScanPos = {k:args.ScanPos[k] for k in args.pos if k in args.ScanPos.keys()} 
        #args.ScanPos = [os.path.join(args.riproject, p) for p in args.pos]

    # write tile index
    args.tiles[['tile', 'x', 'y']].to_csv(os.path.join(args.odir, 'tile_index.dat'), 
                                          sep=' ', index=False, header=False)

    if args.save_matrix: 
        for k, v in args.ScanPos.items():
            np.savetxt(os.path.join(args.save_matrix, f'{v["name"]}.DAT'), v['mat'], fmt='%0.10f')               
    if args.save_scan_positions: xyz.to_file(args.save_scan_positions)
    if args.bbox_only: sys.exit()

    # read in and tile scans
    Pool = multiprocessing.Pool(args.num_prcs)
    m = multiprocessing.Manager()
    args.Lock = m.Lock()
    #[tile_data(args.ScanPos[sp], args) for sp in sorted(args.ScanPos.keys())]
    Pool.starmap(tile_data, [(args.ScanPos[sp], args) for sp in sorted(args.ScanPos.keys())])

    # write to ply - reusing Pool
    if args.store_tmp_with_sp:
        Pool.starmap_async(xyz2ply_w_sp, [(xyz, args) for xyz in np.sort(args.tiles.tile)])
    else:
        xyz = glob.glob(os.path.join(args.odir, '*.xyz'))
        Pool.starmap_async(xyz2ply, [(xyz, args) for xyz in np.sort(xyz)])
    #[xyz2ply_w_sp(xyz, args) for xyz in np.sort(args.tiles.tile)]
    Pool.close()
    Pool.join()

    print(f'runtime: {(datetime.now() - start).seconds}')
