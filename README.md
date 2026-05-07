# rxp-pipeline: Tools to transform RIEGL terrestrial LiDAR data

[![DOI](https://zenodo.org/badge/632593854.svg)](https://doi.org/10.5281/zenodo.15196450)

## Overview

Pipeline and tools to convert co-registered RIEGL TLS data (either raw `.rxp` files or RiSCAN Pro exported `.rdbx` files) into tiled and downsampled PLY point clouds with configurable tile size, overlap, buffer and filtering. These methods are used by UCL Geography to preprocess TLS data for downstream processing workflows including tree extraction and biomass estimation.



## Authors

- Dr Phil Wilkes
- Dr Wanxin Yang

## Citation

Wilkes, P., & Yang, W. (2025). *rxp-pipeline: Tools to transform RIEGL terrestrial LiDAR data*. Zenodo. https://doi.org/10.5281/zenodo.15196452

## Installation

#### Create and activate the conda environment:

```bash
    $ conda env create -f environment.yml
    $ conda activate pdal
```

#### Compiling PDAL with python bindings and .rxp support
    
1.  Download the [PDAL current release](https://pdal.io/download.html#current-release-s).
    
    Example commands in linux:
    
    ```bash
        $ wget https://github.com/PDAL/PDAL/releases/download/2.3.0/PDAL-2.3.0-src.tar.gz
        $ tar -xf PDAL-2.3.0-src.tar.gz
    ```    

2.  Download the `rivlib-2_5_10-x86_64-linux-gcc9.zip` (make sure to get the gcc9 version) and the the `rdblib-2.4.0-x86_64-linux.tar.gz` from the memebers area of the RIEGL website. 
    - Unzip `rivlib-2_5_10-x86_64-linux-gcc9.zip` and add an environmental variable to point at the directory `export RiVLib_DIR=/path/to/rivlib-2_5_10-x86_64-linux-gcc9`. 
    - Untar `rdblib-2.4.0-x86_64-linux.tar.gz` and add an environmental variable to point at the directory `export rdb_DIR=/path/to/rdblib-2.4.0-x86_64-linux/interface/cpp`

3.  Before running cmake
    - edit line 58 of `cmake/options.cmake` to `"Choose if RiVLib support should be built" True)`
    - edit line 63 of `cmake/options.cmake` to `"Choose if rdblib support should be built" True)`
    - edit line 56 of `plugins/rxp/io/RxpReader.hpp` to `const bool DEFAULT_SYNC_TO_PPS = false;`
    
    <br>
    Then, follow the [PDAL Unix Compilation](https://pdal.io/development/compilation/unix.html) notes to compile PDAL. Example commands in Linux:
    
    ```bash 
        $ cd /path/to/PDAL-2.3.0-src
        $ mkdir build
        $ cd build
        $ cmake -G Ninja ..
        $ ninja
        $ ls bin/pdal
        bin/pdal
    ```    
    <br>
    Next, add the this bin path to the environmental variable $PATH 
    
    ```bash
        export PATH=/path/to/PDAL-2.3.0-src/build/bin:$PATH
    ```
<br>

4. Copy lib files to pdal lib, this is required to open `.rxp` and `.rdbx` in Python.
    ```bash
        $ cp build/lib/libpdal_plugin_reader_*.so /path/to/.conda/envs/pdal/lib/.
    ```




## Project Data File Structure

File structure should be as below where `ScanPos001`, `ScanPos002`, ... `ScanPosXXX` are the raw data from the scanner and matrix are derived RiSCAN Pro. Anything in `extraction` is _temporary_ and could be deleted i.e. don't put anything in there that needs to be kept. `clouds` and `models` are kept for downstream use.

```
20XX-XX-XX.XXX.riproject
  ├── ScanPos001
  ├── ScanPos002
  ├── ScanPosXXX
  ├── matrix
  |   ├── ScanPos001.DAT
  |   ├── ScanPos002.DAT
  |   └── ScanPosXXX.DAT
  ├── extraction
  |   ├── rxp2ply
  |   |   └── <tiles created by rxp2ply.py
  |   ├── downsample
  |   |   └── <tiles created by downsample.py
  |   ├── fsct
  |   |   └── output from FSCT
  |   └── tile_index.dat
  ├── clouds
  |   └── <trees extracted with FSCT or other>
  └── models
      └── <QSMs from either TreeQSM or Treegraph>
```

## Processing Workflow

Assuming you have prepared the `pdal` environment and set up your RIEGL project with matrix files:

```bash
# Activate environment
$ conda activate pdal

# Navigate to project extraction directory
$ mkdir -p /path/to/riproject/extraction/rxp2ply
$ cd /path/to/riproject/extraction/rxp2ply
```

#### Step 1. Convert plot-level rxp or rdbx files to tiled PLY with full resolution

```bash
# Case 1: If process RIEGL raw project (.rxp files)
$ python rxp2ply.py \
    --project /PATH/TO/XXXX.riproject \
    --matrix-dir /PATH/TO/matrix \
    --odir ./ \
    --tile 10 \
    --deviation 15 \
    --reflectance -20 5 \
    --rotate-bbox \
    --store-tmp-with-sp \
    --verbose 

# Case 2: If process RISCAN PRO project (.rdbx files)
$ python riscan2ply.py \
    --riproject /PATH/TO/XXXX.RiSCAN \
    --odir ./ \
    --tile 10 \
    --deviation 15 \
    --reflectance -20 5 \
    --rotate-bbox \
    --store-tmp-with-sp \
    --verbose 
```

Parameters explanation:
- `--tile <size>`: Defines the size of each tile (default: 20), unit in metre.
- `--tile-overlap <size>`: Sets the overlap between tiles (default: 5), unit in metre.
- `--buffer <size>`: Sets the size of buffer around the bounding box (plot boundary) (default: 10), unit in metre.
- `--deviation <value>`: Filters points based on deviation, keeping only those below the specified value (default: 15).
- `--reflectance <min> <max>`: Filters points based on reflectance values within the specified range (e.g., -20 to 5).
- `--rotate-bbox`: Rotates the bounding box to better align with the point cloud's orientation.
- `--store-tmp-with-sp`: spits out individual tmp files for scans and merge them back afterwards.
- `--verbose`: Enables detailed logging for debugging and progress tracking.
- `--pos <position>`: Specifies a scan position identifier to process a single scan (e.g., `001`).

More available parameters can be viewed with the `-h` flag:

```bash
$ python rxp2ply.py -h
$ python riscan2ply.py -h
```


#### Step 2. Downsample the data to a uniform point density
```bash
$ mkdir ../downsample
$ cd ../downsample
$ python downsample.py -i ../rxp2ply/ --length 0.02 --verbose
```
Parameters explanation:
- `-i <input_path>`: Specifies the input directory containing tiled point clouds.
- `--length <voxel_size>`: Defines the voxel size for downsampling (default: 0.02), unit in metre.
- `--verbose`: Enables detailed logging for debugging and progress tracking.


#### Step 3. Generate spatial tile index
```bash
$ cd ..
$ python tile_index.py -i downsample/*.ply -t tile_index.dat
```

## Downstream processing: Extract individual trees 
#### Method 1: TLS2trees 
Instructions see https://github.com/tls-tools-ucl/TLS2trees

#### Method 2: RCT-pipeline 
Instructions see https://github.com/wanxinyang/rct-pipeline
