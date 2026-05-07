"""
rxp-pipeline: Tools to transform RIEGL terrestrial LiDAR data

This package provides tools to preprocess co-registered RIEGL TLS data into tiled, 
downsampled PLY point clouds with configurable tile size, overlap, buffer and filtering.
"""

__version__ = "0.2.0"
__author__ = "Phil Wilkes and Wanxin Yang"
__description__ = "Tools to transform RIEGL terrestrial LiDAR data"

from . import ply_io

__all__ = [
    "ply_io",
]
