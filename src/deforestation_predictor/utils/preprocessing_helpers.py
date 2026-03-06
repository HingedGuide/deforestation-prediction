import os
from osgeo import gdal
import errno


def check_folder(folder_dir):
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_tiff(tiff_file):
    #print(tiff_file)
    data = gdal.Open(tiff_file).ReadAsArray()
    return data