

import os, sys
from ulitities.base_functions import get_file, polygonize




if __name__=='__main__':


    shp_file = ''.join([output_dir, '/', abs_filename, '.shp'])
    polygonize(output_file, shp_file)