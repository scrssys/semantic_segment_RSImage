import os, sys
import numpy as np
import subprocess
import fnmatch
import numpy.ma as ma

from ulitities.base_functions import get_file
inputdir = '/home/omnisky/PycharmProjects/data/samples/cloud_samples/test/src_16bits'

outputdir='/home/omnisky/PycharmProjects/data/samples/cloud_samples/test/src'


def getStatistics(inputRaster):
    from osgeo import gdal,ogr,osr
    srcRaster = gdal.Open(inputRaster)
    # iterate through bands
    arr=[]
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId + 1
        band = srcRaster.GetRasterBand(bandId)
        stats=band.GetStatistics(0,1)
        arr.append(stats[2])
        arr.append(stats[3])
    print(arr)
    return arr
def convert_to_8Bit2(inputRaster, outputRaster,
                    outputDataType='Byte',
                    outputFormat='GTiff',
                    stretch_type='rescale',
                     nodata=65535,
                    percentiles=[0.5, 95]):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, rescale]
        if clip, scaling is done strictly between 0 65535
        if rescale, each band is rescaled to a min and max
        set by percentiles
    '''
    from osgeo import gdal,ogr,osr
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputDataType, '-of',
           outputFormat]

    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId + 1
        band = srcRaster.GetRasterBand(bandId)
        # band.SetNoDataValue ( -333 )
        if stretch_type == 'rescale':
            band.SetNoDataValue(nodata)

            bmin = band.GetMinimum()
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            band_arr_tmp = band.ReadAsArray()

            index = np.where(band_arr_tmp==nodata)
            new_data = np.asarray(band_arr_tmp, dtype=np.float)
            new_data[index]=np.nan
            bmin = np.nanpercentile(new_data.flatten(),
                                 percentiles[0])
            bmax = np.nanpercentile(new_data.flatten(),
                                 percentiles[1])
        elif isinstance(stretch_type, dict):
            bmin, bmax = stretch_type[bandId]
        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print("Conversin command:", cmd)
    subprocess.call(cmd)




if __name__=='__main__':
    if not os.path.isdir(inputdir):
        print("Please check input directory:{}".format(inputdir))
        sys.exit(-1)

    if not os.path.isdir(outputdir):
        print('Warning: output directory is not existed')
        os.mkdir(outputdir)

    files,_=get_file(inputdir)
    for file in files:
        absname = os.path.split(file)[1]
        outputfile = os.path.join(outputdir,absname)
        convert_to_8Bit2(file, outputfile,
                         outputDataType='Byte',
                         stretch_type='rescale',
                         nodata=65535,
                         percentiles=[1, 99])


