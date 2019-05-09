
import os, sys


input_raster_file = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/test_global/pred/2019-04-11_19-31-32--miandiantaiguo-histmatched/c-ZY302918320140104s.tif'
output_shp_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/test_global/pred/2019-04-11_19-31-32--miandiantaiguo-histmatched/'

import gdal,osr,ogr
def polygonize(rasterTemp, outShp, sieveSize=1):
    sourceRaster = gdal.Open(rasterTemp)
    band = sourceRaster.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # If shapefile already exist, delete it
    if os.path.exists(outShp):
        driver.DeleteDataSource(outShp)

    outDatasource = driver.CreateDataSource(outShp)
    # get proj from raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(sourceRaster.GetProjectionRef())
    # create layer with proj
    outLayer = outDatasource.CreateLayer(outShp, srs)
    # Add class column (1,2...) to shapefile

    newField = ogr.FieldDefn('grid_code', ogr.OFTInteger)
    outLayer.CreateField(newField)

    gdal.Polygonize(band, None, outLayer, 0, [], callback=None)

    outDatasource.Destroy()
    sourceRaster = None
    band = None

    try:
        # Add area for each feature
        ioShpFile = ogr.Open(outShp, update=1)

        lyr = ioShpFile.GetLayerByIndex(0)
        lyr.ResetReading()

        field_defn = ogr.FieldDefn("Area", ogr.OFTReal)
        lyr.CreateField(field_defn)
    except:
        print("Can not add filed of Area!")

    for i in lyr:
        # feat = lyr.GetFeature(i)
        geom = i.GetGeometryRef()
        area = round(geom.GetArea())

        lyr.SetFeature(i)
        i.SetField("Area", area)
        lyr.SetFeature(i)
        # if area is less than inMinSize or if it isn't forest, remove polygon
        if area < sieveSize:
            lyr.DeleteFeature(i.GetFID())
    ioShpFile.Destroy()

    return outShp


if __name__=='__main__':

    if not os.path.isfile(input_raster_file):
        print("Error: Please input a raster file!")
        sys.exit(-1)

    if not os.path.isdir(output_shp_dir):
        print("Warning: output directory do not exist!")
        os.mkdir(output_shp_dir)

    absname = os.path.split(input_raster_file)[1]
    print('\n\t[Info] images:{}'.format(absname))
    absname = absname.split('.')[0]
    absname = ''.join([absname, '4.shp'])
    shp_file = os.path.join(output_shp_dir,absname)

    polygonize(input_raster_file, shp_file)

