import gdal
import os ,sys
from ulitities.base_functions import load_img_by_gdal_geo


input_file = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/test/WHU/images/test_2w.png'
output_file = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/test/WHU/images/test_2w-C.png'


if __name__=="__main__":
    try:
        data, geotransform = load_img_by_gdal_geo(input_file)
        print("Geotransform:{}".format(geotransform))
    except:
        print("Error: Failde load image..")
        sys.exit(-1)
    tmp = list(geotransform)
    tmp[-1]=-1*tmp[-1]

    new_geo=tuple(tmp)
    print("new Geotransform:{}".format(new_geo))

    w,h,c = data.shape

    driver = gdal.GetDriverByName("GTiff")
    # driver = gdal.GetDriverByName("PNG")
    # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
    outdataset = driver.Create(output_file, w, h, c, gdal.GDT_Byte)
    outdataset.SetGeoTransform(new_geo)
    if outdataset == None:
        print("create dataset failed!\n")
        sys.exit(-2)
    if c == 1:
        outdataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(c):
            outdataset.GetRasterBand(i + 1).WriteArray(data[:,:,i])
    del outdataset
