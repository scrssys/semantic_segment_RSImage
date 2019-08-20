import os,sys
import numpy as np
import cv2
import gdal
from tqdm import tqdm
from ulitities.base_functions import get_file, load_img_by_gdal, find_file


inputdir = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/originaldata/RGBLabel/sat_urban_rgb'
outputdir = '/home/omnisky/PycharmProjects/data/samples/buildings/train'
patch_size=5000

if __name__=='__main__':
    if not os.path.isdir(inputdir):
        print("Error: input directory is not existed")
        sys.exit(-1)
    if not os.path.isdir(outputdir):
        print("Warning: output directory is not existed")
        os.mkdir(outputdir)
    out_label_dir=outputdir+'/label/'
    out_src_dir = outputdir + '/src/'

    label_list, img_list =[], []

    label_files, _=get_file(inputdir+'/label')
    img_files =[]
    for file in label_files:
        absname = os.path.split(file)[1]
        absname = absname.split('.')[0]
        img_f = find_file(inputdir+'/src',absname)
        img_files.append(img_f)
    # img_files = list()
    # img_files, _=get_file(inputdir+'/src')
    assert(len(label_files)==len(img_files))
    name_list =[]
    for file in label_files:
        l_img = load_img_by_gdal(file, grayscale=True)
        if len(l_img)==0:
            continue
        label_list.append(l_img)
        absname = os.path.split(file)[1]
        only_name = absname.split('.')[0]
        name_list.append(only_name)
        src_file = inputdir+'/src/'+absname
        s_img = load_img_by_gdal(src_file)
        if len(s_img)==0:
            continue
        img_list.append(s_img)

    assert(len(label_list)==len(img_list))

    for i in tqdm(range(len(label_list))):
        only_name = name_list[i]
        print("deal: {}".format(only_name))
        label=np.asarray(label_list[i], np.uint8)
        img=np.asarray(img_list[i], np.uint8)
        assert(label.shape[:2]==img.shape[:2])
        h,w,c = img.shape
        if h//patch_size==0 or w//patch_size==0:
            crop_label = label
            crop_img = img

            cv2.imwrite(out_label_dir+only_name+'.png', crop_label)
            driver = gdal.GetDriverByName("GTiff")
            outdataset = driver.Create(out_src_dir+only_name+'.png', w, h, c, gdal.GDT_Byte)
            if outdataset == None:
                print("create dataset failed!\n")
                sys.exit(-2)
            if c == 1:
                outdataset.GetRasterBand(1).WriteArray(crop_img)
            else:
                for s in range(c):
                    outdataset.GetRasterBand(s + 1).WriteArray(crop_img[:,:,s])
            del outdataset
        else:
            for i in range(h//patch_size):
                for j in range(w//patch_size):
                    if i==h//patch_size-1:
                        crop_label = label[i * patch_size:h, j * patch_size:(j + 1) * patch_size]
                        crop_img = img[i * patch_size:h, j * patch_size:(j + 1) * patch_size, :]
                    elif j==w//patch_size-1:
                        crop_label = label[i * patch_size:(i + 1) * patch_size, j * patch_size:w]
                        crop_img = img[i * patch_size:(i + 1) * patch_size, j * patch_size:w, :]
                    else:
                        crop_label = label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                        crop_img = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]

                    t_h,t_w = crop_label.shape
                    cv2.imwrite(out_label_dir + only_name+'_'+str(i)+'_'+str(j) + '.png', crop_label)
                    driver = gdal.GetDriverByName("GTiff")
                    outdataset = driver.Create(out_src_dir + only_name + '_'+str(i)+'_'+str(j) +'.png', t_w, t_h, c, gdal.GDT_Byte)
                    if outdataset == None:
                        print("create dataset failed!\n")
                        sys.exit(-2)
                    if c == 1:
                        outdataset.GetRasterBand(1).WriteArray(crop_img)
                    else:
                        for s in range(c):
                            outdataset.GetRasterBand(s + 1).WriteArray(crop_img[:,:,s])
                    del outdataset