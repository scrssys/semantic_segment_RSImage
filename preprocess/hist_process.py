

import os, sys,gc
import numpy as np
import gdal

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from ulitities.base_functions import get_file, load_img_by_gdal,load_img_by_gdal_geo



Flag_Hist_match=0 #0:直方图统计，1:直方图匹配


# input_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/test_global/images_ori/xielan/'
input_dir ='/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/test_global/images_forClass/test_normal/'
B=4
S=1024
result_bits='int16'
dest_file='/home/omnisky/PycharmProjects/data/originaldata/global/16bit_miandiantaiguo/total_zy3test.csv'
# src_file='/home/omnisky/PycharmProjects/data/test/global/test_csv/src.csv'

# test_file = '/home/omnisky/PycharmProjects/data/test/global/test_csv/histMap.csv'

output_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/test_global/images_forClass/test_histM/'

def get_hist(files, bands=4, scale=1024):
    print("[Info] Statisify histogram from images...")
    hist = np.zeros((scale,bands),np.uint64)
    in_files =[]
    if isinstance(files,str):
        in_files.append(files)
    elif isinstance(files,list):
        in_files=files
    # in_files=list(in_files)


    for file in in_files:
        print("\n\t[info]deal:{}".format(file))
        img = load_img_by_gdal(file)
        a,b,c = img.shape
        if c != bands:
            print("Warning: bands of img is :{}, but setting bands is:{}".format(c,bands))
        real_b = min(c,bands)
        for i in range(real_b):
            h, bin_edges = np.histogram(img[:,:, i], bins=range(scale+1))
            h = np.array(h,np.uint64)
            hist[:,i] +=h[:scale]
            # plt.plot(range(scale), h)
            # plt.show()


    return hist



def save_hist_to_csv(in_dir,csv_file,bands,scale):
    input_files, _ = get_file(in_dir)

    Hist = get_hist(input_files, bands, scale)

    # Data = {'band_1':Hist[:,0], 'band_2':Hist[:,1],'band_3':Hist[:,2],'band_4':Hist[:,3]}

    df = pd.DataFrame(Hist)
    df.to_csv(csv_file)



def calc_cum_hist(hist):
    read_scale = len(hist)
    cum_hist = np.zeros((read_scale, 1), np.float64)

    for i in range(read_scale):
        for j in range(i):
            cum_hist[i] += hist[j]
    total_pixel = np.zeros(1,np.uint64)
    total_pixel = np.sum(hist)
    f_cum_hist = np.zeros((read_scale,1), np.float32)
    f_cum_hist = cum_hist/total_pixel

    return f_cum_hist




def HistMappingGML(scr, dest, scale):

    diff = np.zeros((scale,scale), np.float)

    histMap = np.zeros(scale,np.uint16)
    # for j in range(scale):
    #     histMap[j]=scale-1

    # minValue = 0.0
    startX = 0
    # lastStartY = 0
    lastEndY = 0
    startY = 0
    endY = 0
    i = 0
    x = 0
    y = 0
    # a = 1
    # b = 0

    for y in range(scale):
        for x in range(scale):
            diff[x,y] = abs(dest[x]-scr[y])

    for x in range(scale):
        minValue=diff[x,0]
        for y in range(scale):
            if minValue>diff[x,y]:
                endY=y
                minValue=diff[x,y]
        if endY !=lastEndY:
            for i in range(startY,endY+1):
                if endY==startY:
                    temp=int((startX+x)/2+0.5)
                    if temp<0:
                        temp=0
                    histMap[i]=temp
                elif startX==x:
                    histMap[i]=x
                else:
                    a = (endY-startY)/(x-startX)
                    b = endY-a*x
                    temp = int((i-b)/a +0.5)
                    histMap[i]=temp

            lastStartY=startY
            lastEndY=endY
            startY=lastEndY+1
            startX=x+1

    for i in range(endY+1,scale):
        histMap[i]=scale-1

    for i in range(scale):
        # if histMap[i]<histMap[i-1]:
        #     histMap[i]=histMap[i-1]
        if histMap[i]>=scale:
            histMap[i]=scale-1



    return histMap



def HistMappingSML(scr, dest, scale):

    diff = np.zeros((scale,scale), np.float)

    histMap = np.zeros(scale,np.uint16)
    for j in range(scale):
        histMap[j]=scale-1

    endY = 0


    for y in range(scale):
        for x in range(scale):
            diff[x,y] = abs(dest[x]-scr[y])

    for x in range(scale):
        minValue=diff[x,0]
        for y in range(scale):
            if minValue>diff[x,y]:
                endY=y
                minValue=diff[x,y]
            histMap[x] = endY

    return histMap




if __name__=='__main__':

    if Flag_Hist_match==0:
        save_hist_to_csv(input_dir,dest_file,B,S)
    else:
        all_dest = np.array(pd.read_csv(dest_file))

        files,_=get_file(input_dir)

        for file in tqdm(files):
            absname = os.path.split(file)[1]
            print('\n\t[Info] images:{}'.format(absname))
            absname = absname.split('.')[0]
            absname = ''.join([absname, '.tif'])

            img, geoinfo= load_img_by_gdal_geo(file)
            H,W,C = img.shape
            assert(C==B)

            all_src = get_hist(file,B,S)
            result = []

            if not os.path.isdir(output_dir):
                print("Warning: output directory does not exist")
                os.mkdir(output_dir)

            for t in range(C):
                print("[Info]\t\tband:{}".format(t+1))
                tmp =np.zeros((H,W), np.uint16)
                dest=all_dest[:,t+1]
                src = all_src[:, t]
                assert (len(dest) == len(src))
                assert(len(dest)==S)

                cum_dest = calc_cum_hist(dest)
                for i in range(S):
                    if i == S - 1:
                        cum_dest[i] *= S
                    else:
                        cum_dest[i] *= S - 1

                cum_src = calc_cum_hist(src)
                for i in range(S):
                    if i == S - 1:
                        cum_src[i] *= S
                    else:
                        cum_src[i] *= S - 1

                histM = HistMappingGML(cum_src, cum_dest, S)
                # plt.plot(range(S),histM)
                ori_img=img[:,:,t]

                for i in range(S):
                    index = np.where(ori_img==i)
                    tmp[index]=histM[i]



                # plt.imshow(tmp)
                # plt.show()
                result.append(tmp)

            outputfile = os.path.join(output_dir, absname)
            driver = gdal.GetDriverByName("GTiff")

            if '8' in result_bits:
                outdataset = driver.Create(outputfile, W, H, C, gdal.GDT_Byte)
                # outdataset.SetGeoTransform(geoinfo)
            elif '16' in result_bits:
                outdataset = driver.Create(outputfile, W, H, C, gdal.GDT_UInt16)
                # outdataset.SetGeoTransform(geoinfo)

            outdataset.SetGeoTransform(geoinfo)
            for i in range(C):
                outdataset.GetRasterBand(i + 1).WriteArray(result[i])

            del outdataset
            print("Result saved to:{}".format(outputfile))
            gc.collect()







        '''
        all_src = np.array(pd.read_csv(src_file))
        dest = all_dest[:, 1]
        src = all_src[:, 1]

        assert (len(dest) == len(src))

        scale = len(dest)

        cum_dest = calc_cum_hist(dest)
        for i in range(scale):
            if i == scale - 1:
                cum_dest[i] *= scale
            else:
                cum_dest[i] *= scale - 1

        cum_src = calc_cum_hist(src)
        for i in range(scale):
            if i == scale - 1:
                cum_src[i] *= scale
            else:
                cum_src[i] *= scale - 1

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        plt.sca(ax1)
        plt.plot(range(scale), cum_src)
        plt.sca(ax2)
        plt.plot(range(scale), cum_dest)
        # plt.show()

        histM = HistMappingGML(cum_src, cum_dest, scale)

        print(histM)
        df = pd.DataFrame(histM)
        df.to_csv(test_file)
        plt.sca(ax3)
        plt.plot(range(scale), histM)
        plt.show()
        '''



