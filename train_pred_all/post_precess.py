import os
import numpy as np
from  skimage.io import imread, imsave
import PIL.Image
from tqdm import tqdm
PIL.Image.MAX_IMAGE_PIXELS=10000000000
from skimage.morphology import remove_small_objects, watershed
from skimage.morphology import erosion, dilation, rectangle,opening,closing,disk

def remove_small_objects_deal(predict, mask):
    lst = os.listdir(predict)
    for f in tqdm(lst):
        if not f.endswith(".png"):
            continue
        img_pre = imread(os.path.join(predict, f))
        # absname = os.path.split(f)[1]
        # img_mask = imread()
        # img_pre[np.where(img_pre==255)]=1
        # img_mask[np.where(img_mask==2)]=0
        # img_mask[ np.where ( img_mask == 6 ) ] = 1
        # img_mask[ np.where ( img_mask == 255 ) ] = 1
        # if img_mask.max() > 1:
        #     print("error")
        size = 3
        im_open = opening(img_pre, disk(size))
        img_pre = im_open
        img_pre = img_pre.astype(np.bool)
        img_pre = remove_small_objects(img_pre, 100).astype(np.uint8)



        imsave(os.path.join(mask, f), img_pre)


def dowork (predict,mask):
    lst = os.listdir(predict)
    for f in lst:
        if not f.endswith(".png"):
            continue
        img_pre = imread(os.path.join(predict, f))
        # absname = os.path.split(f)[1]
        # img_mask = imread()
        # img_pre[np.where(img_pre==255)]=1
        # img_mask[np.where(img_mask==2)]=0
        # img_mask[ np.where ( img_mask == 6 ) ] = 1
        # img_mask[ np.where ( img_mask == 255 ) ] = 1
        # if img_mask.max() > 1:
        #     print("error")
        size = 5
        im_open = opening(img_pre, disk(size))
        im_close = closing(im_open, disk(size))

        img_pre = im_open
        img_pre = img_pre.astype(np.bool)
        # img_pre = remove_small_objects(img_pre, 100).astype(np.uint8)
        imsave(os.path.join(mask, f), img_pre)


def calMetric(predict,mask):
    lst=os.listdir(predict)
    f_score=[]
    percision_score = [ ]
    recall_score = [ ]
    for f in lst:
        if not  f.endswith(".png"):
            continue
        img_pre=imread(os.path.join(predict,f))
        # absname = os.path.split(f)[1]
        img_mask=imread(os.path.join(mask,f))
        # img_pre[np.where(img_pre==255)]=1
        # img_mask[np.where(img_mask==2)]=0
        # img_mask[ np.where ( img_mask == 6 ) ] = 1
        # img_mask[ np.where ( img_mask == 255 ) ] = 1
        if img_mask.max()>1:
            print("error")
        size=3
        im_open = opening ( img_pre , disk ( size ) )
        img_pre = im_open
        img_pre = img_pre.astype(np.bool)
        img_pre = remove_small_objects(img_pre, 100).astype(np.uint8)
        im1=img_pre
        im2=img_mask
        valid = im1 >= 1
        iou = np.sum ( valid * (im1 == im2) )
        acc = np.sum ( (im1 == im2) ) / (im1.shape[ 0 ] * im1.shape[ 1 ])
        # iou=iou.sum()
        pre = im1.sum ( )
        p_true = im2.sum ( )
        f_score.append ( (2 * iou + 0.001) / (p_true + pre + 0.001) )
        percision_score.append(iou/pre)
        recall_score.append(iou/p_true)
        print ( '{} im1:{} p_true:{} iou:{} acc:{:.4f}  recall:{:.4f} precision:{:.4f}  f1:{:.4f}'.format ( f , pre ,
                                                                                                            p_true ,
                                                                                                            iou , acc ,
                                                                                                            iou / p_true ,
                                                                                                            iou / pre ,
                                                                                                            (
                                                                                                                        2 * iou + 0.001) / (
                                                                                                                        p_true + pre + 0.001) ) )


    print(f_score)

    print ("recall: "+ str(np.mean ( recall_score )) )
    print ( "percision: " + str ( np.mean ( percision_score ) ) )
    print ( "f: " + str ( np.mean ( f_score ) ) )


if __name__=="__main__":
    remove_small_objects_deal("/home/omnisky/PycharmProjects/data/samples/cloud_samples/test/pred/2019-08-19_08-17-00"
                              ,"/home/omnisky/PycharmProjects/data/samples/cloud_samples/test/pred/masks")
    # calMetric("/home/omnisky/PycharmProjects/data/samples/cloud_samples/test/pred/2019-08-16_15-51-46"
    #           ,"/home/omnisky/PycharmProjects/data/samples/cloud_samples/test/label")