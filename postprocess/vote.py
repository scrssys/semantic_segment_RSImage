import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from ulitities.base_functions import load_img_by_cv2

target_values=[0, 1, 2] #

input_path = '../../data/test/paper/'
input_masks=['pred_224/combined/unet_jaccard_yujiang_test_4bands_combined.png',
             'pred_288/combined/unet_jaccard_yujiang_test_4bands_combined.png',
             'pred_256/combined/unet_jaccard_yujiang_4bands_combined.png',
             'pred_256/combined/unet_multiclass_yujiang_4bands_combined.png',
             'pred_256/combined/unet_notonehot_yujiang_test_4bands_combined.png',
             'pred_256/combined/unet_onlyjaccard_yujiang_4bands_combined.png']

output_file = '../../data/test/paper/voted/unet_yujiang_4bands_voted0.png'


def check_input_file(path, masks):
    ret, img_1 = load_img_by_cv2(path+masks[0],grayscale=True)
    assert (ret == 0)

    height, width = img_1.shape
    num_img = len(masks)

    for next_index in range(1,num_img):
        next_ret, next_img=load_img_by_cv2(path+masks[next_index],grayscale=True)
        assert (next_ret ==0 )
        next_height, next_width = next_img.shape
        assert(height==next_height and width==next_width)
    return height, width



def vote_per_image(height, width, path, masks):
    num_target = len(target_values)

    mask_list = []
    for tt in range(len(masks)):
        ret, img = load_img_by_cv2(path+masks[tt],grayscale=True)
        assert(ret ==0)
        mask_list.append(img)

    vote_mask=np.zeros((height,width), np.uint8)

    for i in tqdm(range(height)):
        for j in range(width):
            # record=np.zeros(256,np.uint8)
            record = np.zeros(num_target, np.uint8)
            for n in range(len(mask_list)):
                mask=mask_list[n]
                pixel=mask[i,j]
                record[pixel] +=1

            # """Alarming"""
            if record.argmax()==0: # if argmax of 0 = 125 or 255, not prior considering background(0)
                record[0] -=1
            # print("record:{}".format(record))
            # a = record[1:]
            # print("else:{}".format(a))
            # if a.any()>1:
            #     record[0]=0

            label=record.argmax()
            # print ("{},{} label={}".format(i,j,label))
            vote_mask[i,j]=label
    # vote_mask[vote_mask==125]=1
    # vote_mask[vote_mask == 255] = 2
    print(np.unique(vote_mask))

    return vote_mask



if __name__=='__main__':
    x,y = check_input_file(input_path, input_masks)

    final_mask = vote_per_image(x,y,input_path, input_masks)
    plt.imshow(final_mask)
    plt.show()

    cv2.imwrite(output_file, final_mask)