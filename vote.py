import numpy as np
import cv2
from tqdm import tqdm

from ulitities.base_functions import load_img

input_path = '../data/predict/'
input_masks=['unet/unet_combined_3.png','segnet/segnet_combined_3.png']
output_file = '../data/predict/final_result_33.png'


def check_input_file(path, masks):
    ret, img_1 = load_img(path+masks[0],grayscale=True)
    assert (ret == 0)

    height, width = img_1.shape
    num_img = len(masks)

    for next_index in range(1,num_img):
        next_ret, next_img=load_img(path+masks[next_index],grayscale=True)
        assert (next_ret ==0 )
        next_height, next_width = next_img.shape
        assert(height==next_height and width==next_width)
    return height, width

# each mask has 5 classes: 0~4

def vote_per_image(height, width, path, masks):

    mask_list = []
    for tt in range(len(masks)):
        ret, img = load_img(path+masks[tt],grayscale=True)
        assert(ret ==0)
        mask_list.append(img)

    vote_mask=np.zeros((height,width), np.uint8)

    for i in tqdm(range(height)):
        for j in range(width):
            record=np.zeros(256,np.uint8)
            for n in range(len(mask_list)):
                mask=mask_list[n]
                pixel=mask[i,j]
                record[pixel] +=1

            """Alarming"""
            # if record.argmax()==0: # if argmax of 0 = 125 or 255, not prior considering background(0)
            #     record[0] -=1

            label=record.argmax()
            # print ("{},{} label={}".format(i,j,label))
            vote_mask[i,j]=label

    return vote_mask



if __name__=='__main__':
    x,y = check_input_file(input_path, input_masks)

    final_mask = vote_per_image(x,y,input_path, input_masks)

    cv2.imwrite(output_file, final_mask)