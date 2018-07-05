import numpy as np
import cv2

from ulitities.base_functions import load_img

# RESULT_PREFIXX = ['./result1/', './result2/', './result3/']

input_path = '../data/predict/'
input_masks=['result_unet_combined_100.png','result_unet_combined125.png',
             'result_segnet_combined.png','result_segnet_combined80.png']
output_file = '../data/predict/final_result.png'


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

    for i in range(height):
        for j in range(width):
            record=np.zeros(256,np.uint8)
            for n in range(len(mask_list)):
                mask=mask_list[n]
                pixel=mask[i,j]
                record[pixel] +=1

            label=record.argmax()
            print ("{},{} label={}".format(i,j,label))
            vote_mask[i,j]=label

    return vote_mask



    # # for j in range(len(RESULT_PREFIXX)):
    # #     im = cv2.imread(RESULT_PREFIXX[j] + str(image_id) + '.png', 0)
    # #     result_list.append(im)
    #
    # # each pixel
    # height, width = result_list[0].shape
    # vote_mask = np.zeros((height, width))
    # for h in range(height):
    #     for w in range(width):
    #         record = np.zeros((1, 5))
    #         for n in range(len(result_list)):
    #             mask = result_list[n]
    #             pixel = mask[h, w]
    #             # print('pix:',pixel)
    #             record[0, pixel] += 1
    #
    #         label = record.argmax()
    #         # print(label)
    #         vote_mask[h, w] = label
    #
    # cv2.imwrite('vote_mask' + str(image_id) + '.png', vote_mask)



if __name__=='__main__':
    x,y = check_input_file(input_path, input_masks)

    final_mask = vote_per_image(x,y,input_path, input_masks)

    cv2.imwrite(output_file, final_mask)