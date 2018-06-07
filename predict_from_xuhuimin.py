import mxnet as mx
import numpy as np
import cv2

from test_utils.predict import predict


def pred(image, net, step, ctx):#step为样本选取间隔
    h, w, channel = image.shape
    image = image.astype('float32')
    size = int(step *0.75) #取样本中size尺寸为最终预测尺寸
    margin = int((step - size) / 2)
    inhang = int(np.ceil(h/size))
    inlie = int(np.ceil(w / size))

    # newimage0=np.zeros((inhang*size, inlie*size,channel))
    # borderType = cv2.BORDER_REFLECT
    # newimage = cv2.copyMakeBorder(newimage0, margin, margin, margin, margin, borderType)

    newimage = np.zeros((inhang*size + margin*2 , inlie*size +2*margin,channel))
    newimage[margin : h + margin,margin : w + margin ,:] = image
    newimage /= 255
    predictions = np.zeros((inhang*size , inlie*size), dtype=np.int64)
    for i in range(inhang):
        for j in range(inlie):
            patch = newimage[ i*size: i*size+step ,j*size: j*size+step  ,:]
            patch = np.transpose(patch, axes=(2, 0, 1)).astype(np.float32)
            patch = mx.nd.array(np.expand_dims(patch, 0), ctx=ctx)
            pred = predict(patch, net)#预测
            predictions[ i*size: (i+1)*size ,j*size: (j+1)*size] = pred[margin:size+margin,margin:size+margin]
    result = predictions[:h,:w]
    return result

