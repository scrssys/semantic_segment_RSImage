import os
import sys
import gdal
gdal.UseExceptions()
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import medfilt, medfilt2d
from skimage import exposure
import cv2
from tqdm import tqdm
from ulitities.base_functions import *
# from error_code import *
from PIL import Image


def resample_data(img, dst_h, dst_w, mode = Image.ANTIALIAS, bits=8):
    if len(img.shape)>2:
        if bits==8:
            n_img = np.zeros((dst_h, dst_w, img.shape[-1]), np.uint8)
            img = np.asarray(img, np.uint8)
            for i in range(img.shape[-1]):
                b_img = img[:, :, i]
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
                # b_img = np.asarray(b_img, np.uint8)
                b_img = Image.fromarray(b_img, mode='L')
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
                b_img = b_img.resize((dst_h, dst_w), mode)
                b_img = np.array(b_img, np.uint8)
                n_img[:, :, i] = b_img[:, :]
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
            return n_img
        else:
            n_img = np.zeros((dst_h, dst_w, img.shape[-1]), np.uint16)
            img = np.asarray(img, np.uint32)
            for i in range(img.shape[-1]):
                b_img = img[:, :, i]
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
                # b_img = np.asarray(b_img, np.uint)
                b_img = Image.fromarray(b_img, mode='I')
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
                b_img = b_img.resize((dst_h, dst_w), mode)
                b_img = np.array(b_img, np.uint16)
                n_img[:, :, i] = b_img[:, :]
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
            return n_img
    else:
        img = Image.fromarray(img, mode='L')
        img = img.resize((dst_h, dst_w), mode)
        img = np.array(img, np.uint8)
        return img



class SampleGenerate():
    def __init__(self, input_dict ={}):
        self.input_dict = input_dict

    def rotate(self, xb, yb, angle):
        xb = np.transpose(xb, (1, 2, 0))
        xb = np.rot90(np.array(xb), k=angle)
        xb = np.transpose(xb, ((2, 0, 1)))

        yb = np.rot90(np.array(yb), k=angle)

        return xb, yb

    def add_noise(self, xb, width, height, dtype=1):
        if dtype == 1:
            noise_value = 255
        elif dtype == 2:
            noise_value = 1024
        else:
            noise_value = 65535

        tmp = np.random.random() / 20.0  # max = 0.05
        noise_num = int(tmp * width * height)
        for i in range(noise_num):
            temp_x = np.random.randint(0, xb.shape[1])
            temp_y = np.random.randint(0, xb.shape[2])
            xb[:, temp_x, temp_y] = noise_value
        return xb


    def gamma_tansform(self, xb, g=2.0):
        tmp = np.random.random() * g
        # print("gamma:{}".format(tmp))
        if tmp < 0.6:
            tmp = 0.6
        if tmp > 1.4:
            tmp = 1.4
        a, b, c = xb.shape
        if a > c:
            xb = xb.transpose(2, 0, 1)
        xb = exposure.adjust_gamma(xb, tmp)
        if a < c:
            xb = xb.transpose(1, 2, 0)
        return xb

    def med_filtering(self, xb, w=3):
        xb = xb.astype(np.float32)
        a, b, c = xb.shape
        if a < c:
            xb = xb.transpose(1, 2, 0)
        _, _, bands = xb.shape

        for i in range(bands):
            xb[:, :, i] = medfilt2d(xb[:, :, i], (w, w))
        if a < c:
            xb = np.transpose(xb, (2, 0, 1))
        xb = xb.astype(np.uint16)
        return xb

    def data_augment(self, xb, yb, w, h, d_type=1):
        if np.random.random() < 0.25:
            assert (yb.shape[0] == yb.shape[1])
            assert (xb.shape[1] == xb.shape[2])
            xb, yb = self.rotate(xb, yb, 1)
        if np.random.random() < 0.25:
            xb, yb = self.rotate(xb, yb, 2)
        if np.random.random() < 0.25:
            assert (yb.shape[0] == yb.shape[1])
            assert (xb.shape[1] == xb.shape[2])
            xb, yb = self.rotate(xb, yb, 3)
        if np.random.random() < 0.25:
            xb = np.transpose(xb, (1, 2, 0))
            xb = np.fliplr(xb)  # flip an array horizontally
            xb = np.transpose(xb, (2, 0, 1))
            yb = np.fliplr(yb)
        if np.random.random() < 0.25:
            xb = np.transpose(xb, (1, 2, 0))
            xb = np.flipud(xb)  # flip an array vertically (up down directory)
            xb = np.transpose(xb, (2, 0, 1))
            yb = np.flipud(yb)

        if np.random.random() < 0.25:  # gamma adjust
            tmp = np.random.random() * 2
            xb = self.gamma_tansform(xb,tmp)
            # xb = exposure.adjust_gamma(xb, tmp)

        if np.random.random() < 0.25:  # medium filtering
            xb = self.med_filtering(xb,3)
            '''
            xb = xb.astype(np.float32)
            xb = np.transpose(xb, (1, 2, 0))
            _, _, bands = xb.shape
            for i in range(bands):
                xb[:, :, i] = medfilt2d(xb[:, :, i], (3, 3))
            xb = np.transpose(xb, (2, 0, 1))
            xb = xb.astype(np.uint16)
            '''


        if np.random.random() < 0.2:
            xb = self.add_noise(xb, w, h, d_type)

        return xb, yb

    def produce_training_samples_binary(self):
        print('\ncreating dataset...')
        in_path = self.input_dict['input_dir']
        out_path = self.input_dict['output_dir']
        valid_labels = list(range(int(self.input_dict['min']), int(self.input_dict['max']+1)))
        target_label = int(self.input_dict['target_label'])

        label_files, tt = get_file(os.path.join(in_path, 'label/'))
        assert (tt != 0)

        image_num = int(self.input_dict['sample_num'])

        image_each = image_num / len(label_files)
        img_w = int(self.input_dict['window_size'])
        img_h = int(self.input_dict['window_size'])

        print("\n[INFO] produce samples---------------------")
        g_count = 0
        for label_file in tqdm(label_files):

            src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
            if not os.path.isfile(src_file):
                print("Have no file:".format(src_file))
                continue

            print("src file:{}".format(os.path.split(src_file)[1]))
            label_img = load_img_by_gdal(label_file, grayscale=True)
            # print("label_img: {}".format(np.unique(label_img)))
            label_img = label_img.astype(np.uint8)
            y, x = label_img.shape
            # print("label_img: {}".format(np.unique(label_img)))


            dataset = gdal.Open(src_file)
            if dataset == None:
                print("open failed!\n")
                continue

            Y_height = dataset.RasterYSize
            X_width = dataset.RasterXSize
            if (X_width != x and Y_height != y):
                print("label and source image have different size:".format(label_file))
                continue

            im_bands = dataset.RasterCount
            data_type = dataset.GetRasterBand(1).DataType

            src_img = dataset.ReadAsArray(0, 0, X_width, Y_height)
            src_img = np.array(src_img)

            del dataset

            index = np.where(label_img == target_label)
            all_label = np.zeros((Y_height, X_width), np.uint8)
            all_label[index] = 1

            print(np.unique(all_label))
            # if no pixel in target value, ignore this label file
            tp = np.unique(all_label)
            # if tp[0]==0:
            #     print("no target value in {}".format(label_file))
            #     continue
            #
            if len(tp) < 2:
                print("Only one value {} in {}".format(tp, label_file))
                if tp[0] == 0:
                    print("no target value in {}".format(label_file))
                    continue

            count = 0
            while count < image_each:
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, Y_height - img_h - 1)
                src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = all_label[random_height: random_height + img_h, random_width: random_width + img_w]

                """ignore nodata area"""
                FLAG_HAS_NODATA = False
                tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
                for tt in tmp:
                    if tt not in valid_labels:
                        FLAG_HAS_NODATA = True
                        continue

                if FLAG_HAS_NODATA == True:
                    continue

                """ignore pure background area"""
                if len(np.unique(label_roi)) < 2:
                    if 0 in np.unique(label_roi):
                        continue

                if 'augment' in self.input_dict['mode']:
                    src_roi, label_roi = self.data_augment(src_roi, label_roi, img_w, img_h, data_type)

                visualize = label_roi * 50

                cv2.imwrite((out_path + '/visualize/%d.png' % g_count), visualize)
                cv2.imwrite((out_path + '/label/%d.png' % g_count), label_roi)

                src_sample_file = out_path + '/src/%d.png' % g_count
                driver = gdal.GetDriverByName("GTiff")
                # driver = gdal.GetDriverByName("PNG")
                # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
                outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
                if outdataset == None:
                    print("create dataset failed!\n")
                    sys.exit(-2)
                if im_bands == 1:
                    outdataset.GetRasterBand(1).WriteArray(src_roi)
                else:
                    for i in range(im_bands):
                        outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
                del outdataset

                count += 1
                g_count += 1

    def produce_training_samples_multiclass(self):
        print('\ncreating dataset...')
        in_path = self.input_dict['input_dir']
        out_path = self.input_dict['output_dir']
        valid_labels = list(range(int(self.input_dict['min']), int(self.input_dict['max'] + 1)))
        target_label = int(self.input_dict['target_label'])

        label_files, tt = get_file(os.path.join(in_path, 'label/'))
        assert (tt != 0)

        image_num = int(self.input_dict['sample_num'])

        image_each = image_num / len(label_files)
        img_w = int(self.input_dict['window_size'])
        img_h = int(self.input_dict['window_size'])

        g_count = 0
        for label_file in tqdm(label_files):

            src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
            if not os.path.isfile(src_file):
                print("Have no file:".format(src_file))
                continue
                # sys.exit(-1)

            print("src file:{}".format(os.path.split(src_file)[1]))

            label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            absname = os.path.split(label_file)[1]
            absname = absname.split('.')[0]

            dataset = gdal.Open(src_file)
            if dataset == None:
                print("open failed!\n")
                continue

            X_height = dataset.RasterYSize
            X_width = dataset.RasterXSize
            im_bands = dataset.RasterCount
            data_type = dataset.GetRasterBand(1).DataType

            # check size of label and src images
            x, y = label_img.shape
            print("Heigh, width of label is :{}, {}".format(x, y))
            print("Heigh, width of src is :{}, {}".format(X_height, X_width))
            if x != X_height or y != X_width:
                print("Warning: src and label have different size!")
                continue

            src_img = dataset.ReadAsArray(0, 0, X_width, X_height)
            src_img = np.array(src_img)

            del dataset

            count = 0
            while count < image_each:
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, X_height - img_h - 1)
                src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

                """ignore nodata area"""
                FLAG_HAS_NODATA = False
                tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
                for tt in tmp:
                    if tt not in valid_labels:
                        FLAG_HAS_NODATA = True
                        continue

                if FLAG_HAS_NODATA == True:
                    continue

                """ignore pure background area"""
                if len(np.unique(label_roi)) < 2:
                    if 0 in np.unique(label_roi):
                        continue
                # print(np.unique(label_roi))

                if 'augment' in self.input_dict['mode']:
                    src_roi, label_roi = self.data_augment(src_roi, label_roi, data_type)

                visualize = label_roi * 50

                cv2.imwrite((out_path + '/visualize/%d_%s.png' % (g_count,absname)), visualize)
                cv2.imwrite((out_path + '/label/%d_%s.png' % (g_count, absname)), label_roi)

                src_sample_file = out_path + '/src/%d_%s.png' % (g_count,absname)
                driver = gdal.GetDriverByName("GTiff")
                outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
                if outdataset == None:
                    print("create dataset failed!\n")
                    sys.exit(-2)
                if im_bands == 1:
                    outdataset.GetRasterBand(1).WriteArray(src_roi)
                else:
                    for i in range(im_bands):
                        outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
                del outdataset

                count += 1
                g_count += 1

    def produce_training_samples_binary_selfAdapt(self):
        print('\ncreating dataset...')
        in_path = self.input_dict['input_dir']
        out_path = self.input_dict['output_dir']
        valid_labels = list(range(int(self.input_dict['min']), int(self.input_dict['max']+1)))
        target_label = int(self.input_dict['target_label'])

        label_files, tt = get_file(os.path.join(in_path, 'label/'))
        assert (tt != 0)

        image_num_rate = float(self.input_dict['sample_scaleRate'])

        # image_each = image_num / len(label_files)
        img_w = int(self.input_dict['window_size'])
        img_h = int(self.input_dict['window_size'])

        print("\n[INFO] produce samples---------------------")
        g_count = 0
        for label_file in tqdm(label_files):
            absname = os.path.split(label_file)[1]
            absname = absname.split('.')[0]
            # src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]

            try:
                label_img = load_img_by_gdal(label_file, grayscale=True)
            except:
                print("Warning: could not open labef image:{}".format(label_file))
                continue
            else:
                print("loaded label image:{}".format(label_file))
            label_img = label_img.astype(np.uint8)

            if "normalize" in self.input_dict["imgmode"]:
                src_dir = os.path.join(in_path, 'norm_src/')
            else:
                src_dir = os.path.join(in_path, 'ori_src/')


            """find file in src directory, suffix may not be the same as that in label directory"""
            try:
                # src_dir = os.path.join(in_path, 'ori_src/')
                tmp_file = find_file(src_dir, absname)
                if tmp_file==None:
                    raise FError
            except FError:
                print("Could not find source file in:".format(os.path.join(in_path, '/src/')))
                continue
            else:
                src_file = tmp_file
                print("find src file:{}".format(src_file))

            try:
                dataset = gdal.Open(src_file)
            except RuntimeError:
                print("Open failed :{}".format(src_file))
            else:
                print("Has opened the image")

            x, y = label_img.shape
            print("Heigh, width of label is :{}, {}".format(x, y))

            Y_height = dataset.RasterYSize
            X_width = dataset.RasterXSize
            print("Heigh, width of src is :{}, {}".format(Y_height, X_width))
            if x != Y_height or y != X_width:
                print("Warning: src and label have different size!")
                continue

            im_bands = dataset.RasterCount
            data_type = dataset.GetRasterBand(1).DataType

            src_img = dataset.ReadAsArray(0, 0, X_width, Y_height)
            src_img = np.array(src_img)

            del dataset

            index = np.where(label_img == target_label)
            all_label = np.zeros((Y_height, X_width), np.uint8)
            all_label[index] = 1

            # print(np.unique(all_label))
            # if no pixel in target value, ignore this label file
            tp = np.unique(all_label)
            # if tp[0]==0:
            #     print("no target value in {}".format(label_file))
            #     continue
            #
            if len(tp) < 2:
                print("Only one value {} in {}".format(tp, label_file))
                if tp[0] == 0:
                    print("no target value in {}".format(label_file))
                    continue

            # Evaluate samples numbers according to the image_size, window_size and sample_scaleRate
            samples_num_of_current_image = int((X_width*Y_height*image_num_rate)/(img_w*img_h)+0.5)
            if 'augment' in self.input_dict['mode']:
                samples_num_of_current_image = 6*samples_num_of_current_image
            print("Extract {} samples from {}".format(samples_num_of_current_image, os.path.split(label_file)[1]))

            count = 0
            while count < samples_num_of_current_image:
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, Y_height - img_h - 1)

                src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = all_label[random_height: random_height + img_h, random_width: random_width + img_w]
                # print(np.unique(label_roi))

                """ignore nodata area"""
                FLAG_HAS_NODATA = False
                tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
                for tt in tmp:
                    if tt not in valid_labels:
                        FLAG_HAS_NODATA = True
                        continue

                if FLAG_HAS_NODATA == True:
                    continue

                """ignore pure background area"""
                if len(np.unique(label_roi)) < 2:
                    if 0 in np.unique(label_roi):
                        continue

                if 'augument' in self.input_dict['mode']:
                    src_roi, label_roi = self.data_augment(src_roi, label_roi, img_w, img_h, data_type)

                # print(np.unique(label_roi))
                visualize = label_roi * 100
                # plt.imshow(visualize)
                # plt.show()

                cv2.imwrite((out_path + '/visualize/%d_%s.png' % (g_count, absname)), visualize)
                cv2.imwrite((out_path + '/label/%d_%s.png' % (g_count, absname)), label_roi)

                src_sample_file = out_path + '/src/%d_%s.png' % (g_count, absname)
                driver = gdal.GetDriverByName("GTiff")
                # driver = gdal.GetDriverByName("PNG")
                # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
                outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
                if outdataset == None:
                    print("create dataset failed!\n")
                    sys.exit(-2)
                if im_bands == 1:
                    outdataset.GetRasterBand(1).WriteArray(src_roi)
                else:
                    for i in range(im_bands):
                        outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
                del outdataset

                count += 1
                g_count += 1
            """validate the target labels"""
            print(np.unique(label_roi))

    def produce_training_samples_multiclass_selfAdapt(self):
        print('\ncreating dataset...')
        in_path = self.input_dict['input_dir']
        out_path = self.input_dict['output_dir']
        valid_labels = list(range(int(self.input_dict['min']), int(self.input_dict['max'] + 1)))
        target_label = int(self.input_dict['target_label'])

        label_files, tt = get_file(os.path.join(in_path, 'label/'))
        assert (tt != 0)

        image_num_rate = float(self.input_dict['sample_scaleRate'])

        # image_each = image_num / len(label_files)
        img_w = int(self.input_dict['window_size'])
        img_h = int(self.input_dict['window_size'])

        g_count = 0
        for label_file in tqdm(label_files):
            absname = os.path.split(label_file)[1]
            absname = absname.split('.')[0]
            # src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]

            try:
                label_img = load_img_by_gdal(label_file, grayscale=True)
            except:
                print("Warning: could not open labef image:{}".format(label_file))
                continue
            else:
                print("loaded label image:{}".format(label_file))
            label_img = label_img.astype(np.uint8)

            """find file in src directory, suffix may not be the same as that in label directory"""
            if "normalize" in self.input_dict["imgmode"]:
                src_dir = os.path.join(in_path, 'norm_src/')
            else:
                src_dir = os.path.join(in_path, 'ori_src/')

            try:
                # src_dir = os.path.join(in_path, 'src/')
                tmp_file = find_file(src_dir, absname)
                if tmp_file==None:
                    raise FError
            except FError:
                print("Could not find source file in:".format(os.path.join(in_path, '/src/')))
                continue
            else:
                src_file = tmp_file
                print("find src file:{}".format(src_file))

            try:
                dataset = gdal.Open(src_file)
            except RuntimeError:
                print("Open failed :{}".format(src_file))
            else:
                print("Has opened the image")

            Y_height = dataset.RasterYSize
            X_width = dataset.RasterXSize
            im_bands = dataset.RasterCount
            data_type = dataset.GetRasterBand(1).DataType

            # check size of label and src images
            x, y = label_img.shape
            print("Heigh, width of label is :{}, {}".format(x, y))
            print("Heigh, width of src is :{}, {}".format(Y_height, X_width))
            if x != Y_height or y != X_width:
                print("Warning: src and label have different size!")
                continue

            src_img = dataset.ReadAsArray(0, 0, X_width, Y_height)
            src_img = np.array(src_img)

            del dataset

            # Evaluate samples numbers according to the image_size, window_size and sample_scaleRate
            samples_num_of_current_image = int((X_width * Y_height * image_num_rate) / (img_w * img_h) + 0.5)
            if 'augument' in self.input_dict['mode']:
                samples_num_of_current_image = 6*samples_num_of_current_image
            print("Extract {} samples from {}".format(samples_num_of_current_image, os.path.split(label_file)[1]))

            count = 0
            while count < samples_num_of_current_image:
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, Y_height - img_h - 1)
                src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

                """ignore nodata area"""
                FLAG_HAS_NODATA = False
                tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
                for tt in tmp:
                    if tt not in valid_labels:
                        FLAG_HAS_NODATA = True
                        continue

                if FLAG_HAS_NODATA == True:
                    continue

                """ignore pure background area"""
                if len(np.unique(label_roi)) < 2:
                    if 0 in np.unique(label_roi):
                        continue
                # print(np.unique(label_roi))

                if 'augment' in self.input_dict['mode']:
                    src_roi, label_roi = self.data_augment(src_roi, label_roi, data_type)

                visualize = label_roi * 50

                cv2.imwrite((out_path + '/visualize/%d_%s.png' % (g_count,absname)), visualize)
                cv2.imwrite((out_path + '/label/%d_%s.png' % (g_count, absname)), label_roi)

                src_sample_file = out_path + '/src/%d_%s.png' % (g_count,absname)
                driver = gdal.GetDriverByName("GTiff")
                outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
                if outdataset == None:
                    print("create dataset failed!\n")
                    sys.exit(-2)
                if im_bands == 1:
                    outdataset.GetRasterBand(1).WriteArray(src_roi)
                else:
                    for i in range(im_bands):
                        outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
                del outdataset

                count += 1
                g_count += 1