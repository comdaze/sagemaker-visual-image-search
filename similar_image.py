import cv2
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import argparse
import mxnet as mx
from mxnet import nd, image
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import ImageNet1kAttr
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model
from mxnet.gluon import nn
from scipy.spatial import distance

 
def compare_img_default(img1, img2):
    """
    Strictly compare whether two pictures are equal
        Attention: Even just a little tiny bit different (like 1px dot), will return false.
 ​
    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: true for equal or false for not equal
    """
    difference = cv2.subtract(img1, img2)
    result = not np.any(difference)
 
    return result


def compare_img_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram(直方图)
        Attention: this is a comparision of similarity, using histogram to calculate

        For example:
        1. img1 and img2 are both 720P .PNG file,
           and if compare with img1, img2 only add a black dot(about 9*9px),
           the result will be 0.999999999953

    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)

    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)

    similarity = cv2.compareHist(img1_hist, img2_hist, 0)

    return similarity


def compare_img_p_hash(img1, img2):
    """
    Get the similarity of two pictures via pHash
        Generally, when:
           ham_dist == 0 -> particularly like
           ham_dist < 5  -> very like
           ham_dist > 10 -> different image

        Attention: this is not accurate compare_img_hist() method, so use hist() method to auxiliary comparision.
           This method is always used for graphical search applications, such as Google Image(Use photo to search photo)

    :param img1:
    :param img2:
    :return:
    """
    hash_img1 = get_img_p_hash(img1)
    hash_img2 = get_img_p_hash(img2)

    return ham_dist(hash_img1, hash_img2)


def get_img_p_hash(img):
    """
    Get the pHash value of the image, pHash : Perceptual hash algorithm(感知哈希算法)
 ​
    :param img: img in MAT format(img = cv2.imread(image))
    :return: pHash value
    """
    hash_len = 32

    # GET Gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize image, use the different way to get the best result
    resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_AREA)
    # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_LANCZOS4)
    # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_LINEAR)
    # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_NEAREST)
    # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_CUBIC)

    # Change the int of image to float, for better DCT
    h, w = resize_gray_img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = resize_gray_img

    # DCT: Discrete cosine transform(离散余弦变换)
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(hash_len, hash_len)
    img_list = vis1.flatten()

    # Calculate the avg value
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = []
    for i in img_list:
        if i < avg:
           tmp = '0'
        else:
           tmp = '1'
        avg_list.append(tmp)

    # Calculate the hash value
    p_hash_str = ''
    for x in range(0, hash_len * hash_len, 4):
        p_hash_str += '%x' % int(''.join(avg_list[x:x + 4]), 2)
    return p_hash_str


def ham_dist(x, y):
    """
    Get the hamming distance of two values.
        hamming distance(汉明距)
    :param x:
    :param y:
    :return: the hamming distance
    """
    assert len(x) == len(y)
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])


def compare_img_sift(filename1, filename2):
    start = time.time()
    # queryImage
    img1 = cv2.imread(filename1, 0)
    # trainImage
    img2 = cv2.imread(filename2, 0) 

    # Initiate SIFT detector
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2,outImg=None)
    plt.imshow(img3),plt.show()
    
    score = 1.0 - len(good)/500.0
    
    end = time.time()
    # print('time:', end-start)
    
    return score


def init_model():
    parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
    parser.add_argument('--model', type=str, default='ResNet50_v2',
                        help='name of the model to use')
    parser.add_argument('--saved-params', type=str, default='endpoint/model/model-0000.params',
                        help='path to the saved model parameters')  # 'model/model-0000.params'
    # parser.add_argument('--input-pic', type=str, required=True,
    #                     help='path to the input picture')
    opt = parser.parse_args()

    num_gpus = 0
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    # print(ctx)

    # Load Model
    model_name = opt.model
    pretrained = True if opt.saved_params == '' else False

    if not pretrained:
        classes = [i for i in range(5)]
        net = get_model(model_name, classes=len(classes), pretrained=pretrained)
        net.load_parameters(opt.saved_params)
        
    else:
        net = get_model(model_name, pretrained=pretrained)
        classes = net.classes
        
    net.collect_params().reset_ctx(ctx)

    # print(len(net.features))
    seq_net = nn.Sequential()
    for i in range(len(net.features)):
        seq_net.add(net.features[i])

    return seq_net, ctx


def get_embedding_advance(input_pic):
    # Load Images
    img = image.imread(input_pic)

    # Transform
    img = transform_eval(img).copyto(ctx[0])
    
    pred = None
    use_layers = [len(seq_net)-1]  # [i for i in range(len(seq_net))]
    for i in range(len(seq_net)):
        img = seq_net[i](img)
        if i in use_layers:
#             print(img.shape)
            pred = img[0]

    return pred.asnumpy()


def compare_img_image_embedding(filename1, filename2):
    hash1 = get_embedding_advance(filename1)
    hash2 = get_embedding_advance(filename2)
    n1 = distance.cosine(hash1, hash2)
    return n1


filename1 = '1,12,0,4,22432,3005,2000,8a28288b.jpg'
filename2 = '1,12,0,60,22846,3007,2000,ef9addd4.jpg'
img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)

result = compare_img_default(img1, img2)
print('compare_img_default:', result)

result = compare_img_hist(img1, img2)
print('compare_img_hist:', result)

result = compare_img_p_hash(img1, img2)
print('compare_img_p_hash:', result)

result = compare_img_sift(filename1, filename2)
print('compare_img_sift:', result)

seq_net, ctx = init_model()
result = compare_img_image_embedding(filename1, filename2)
print('compare_img_image_embedding:', result)