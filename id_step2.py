# -*- coding: utf-8 -*-
# import the necessary packages
# import the necessary images
# import the necessary packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
debug = 1

imgPath="test1123.jpg"
imgPath2="idtest.jpg"
img2 = cv2.imread(imgPath2, cv2.IMREAD_COLOR)
img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
#cv2.imshow('image', img)
#print("origion")
cv2.waitKey(0)
def grayImg(img):
    # 转化为灰度图
    print("grayimg")
    gray = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #otsu二值化操作
    retval, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return gray

def preprocess(gray):
    #二值化操作，但与前面grayimg二值化操作中不一样的是要膨胀选定区域所以是反向二值化
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    #膨胀操作
    dilation = cv2.dilate(binary, ele, iterations=1)
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.waitKey(0)
    return dilation


def findTextRegion(img):
    region = []
    # 1. 查找轮廓 

    #image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 300):
            continue
        # 轮廓近似，作用很小
    #   epsilon = 0.001 * cv2.arcLength(cnt, True)
     #   approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        #函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
        if debug:
            print("rect is: ", rect)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2):
            continue
        # 太扁的也不要
        if (height * 18 < width):
            continue
        if (width > img.shape[1] / 2 and height > img.shape[0] / 20):
            region.append(box)
    return region

def detect(img):
    # fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )
    gray = cv2.fastNlMeansDenoisingColored(img, None, 10, 3, 3, 3)
    cv2.imshow("_dimg1", gray)
    cv2.waitKey(0)
    #cv2.fastNlMeansDenoisingColored作用为去噪
    coefficients = [0, 1, 1]
    m = np.array(coefficients).reshape((1, 3))
    gray = cv2.transform(gray, m)
    cv2.imshow("_dimg2", gray)
    #cv2.waitKey(0)
    if debug:
        cv2.imwrite("dgray.png", gray)
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)
    
    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        h = abs(box[0][1] - box[2][1])
        w = abs(box[0][0] - box[2][0])
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        y1 = min(Ys)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        # 使用 Tesseract 进行 OCR（光学字符识别）
        if w > 0 and h > 0 and x1 < gray.shape[1] / 2:
            idImg = grayImg(img[y1:y1 + h, x1:x1 + w])
            cv2.imwrite("idImg.png", idImg)
            cv2.imwrite("contours.png", img)
            id_number_region_blurred = cv2.blur(img[y1:y1+h, x1:x1+w],(15,15))
            # 将模糊处理后的区域放回原图
            # 调整模糊后的区域的大小，使其与身份证号区域相同
            id_number_region_blurred_resized = cv2.resize(id_number_region_blurred, (w, h))

        # 将模糊处理后的区域放回原图
            img[y1:y1+h, x1:x1+w] = id_number_region_blurred_resized
            #cv2.rectangle(mask, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), thickness=cv2.FILLED)
            # 调整 mask 的大小以匹配 idImg
            #mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            # 进行按位与运算
            #result = cv2.bitwise_and(idImg, mask)
            # 应用遮蔽
            #result = cv2.bitwise_and(idImg, mask)
            return img
            #只顯示身分證號碼

imgy = cv2.imread(imgPath2, cv2.IMREAD_COLOR)
imgy = cv2.resize(img2, (224, 135), interpolation=cv2.INTER_CUBIC)
idImg2 = detect(imgy)
print("Image Shape:", idImg2.shape)

#image = Image.fromarray(idImg)
image_np = np.array(idImg2)
cv2.imshow("_imgyy", idImg2)

cv2.waitKey(0)

imgx = cv2.imread(imgPath, cv2.IMREAD_COLOR)
imgx = cv2.resize(img, (224, 135), interpolation=cv2.INTER_CUBIC)
idImg = detect(imgx)
print("Image Shape:", idImg.shape)

#image = Image.fromarray(idImg)
image_np = np.array(idImg)
cv2.imshow("_imgxx", idImg)

cv2.waitKey(0)

#detect("tree.jpg")










