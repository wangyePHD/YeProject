import cv2
import numpy as np
import os

data_dir = './data'


def work(img_name):
    # 数据格式：
    # RGB: x.jpg
    # Mask: x_bg.png
    # HF-MAP: x_hf.jpg

    img = cv2.imread(f'{data_dir}/{img_name}.jpg')
    mask = cv2.imread(f'{data_dir}/{img_name}_bg.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    zero = np.zeros(img.shape[0:3], dtype="uint8")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # 计算x方向和y方向的导数
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    # 将x方向和y方向的导数转换为绝对值
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 合并x方向和y方向的导数
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # grad = cv2.multiply(grad, img)
    result = cv2.add(grad, zero, mask=mask)

    # 显示结果
    # cv2.imshow('image', img)
    # cv2.imshow('grad', grad)
    # cv2.imshow('result', result)

    cv2.imwrite(f'./result/{img_name}_hf.jpg', result)


names = os.listdir(data_dir)
names = [x.split('.')[0] for x in names if x.find('_') == -1]

for x in names:
    work(x)
