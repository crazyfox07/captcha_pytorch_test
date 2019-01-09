# @Time    : 2019/1/9 8:05
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : gen_captcha.py
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import cv2
import os
import string
import random
import time

number = string.digits
alphabet = string.ascii_lowercase
ALPHABET = string.ascii_uppercase

char_set = number
char_set_len = len(char_set)
captcha_size = 4
img_w = 160
img_h = 60
img_c = 3

train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'train')
test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'test')
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)


def random_captcha_text(char_set=char_set, captcha_size=captcha_size):  # 可以设置只用来生成数字
    captcha_text = random.sample(char_set, captcha_size)
    return ''.join(captcha_text)


def gen_captcha_text_and_image(img_path, batch_size=128):
    # 生成图片
    for _ in range(batch_size):
        image = ImageCaptcha()
        captcha_text = random_captcha_text()  # 生成标签
        captcha = image.generate(captcha_text)

        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)

        img_name = captcha_text + '_' + str(int(time.time())) + '.jpg'
        cv2.imwrite(os.path.join(img_path, img_name), captcha_image)  # 保存


def img_label_one_hot(img_label):
    y = np.zeros(char_set_len * captcha_size)
    for i, ch in enumerate(img_label):
        ch_index = char_set.index(ch)
        y[i * char_set_len + ch_index] = 1
    return y


def remove_captcha(img_path):
    imgs = os.listdir(img_path)
    for img in imgs:
        os.remove(os.path.join(img_path, img))


def read_captcha_text_and_image(img_path, batch_size=64):
    remove_captcha(img_path)
    gen_captcha_text_and_image(img_path, batch_size=batch_size)
    x = np.zeros(shape=(batch_size, img_h, img_w, img_c))
    y = np.zeros(shape=(batch_size, char_set_len * captcha_size))
    imgs = os.listdir(img_path)
    random.shuffle(imgs)
    for i, img in enumerate(imgs):
        img_array = np.array(Image.open(os.path.join(img_path, img)))
        img_array = img_array / 255
        x[i:i + 1] = img_array
        img_label = img_label_one_hot(img.split('_')[0])
        y[i:i + 1] = img_label
    x = np.transpose(x, axes=(0, 3, 1, 2))
    return x, y


if __name__ == '__main__':
    # gen_captcha_text_and_image(train_path)
    read_captcha_text_and_image(train_path)
    # remove_captcha(train_path)