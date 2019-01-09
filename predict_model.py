# @Time    : 2019/1/9 8:10
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : predict_model.py

import torch
import numpy as np
from gen_captcha import read_captcha_text_and_image, test_path, captcha_size, char_set_len, char_set
from tain_model import CNNModel, model_path


# 预测模型
def predict_model():
    cnn_model = CNNModel()
    # 加载模型
    cnn_model.load_state_dict(torch.load(model_path))

    imgs, labels = read_captcha_text_and_image(test_path)
    imgs = torch.FloatTensor(imgs)
    predict_labels = cnn_model(imgs)
    predict_labels = predict_labels.detach().numpy()
    predict_labels = np.reshape(predict_labels, newshape=(predict_labels.shape[0], captcha_size, char_set_len))
    predict_labels = np.argmax(predict_labels, axis=-1)

    labels = np.reshape(labels, newshape=(labels.shape[0], captcha_size, char_set_len))
    labels = np.argmax(labels, axis=-1)
    for i in range(predict_labels.shape[0]):
        y_pred = []
        y_true = []
        for j in range(captcha_size):
            y_pred.append(char_set[predict_labels[i, j]])
            y_true.append(char_set[labels[i, j]])
        print('{}: {}'.format(''.join(y_true), ''.join(y_pred)))


if __name__ == '__main__':
    predict_model()
