# @Time    : 2019/1/9 8:07
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : tain_model.py
import torch.nn as nn
from torch import optim, save
from torch.autograd import Variable
from gen_captcha import char_set_len, captcha_size, read_captcha_text_and_image, train_path, img_h, img_w
import torch
import os

learning_rate = 0.001
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model', 'model-gpu.pkl')
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))


# 创建cnn模型
class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()

    def forward(self, x):
        # 第一层卷积
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2)(x)  # in:(bs,3,60,160)
        conv1 = nn.BatchNorm2d(32)(conv1)
        conv1 = nn.Dropout(0.5)(conv1)
        conv1 = nn.ReLU()(conv1)
        conv1 = nn.MaxPool2d(kernel_size=2, stride=2)(conv1)  # out:(bs,32,30,80)
        # 第二层卷积
        conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)(conv1)
        conv2 = nn.BatchNorm2d(64)(conv2)
        conv2 = nn.Dropout(0.5)(conv2)
        conv2 = nn.ReLU()(conv2)
        conv2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv2)  # out:(bs,64,15,40)
        # 第三层卷积
        conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)(conv2)
        conv3 = nn.BatchNorm2d(128)(conv3)
        conv3 = nn.Dropout(0.5)(conv3)
        conv3 = nn.ReLU()(conv3)
        conv3 = nn.MaxPool2d(kernel_size=2, stride=2)(conv3)  # out:(bs,128,7,20)
        # reshape conv3
        conv3_reshape = conv3.view(conv3.size(0), -1)
        # 全卷积层
        fc1 = nn.Linear(128 * 7 * 20, 1024)(conv3_reshape)
        output = nn.Linear(1024, char_set_len * captcha_size)(fc1)
        return output


# 创建模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(128 * (img_h // 8) * (img_w // 8), 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, char_set_len * captcha_size),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out


# 训练模型
def train_model():
    cnn_model = CNNModel()
    if os.path.exists(model_path):
        # 加载模型
        cnn_model.load_state_dict(torch.load(model_path))

    # cnn_model.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    i = 0
    while True:
        imgs, labels = read_captcha_text_and_image(train_path, batch_size=32)
        imgs = Variable(torch.FloatTensor(imgs))
        labels = Variable(torch.FloatTensor(labels))
        predict_labels = cnn_model(imgs)
        loss = criterion(predict_labels, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("step:", i, "loss:", loss.item())
        if i % 100 == 0:
            save(cnn_model.state_dict(), model_path)  # current is model.pkl
            print("save model")
        i = i + 1


if __name__ == '__main__':
    train_model()
