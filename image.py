#! env python
# coding: utf-8
import os
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from texts import texts


class CaptchaImage:
    def __init__(self, captcha_image):
        """
        :param captcha_image: 验证码图片路径
        """
        self.captcha = cv2.imread(captcha_image)
        self.text = self._get_text()
        self.images = self._get_images()

    def _get_text(self, offset=0):
        """
        识别文字内容
        :param offset: 文字位置偏移量
        """
        model = models.load_model('models/model.v2.0.h5')

        # 预处理
        text = self.captcha[3: 22, offset + 120: offset + 177]
        text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
        text = text / 255.0
        h, w = text.shape
        text.shape = (1, h, w, 1)

        if offset == 0:
            label = model.predict(text)
            label = label.argmax()
            text = texts[label]

            # 根据第一个词语的字数确定第二个词语的位置偏移量
            offset = [27, 47, 60][len(text) - 1]
            text2 = self._get_text(offset)
            if text2:
                return [text, text2]
            else:
                return [text]

        else:
            # 如果不是全白，则第二个词语存在
            if text.mean() < 0.95:
                label = model.predict(text)
                label = label.argmax()
                return texts[label]
            else:
                return

    def _get_images(self):
        """
        识别图片内容
        """
        model = models.load_model('models/12306.image.model.h5')
        images = self._cut_images()

        # 预处理
        images = np.array(images)
        images = images.astype('float32')
        mean = [103.939, 116.779, 123.68]
        images -= mean

        labels = model.predict(images)
        labels = labels.argmax(axis=1)
        images = [texts[i] for i in labels]
        return [images[:4], images[4:]]

    def _cut_images(self):
        """
        切割八张图片
        """
        height, width, _ = self.captcha.shape
        gap = 5
        unit = 67
        images = []
        for x in range(40, height - unit, gap + unit):
            for y in range(gap, width - unit, gap + unit):
                images.append(self.captcha[x: x + unit, y: y + unit])
        return images

    def show_text(self):
        """
        显示切割的文字
        """
        text = self.captcha[3: 22, 120: 40 + 177]
        print('按任意键继续')
        cv2.imshow('Text', text)
        cv2.moveWindow('Text', 500, 500)
        cv2.waitKey(0)
        cv2.destroyWindow('Text')

    def show_images(self):
        """
        显示切割的图片
        """
        print('关闭窗口以继续')
        images = self._cut_images()
        for i in range(len(images)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i], interpolation='nearest', aspect='auto')
            plt.title('Image-%d' % (i + 1), fontsize=8)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def show_captcha(self):
        """
        显示验证码原图
        """
        print('按任意键继续')
        cv2.imshow('Captcha', self.captcha)
        cv2.moveWindow('Captcha', 500, 500)
        cv2.waitKey(0)
        cv2.destroyWindow('Captcha')


def download_image():
    """
    下载验证码
    :return:
    """
    url = [
        'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand',
        'https://kyfw.12306.cn/passport/captcha/captcha-image?login_site=E&module=login&rand=sjrand'
    ]
    r = requests.get(url[1])
    with open('imgs/yzm.jpg', 'wb') as fp:
        fp.write(r.content)


def test():
    if not os.path.isdir('imgs'):
        os.mkdir('imgs')
        download_image()

    captcha = CaptchaImage('imgs/yzm2.jpg')
    print(captcha.text)
    print(captcha.images)
    captcha.show_captcha()


if __name__ == '__main__':
    test()
