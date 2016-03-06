# -*— coding:utf-8 -*-
#生成随机验证码类
from PIL import Image, ImageFilter, ImageFont, ImageDraw
import random
class CodeImage(object):
    def __init__(self, width=240, height=60, number=4):
        self.__width = width   #宽度
        self.__height = height  #高度
        self.__number = number  #字符个数
        font = ImageFont.truetype("arial.ttf", 36)
    #保存图像
    def save(self, name, form):
        self.createImage()
        self.createDrawer()
        self.fillPoint()
        self.fillChar()
        self.image.save(name, form)
    #创建图像
    def createImage(self):
        self.image = Image.new("RGB", (self.__width, self.__height), (255, 255, 255))
    #创建画笔
    def createDrawer(self):
        self.drawer = ImageDraw.Draw(self.image)
    #填充像素点
    def fillPoint(self):
        for x in range(0,self.__width):
            for y in range(self.__height):
                self.drawer.point((x, y),fill=self.randColor(80))
    #输出文字
    def fillChar(self):
        length = self.__width // self.__number
        for n in range(self.__number):
            font = ImageFont.truetype("arial.ttf", random.randint(30,40))
            self.drawer.text((length*n + random.randint(5, 10), random.randint(10, 15)),self.randChar(), font=font, fill=self.randColor(32, 127))
    #返回随机字符
    def randChar(self):
        codeStr = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return codeStr[random.randint(0,len(codeStr)-1)]
    #返回随机颜色，用于
    def randColor(self, low=64, hight=255):
        return (random.randint(low, hight),random.randint(low, hight), random.randint(low, hight))
if __name__ == "__main__":
    codeimage = CodeImage()
    codeimage.save("code.jpg", 'jpeg')
