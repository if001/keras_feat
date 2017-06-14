
from PIL import Image
import numpy as np
import scipy.misc
import pylab as plt

IMG_SIZE = 32

class MyImage():
    def __init__(self) : pass

    def load_img(self,img_name):
        # self.img = np.array(Image.open(img_name).convert('L'))
        self.img = Image.open(img_name).convert('RGB').resize((32,32))

        # self.img = scipy.misc.imresize(self.img,(IMG_SIZE, IMG_SIZE))

        # self.img = np.array(self.img)
        # self.img = self.img.astype(np.float32)/np.max(self.img)

    def black_to_red(self):
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                r,g,b = self.img.getpixel((i,j))
                if ((r < 10) and (g < 10) and (b < 10)):
                    self.img.putpixel((i,j), (255,0,0,0)) # 真っ赤に！
                else:
                    self.img.putpixel((i,j), (r,g,b,0)) # 元の色

    def black_to_white(self):
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                r,g,b = self.img.getpixel((i,j))
                if ((r < 10) and (g < 10) and (b < 10)):
                    self.img.putpixel((i,j), (255,255,255,0))
                else:
                    self.img.putpixel((i,j), (r,g,b,0)) # 元の色

    def red_to_black(self):
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                r,g,b = self.img.getpixel((i,j))
                if r > 220 and g < 10 and b < 10:
                    self.img.putpixel((i,j), (0,0,0,0))
                else:
                    self.img.putpixel((i,j), (r,g,b,0)) # 元の色

    def convert_L(self):
        self.img = self.img.convert('L').resize((32,32))


    def routate_img(self,dig):
        self.img = self.img.rotate(dig)


    def save_img(self,img_name,img_id):
        self.img.save('./'+img_name.split(".")[0]+img_id+".jpg")


    def show_img(self):
        plt.imshow(self.img)
        # plt.gray()
        plt.show()


def set_routate(myimg,dig):
    myimg.black_to_red()
    myimg.routate_img(dig)
    myimg.black_to_white()
    myimg.red_to_black()
    myimg.show_img()


def main():
    myimg = MyImage()

    img_name = "./wo_hiragana.jpg"

    myimg.load_img(img_name)
    set_routate(myimg,30)

    myimg.load_img(img_name)
    set_routate(myimg,-30)

    myimg.load_img(img_name)
    set_routate(myimg,20)

    myimg.load_img(img_name)
    set_routate(myimg,-20)



    # for i in range(2):
    #     myimg.load_img(img_name)
    #     myimg.save_img(img_name,i)


if __name__ == "__main__" :
    main()
