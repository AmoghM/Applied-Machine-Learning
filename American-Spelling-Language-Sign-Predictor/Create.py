import os,cv2
from PIL import Image
from resizeimage import resizeimage
# from scipy.misc import imread, imresize, imshow


X=32
Y=32 
char = 'E1'

datapath = str(X)+'x'+str(Y)+char               

os.mkdir(datapath)

for i in range(97,123):
    if(chr(i)!='j' and chr(i)!='z'):
        for file_name in os.listdir('dataset5/'+char[0]+'/'+chr(i)):
            if(file_name.find('depth') < 0):
                # img = Image.open('dataset5/'+char+'/'+chr(i)+'/'+file_name)
                im_gray = cv2.imread('dataset5/'+char[0]+'/'+chr(i)+'/'+file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                im_back = cv2.imread('dataset5/'+char[0]+'/'+chr(i)+'/'+file_name.split('color')[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
                
                # (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # cv2.imwrite('pic1.png', im_bw)
                # (thresh, im_gray) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                #changing the dimension of the image to 16*8
                # resize and cover as necessarry 
                res =  cv2.bitwise_and(im_gray,im_gray,mask=im_back)

                im_gray = cv2.resize(res,(X,Y))
                # img = resizeimage.resize_cover(img, [X, Y])
                # img = imresize(img,(X,Y,))
                cv2.imwrite(datapath+"/"+file_name, im_gray)

                # img.save(datapath+"/"+file_name, img.format)
                # img.close()
                # print datapath+"/"+file_name
           
