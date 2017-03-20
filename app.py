import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

print "started!"

#load image, resize
im = Image.open('cat.jpg').convert("RGB")
im = im.resize((100, 100), Image.ANTIALIAS)

#convert to grayscale
im = im.convert("L")

ar = np.array(im)
print "ready image shape:"
print ar.shape

reshaped = ar[np.newaxis, np.newaxis, :, :]
print "ready data shape:"
print reshaped.shape

print reshaped[0,0]
net = caffe.Net('conv.prototxt', caffe.TEST)

net.blobs['data'].data[...] = reshaped

net.forward()

print net.blobs['conv'].data.shape
print net.blobs['conv'].data[0,0]

newIm = Image.fromarray(net.blobs['conv'].data[0,0])
newIm.show()

# im_ar = np.array(im)
# reshaped = im_ar[np.newaxis, np.newaxis, :, :]
# print reshaped.shape

# 

# im = np.array(Image.open('cat_gray.jpg'))
# im_input = im[np.newaxis, np.newaxis, :, :]
# print im_input.shape
# net.blobs['data'].reshape(*im_input.shape)
# net.blobs['data'].data[...] = im_input

# net.forward()

# print net.blobs['conv']
# net.save('mymodel.caffemodel')
# print "done!"