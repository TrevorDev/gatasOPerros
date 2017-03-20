import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

def createInputFromImage(imgPath):
	#load image, resize
	im = Image.open(imgPath).convert("RGB")
	im = im.resize((224, 224), Image.ANTIALIAS)

	#convert to grayscale
	#im = im.convert("L")

	ar = np.array(im)
	print "ready image shape:"
	print ar.shape

	#separate each channel to b, g, r
	res = np.array([ar[:,:,0],ar[:,:,1],ar[:,:,2]])
	print "separated channel shape:"
	print res.shape

	reshaped = res[np.newaxis, :, :]
	print "ready data shape:"
	print reshaped.shape
	return reshaped

def createNet():
	#create network from google's pretrained imagenet model
	net = caffe.Net('caffeSource/models/bvlc_googlenet/deploy.prototxt',
	                'caffeSource/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
	                caffe.TEST)

	print "original net input shape:"
	print net.blobs['data'].data[...].shape

	#reshape netowork input
	net.blobs['data'].reshape(1,3,224,224)
	print "reshapesd for batchsize of 1:"
	print net.blobs['data'].data[...].shape
	return net

def main():
	net = createNet()
	net.blobs['data'].data[...] = createInputFromImage('images.jpg')
	out = net.forward()

	#print top catagory results
	labels = np.loadtxt("words.txt", str, delimiter='\t')
	top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
	print labels[top_k]


main()