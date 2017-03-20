import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

class caffeNet:
	_net = None
	def __init__(self):
		self._net = caffe.Net('caffeSource/models/bvlc_googlenet/deploy.prototxt',
	                'caffeSource/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
	                caffe.TEST)
		print "original net input shape:"
		print self._net.blobs['data'].data[...].shape

		#reshape netowork input
		self._net.blobs['data'].reshape(1,3,224,224)
		print "reshapesd for batchsize of 1:"
		print self._net.blobs['data'].data[...].shape

	def runImage(self, img):
		self._net.blobs['data'].data[...] = img
		out = self._net.forward()
		return self._net.blobs['prob']

	@staticmethod
	def createImage(imgPath):
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

	@staticmethod
	def printOutput(out):
		#print top catagory results with data/ilsvrc12/synset_words.txt
		labels = np.loadtxt("words.txt", str, delimiter='\t')
		top_k = out.data[0].flatten().argsort()[-1:-6:-1]
		print labels[top_k]

	

def main():
	net = caffeNet()
	imgImput = caffeNet.createImage("truck.jpg")
	out = net.runImage(imgImput)
	print "OUTPUT catagories:"
	caffeNet.printOutput(out)
	
main()