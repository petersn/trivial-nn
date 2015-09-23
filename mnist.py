#!/usr/bin/python

import numpy, struct, bz2

class Instance:
	def __init__(self, image, label):
		self.image, self.label = image, label

def get(pattern, f):
	v = struct.unpack(pattern, f.read(struct.calcsize(pattern)))
	return v

def read_images(image_path, label_path, flatten=True, count=None):
	instances = []
	with bz2.BZ2File(image_path) as f_image:
		with bz2.BZ2File(label_path) as f_label:
			assert get(">I", f_image)[0] == 2051
			assert get(">I", f_label)[0] == 2049
			item_count = get(">I", f_image)[0]
			assert get(">I", f_label)[0] == item_count
			width, height = get(">II", f_image)
			#print "Image size: %ix%i" % (width, height)
			for _ in xrange(count or item_count):
				image = numpy.array(map(float, get("%iB" % (width*height), f_image)))
				image /= 255.0
				#image = numpy.array([get("%iB" % width, f_image) for _ in xrange(height)])
				label, = get("B", f_label)
				instances.append(Instance(image, label))
			# Make sure we read the whole file, if not reading a truncated set of images.
			if count == None:
				assert f_image.read(1) == ""
				assert f_label.read(1) == ""
	return instances

def make_class_even(instances):
	import random
	labels = [i.label for i in instances]
	classes = set(labels)
	occurances = {label: labels.count(label) for label in classes}
	count = min(occurances.values())
	samples = [random.sample([i for i in instances if i.label == label], count) for label in classes]
	return reduce(lambda x, y: x+y, samples)

train = read_images("data/train-images-idx3-ubyte.bz2", "data/train-labels-idx1-ubyte.bz2")
test = read_images("data/t10k-images-idx3-ubyte.bz2", "data/t10k-labels-idx1-ubyte.bz2")

