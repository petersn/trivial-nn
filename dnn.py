#!/usr/bin/python

import numpy, math, random, copy

class Affine:
	name = "affine"

	def __init__(self, matrix, bias):
		self.matrix, self.bias = matrix, bias
		self.name = "affine(%i->%i)" % (self.matrix.shape[::-1])

	def apply(self, inp):
		self.inp = inp
		return self.bias + self.matrix.dot(inp)

	def backprop(self, gradient, learning_rate):
		self.bias += learning_rate * gradient
		outer = numpy.outer(gradient, self.inp)
		gradient = self.matrix.T.dot(gradient)
		self.matrix += learning_rate * outer
		return gradient

class TanhSigmoid:
	name = "tanh-sigmoid"

	def apply(self, inp):
		self.inp = inp
		return math.tanh(inp)

	def backprop(self, gradient, learning_rate):
		return gradient * (1.0 - numpy.tanh(self.inp)**2)

class Sigmoid:
	name = "sigmoid"

	def apply(self, inp):
		self.inp = inp
		return 1.0 / (1.0 + numpy.exp(-inp))

	def backprop(self, gradient, learning_rate):
		ex = numpy.exp(-self.inp)
		return gradient * (ex / (1.0 + ex)**2.0)

class Softmax:
	name = "softmax"

	def apply(self, inp):
		self.values = numpy.exp(inp)
		self.total = numpy.sum(self.values)
		return self.values / self.total

	def backprop(self, gradient, learning_rate):
		pos = []
		for i in xrange(len(gradient)):
			grad = gradient[i] * (self.total - self.values[i]) * self.values[i] / (self.total**2.0)
			coef = self.values[i] / (self.total**2.0)
			for j in xrange(len(gradient)):
				if j == i:
					continue
				grad += - gradient[j] * self.values[j] * coef
			pos.append(grad)
		return numpy.array(pos)

class Net:
	def __init__(self, layers=None):
		self.layers = layers or []

	def apply(self, inp):
		for layer in self.layers:
			inp = layer.apply(inp)
		return inp

	def backprop(self, gradient, learning_rate):
		for layer in self.layers[::-1]:
			gradient = layer.backprop(gradient, learning_rate)
		return gradient

	def architecture(self):
		return ", ".join(layer.name for layer in self.layers)

	@staticmethod
	def random_network(widths):
		n = Net()
		for w1, w2 in zip(widths[:-1], widths[1:]):
			matrix = numpy.array([[random.normalvariate(0, 0.01) for _ in xrange(w1)] for _ in xrange(w2)])
			bias = numpy.array([random.normalvariate(0, 0.01) for _ in xrange(w2)])
			n.layers.append(Affine(matrix, bias))
			n.layers.append(Sigmoid())
#		n.layers.pop()
#		n.layers.append(Softmax())
		return n

if __name__ == "__main__":
	import mnist, random
	mnist.train = mnist.make_class_even(mnist.train)
	mnist.test = mnist.make_class_even(mnist.test)
	random.shuffle(mnist.train)
	random.shuffle(mnist.test)
	n = Net.random_network([28*28, 40, 10])
	learning_rate = 1.0
	gradients = [numpy.array([float(i == j) for j in xrange(10)]) for i in xrange(10)]
	print "Architecture:", n.architecture()
	print "Training on %i instances, with %i instances for cross-validation." % (len(mnist.train), len(mnist.test))
	print "Using: learning_rate=%r" % learning_rate
	epoch = 1
	old_cv_score = 0.0
	old_model = copy.deepcopy(n)
	def test(n, sample):
		cv_score, correct, choices = 0, 0, [0]*10
		for example in sample:
			classification = n.apply(example.image)
			cv_score += gradients[example.label].dot(classification)
			choice = max(xrange(10), key=classification.__getitem__)
			correct += example.label == choice
			choices[choice] += 1
		return cv_score  / float(len(sample)), correct / float(len(sample)), choices
	while learning_rate > 1e-3:
		random.shuffle(mnist.train)
		# Training epoch.
		for example in mnist.train:
			result = n.apply(example.image)
			n.backprop(gradients[example.label] - result, learning_rate)
		# Cross-validation.
		regular_score, regular_correct, regular_choices = test(n, mnist.train)
		cv_score, correct, choices = test(n, mnist.test)
		print "[%3i] Correct: %7.3f%% CV score: %8.5f -- Correct: %7.3f%% Score: %8.5f -- Choices: %r" % (epoch, 100.0 * correct, cv_score, 100.0 * regular_correct, regular_score, choices)
		if cv_score < old_cv_score:
			n = old_model
			learning_rate *= 0.5
			print "CV reject! Reducing learning rate to: %r" % learning_rate
		epoch += 1
		old_cv_score, old_model = cv_score, copy.deepcopy(n)
	print "Converged."

