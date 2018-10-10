import numpy as np
import random
import mnist_loader
import matplotlib.pyplot as plt
import time
import os

class Network:
	def __init__(self, sizes):
		self.sizes = sizes
		self.n_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
	
	def feedForward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a
	
	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)
	
	def evaluateGetBad(self, test_data):
		test_results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x, y) in test_data]
		ret = []
		for ind, (x, y) in zip(range(len(test_data)), test_results):
			if x != y:
				ret.append((ind, x))
		return ret
	
	def getCost(self, test_data, lmbda):
		sm = 0.0
		for inp, out in test_data:
			a = self.feedForward(inp)
			sm += cost(a, out)
		sm += 0.5 * (lmbda / len(test_data)) * sum(np.linalg.norm(w)**2 \
		                                           for w in self.weights)
		return sm / len(test_data)
	
	def dropoutTake(self):
		# start = time.time()
		self.prev_weights = self.weights
		self.prev_biases = self.biases
		self.prev_sizes = self.sizes
		
		self.sizes = [self.sizes[0]] + [x // 2 for x in self.sizes[1:-1]] + [self.sizes[-1]]
		
		self.chosen = \
			[list(range(self.sizes[0]))] + \
			[random.sample(range(s), s) for s in self.sizes[1:-1]] + \
			[list(range(self.sizes[-1]))]
		
		self.chosen = [list(range(s)) for s in self.sizes]
		
		self.biases = [b[c] / 2.0 for b, c in zip(self.biases, self.chosen[1:])]
		self.weights = [w[cnx][:, cpr] / 2.0 for cpr, cnx, w in zip(self.chosen[:-1], self.chosen[1:], self.weights)]
		# end = time.time()
		# print("take:", end - start)
	
	def dropoutRetake(self):
		# start = time.time()
		for l in range(1, self.n_layers):
			self.biases[l - 1] *= 2.0
			self.weights[l - 1] *= 2.0
			
			self.biases[l - 1] = makeMatrix(self.biases[l - 1], self.prev_sizes[l], 1, \
				self.chosen[l], [0], 0)
			self.weights[l - 1] = makeMatrix(self.weights[l - 1], self.prev_sizes[l], \
				self.prev_sizes[l - 1], self.chosen[l], self.chosen[l - 1], 0)
			
			tm_b = makeMatrix(np.zeros(self.biases[l - 1].shape), self.prev_sizes[l], 1, \
				self.chosen[l], [0], 1)
			tm_w = makeMatrix(np.zeros(self.weights[l - 1].shape), self.prev_sizes[l], \
				self.prev_sizes[l - 1], self.chosen[l], self.chosen[l - 1], 1)
			
			self.prev_biases[l - 1] = self.prev_biases[l - 1] * tm_b + self.biases[l - 1]
			self.prev_weights[l - 1] = self.prev_weights[l - 1] * tm_w + self.weights[l - 1]
		
		self.biases = self.prev_biases
		self.weights = self.prev_weights
		self.sizes = self.prev_sizes
		# end = time.time()
		# print("retake:", end - start)
		
	# def dropoutRetake(self):
	# 	for l in range(1, self.n_layers):
	# 		self.biases[l - 1] *= 2.0
	# 		self.weights[l - 1] *= 2.0
	# 		for nx in range(self.sizes[l]):
	# 			ind_nx = self.chosen[l][nx]
	# 			self.prev_biases[l - 1][ind_nx] = self.biases[l - 1][nx]
				
	# 			for pr in range(self.sizes[l - 1]):
	# 				ind_pr = self.chosen[l - 1][pr]
	# 				self.prev_weights[l - 1][ind_nx][ind_pr] = self.weights[l - 1][nx][pr]
		
	# 	self.biases = self.prev_biases
	# 	self.weights = self.prev_weights
	# 	self.sizes = self.prev_sizes
		
	def SGD(self, tr_data, epochs, mbatch_size, eta, test_data = None,
	        lmbda = 0.0,
	        monitor_tr_cost = False,
	        monitor_tr_accuracy = False,
	        monitor_test_cost = False,
	        monitor_test_accuracy = False,
	        dropout = False):
	
		test_cost, test_accuracy = [], []
		tr_cost, tr_accuracy = [], []
		for ep in range(epochs):
			start = time.time()
			random.shuffle(tr_data)
			
			ins_all = [x[0].transpose()[0] for x in tr_data]
			outs_all = [x[1].transpose()[0] for x in tr_data]
			
			mbatches = [(np.array(ins_all[k:k + mbatch_size]).transpose(), \
			             np.array(outs_all[k:k + mbatch_size]).transpose()) \
			             for k in range(0, len(tr_data), mbatch_size)]
			
			for mbatch in mbatches:
				if dropout: self.dropoutTake()
				self.updateMbatch(mbatch, eta, lmbda, len(tr_data))
				if dropout: self.dropoutRetake()
			print("epoch {} completed".format(ep))
			
			if monitor_tr_cost:
				cost = self.getCost(tr_data, lmbda)
				tr_cost.append(cost)
				print("Cost on training data: {}".format(cost))
			
			if monitor_tr_accuracy:
				good = self.evaluate(tr_data)
				tr_accuracy.append(good)
				print("Accuracy on training data: {} / {} - {}%".format( \
					good, len(tr_data), 100.0 * good / len(tr_data)))
			
			if monitor_test_cost:
				cost = self.getCost(test_data, lmbda)
				test_cost.append(cost)
				print("Cost on test data: {}".format(cost))
			
			if monitor_test_accuracy:
				good = self.evaluate(test_data)
				test_accuracy.append(good)
				print("Accuracy on test data: {} / {} - {}%".format( \
					good, len(test_data), 100.0 * good / len(test_data)))
			
			end = time.time()
			print("time:", end - start)
			print()
			
		return test_cost, test_accuracy, tr_cost, tr_accuracy
	
	def updateMbatch(self, data, eta, lmbda, n):
		nabla_b, nabla_w = self.backprop(data[0], data[1])
		
		self.biases = [prev - eta * add for prev, add in zip(self.biases, nabla_b)]
		self.weights = [(1 - eta * lmbda / n) * prev - eta * add \
		                for prev, add in zip(self.weights, nabla_w)]
	
	def backprop(self, ins, outs):
		mbatch_size = ins.shape[1]
		
		acts = [ins]
		zs = []
		
		act = ins
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, act) + b
			act = sigmoid(z)
			zs.append(z)
			acts.append(act)
		
		delta = outputDelta(act, outs, zs[-1])
		
		dnabla_b = [np.zeros(b.shape) for b in self.biases]
		dnabla_w = [np.zeros(w.shape) for w in self.weights]
		
		dnabla_b[-1] = np.sum(delta, axis = 1, keepdims = True)
		dnabla_w[-1] = np.dot(delta, acts[-2].transpose())
		
		for l in range(2, self.n_layers):
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoidPrime(zs[-l])
			dnabla_b[-l] = np.sum(delta, axis = 1, keepdims = True)
			dnabla_w[-l] = np.dot(delta, acts[-l - 1].transpose())
		
		dnabla_b = [x / mbatch_size for x in dnabla_b]
		dnabla_w = [x / mbatch_size for x in dnabla_w]
		return dnabla_b, dnabla_w

def outputDelta(act, des, z):
	# return (des - act) * sigmoidPrime(z)
	return (act - des)

def cost(act, des):
	return np.sum(np.nan_to_num(-des * np.log(act) - (1 - des) * np.log(1 - act)))

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x) * (1 - sigmoid(x))

def vectorize(out):
	e = np.zeros((10, 1))
	e[out] = 1.0
	return e

def prepareData(data):
	return [(np.array([np.concatenate(x / 255)]).transpose(), vectorize(y)) for x, y in data]

def makeMatrix(arr, y, x, posy, posx, val):
	arr = np.concatenate([arr, np.full((arr.shape[0], x - arr.shape[1]), val)], axis = 1)
	arr = np.concatenate([arr, np.full((y - arr.shape[0], arr.shape[1]), val)])
	
	ky = [-1] * y
	for i in range(len(posy)):
		ky[posy[i]] = i
	
	ile = len(posy)
	for i in range(y):
		if ky[i] == -1:
			ky[i] = ile
			ile += 1
			
	kx = [-1] * x
	for i in range(len(posx)):
		kx[posx[i]] = i
	
	ile = len(posx)
	for i in range(x):
		if kx[i] == -1:
			kx[i] = ile
			ile += 1
	
	return arr[ky][:,kx]

def setSeed(s):
	random.seed(s)
	np.random.seed(s)


def mkdir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)

def main():
	tr_data = mnist_loader.read(dataset = "training", path = "data/")
	tr_data, val_data = tr_data[:50000], tr_data[50000:]
	test_data = mnist_loader.read(dataset = "testing", path = "data/")
	
	tr_data = tr_data[:1000]
	
	tr_data_net = prepareData(tr_data)
	test_data_net = prepareData(test_data)
	val_data_net = prepareData(val_data)
	
	netf = Network([784, 100, 10])
	test_costf, test_accuracyf, tr_costf, tr_accuracyf = \
		netf.SGD(tr_data_net, 100, 10, 0.5,
		test_data = val_data_net,
		lmbda = 0.0,
		monitor_tr_cost = False,
		monitor_tr_accuracy = True,
		monitor_test_cost = False,
		monitor_test_accuracy = True,
		dropout = False)

if __name__ == "__main__":
	main()
