import numpy as np
import random
import mnist_loader

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
	
	def SGD(self, tr_data, epochs, mbatch_size, eta, test_data = None,
	        lmbda = 0.0,
	        monitor_test_cost=False,
	        monitor_test_accuracy=False,
	        monitor_tr_cost=False,
	        monitor_tr_accuracy=False):
	
		test_cost, test_accuracy = [], []
		tr_cost, tr_accuracy = [], []
		for ep in range(epochs):
			random.shuffle(tr_data)
			
			ins_all = [x[0].transpose()[0] for x in tr_data]
			outs_all = [x[1].transpose()[0] for x in tr_data]
			
			mbatches = [(np.array(ins_all[k:k + mbatch_size]).transpose(), \
			             np.array(outs_all[k:k + mbatch_size]).transpose()) \
			             for k in range(0, len(tr_data), mbatch_size)]
			
			for mbatch in mbatches:
				self.updateMbatch(mbatch, eta, lmbda, len(tr_data))
			
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

def prepareData(data):
	def vectorize(out):
		e = np.zeros((10, 1))
		e[out] = 1.0
		return e
	return [(np.array([np.concatenate(x / 255)]).transpose(), vectorize(y)) for x, y in data]

def main():
	net = Network([784, 200, 10])
	
	tr_data = mnist_loader.read(dataset = "training", path = "data/")
	tr_data, val_data = tr_data[:50000], tr_data[50000:]
	test_data = mnist_loader.read(dataset = "testing", path = "data/")
	
	tr_data_net = prepareData(tr_data)
	test_data_net = prepareData(test_data)
	val_data_net = prepareData(val_data)

	net.SGD(tr_data_net, 50, 10, 0.5,
	        test_data = val_data_net,
	        lmbda = 5.0,
	        monitor_test_cost=False,
	        monitor_test_accuracy=True,
	        monitor_tr_cost=False,
	        monitor_tr_accuracy=False)

	# for (ind, ans) in net.evaluateBad(test_data_net):
	# 	print("correct: ", test_data[ind][1], ", net: ", ans)
	# 	print(test_data[ind][0])
		# mnist_loader.show(test_data[ind][0])

if __name__ == "__main__":
	main()
