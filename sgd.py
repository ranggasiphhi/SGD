
from math import exp
import numpy as np
import re
import matplotlib.pyplot as plt

class SGD:
	ROW_DATA = 100
	DATA = np.zeros((100,5))
	ALPHA = 0.1
	EPOCH = 60

	def initiate_array(self):
		self.arrh_func = []
		self.arrsignoid = []
		self.arrerror = []
		self.arrprediction= []
		self.arrdelta = []

	def random_theta(self):
		self.theta = [ [0.5, 0.1, 0.1, 0.9, 0.9] ]

	def change_theta(self):
		old = []
		for i in range(5):
			old.append(self.theta[self.ROW_DATA][i])
		self.theta = [old]

	def save_data(self, data, i, j):
		self.DATA[i][j-1] = data

	def get_x(self,i):
		for j in range(1,5):
			num = float(self.m.group(j));
			self.save_data(num,i,j)

	def get_class(self,i):
		if self.m.group(5) == 'Iris-setosa':
			self.save_data(1,i,5)
		else:
			self.save_data(0,i,5)

	def matching(self,data):
		p = re.compile(r'([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),([\w\-]+)')
		self.m = p.match(data)

	def get_data(self):
		f = open('iris.txt','r')
		f1 = f.readlines()
		for i in range(self.ROW_DATA):
			self.matching(f1[i])	
			self.get_x(i)
			self.get_class(i)
		f.close

	def convert_to_numpy(self,arr):
		return np.array(arr)

	def h_function(self,i):
		return self.DATA[i][0] * self.theta[i][0] + self.DATA[i][1] * self.theta[i][1] + self.DATA[i][2] * self.theta[i][2] + self.DATA[i][3] * self.theta[i][3] + self.theta[i][4]

	def signoid(self,i):
		return 1 / (1 + exp(-self.arrh_func[i]))
		
	def error(self,i):
		return (self.arrsignoid[i] - self.DATA[i][4])**2
		
	def prediction(self,i):
		if self.arrsignoid[i] < 0.5:
			return 0
		else:
			return 1
		
	def delta_bias(self,i):
		return 2*(self.arrsignoid[i] - self.DATA[i][4])*(1 - self.arrsignoid[i])*self.arrsignoid[i]

	def delta_theta(self,i):
		arr_function = []
		for j in range(4):
			function = self.delta_bias(i)*self.DATA[i][j]
			arr_function.append(function)
		arr_function.append(self.delta_bias(i))
		return arr_function
		
	def new_theta(self,i):
		arr_function = []
		for j in range(5):
			function = self.theta[i][j] - self.ALPHA * self.arrdelta[i][j]
			arr_function.append(function)
		return arr_function
		
	def sum_error(self):
		return self.arrerror.sum()

	def main(self):
		arr_sumerror = []
		self.initiate_array()
		self.random_theta()
		self.get_data()
		print self.DATA
		print "THETA LAMA"
		print "=========="
		print self.theta
		for i in range(self.EPOCH):
			for j in range(self.ROW_DATA):
				formula = self.h_function(j)
				self.arrh_func.append(formula)
				formula = self.signoid(j)
				self.arrsignoid.append(formula)
				formula = self.error(j)
				self.arrerror.append(formula)
				formula = self.prediction(j)
				self.arrprediction.append(formula)
				formula = self.delta_theta(j)
				self.arrdelta.append(formula)
				formula = self.new_theta(j)
				self.theta.append(formula)
			#print self.convert_to_numpy(self.arrh_func)
			#print self.convert_to_numpy(self.arrsignoid)
			self.arrerror = self.convert_to_numpy(self.arrerror)
			#print self.arrerror
			#print self.convert_to_numpy(self.arrprediction)
			#print self.convert_to_numpy(self.arrdelta)
			arr_sumerror.append(self.sum_error())
			self.initiate_array()
			self.change_theta()
		arrerrornp = self.convert_to_numpy(arr_sumerror)
		print arrerrornp
		plt.plot(arrerrornp)
		plt.show()

a = SGD()
a.main()


