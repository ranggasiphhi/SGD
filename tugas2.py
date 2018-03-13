from math import exp
import numpy as np
import re
import matplotlib.pyplot as plt

class SGD:
	DATA_CLASS_1_TRAIN_START = 0
	DATA_CLASS_1_TRAIN_END = 40
	DATA_CLASS_1_VALIDATION_START = 40
	DATA_CLASS_1_VALIDATION_END = 50
	DATA_CLASS_2_TRAIN_START = 50
	DATA_CLASS_2_TRAIN_END = 90
	DATA_CLASS_2_VALIDATION_START = 90
	DATA_CLASS_2_VALIDATION_END = 100 
	#ROW_DATA = 100
	TRAINING_DATA = np.zeros((80,5))
	VALIDATION_DATA = np.zeros((20,5))
	ALPHA = 0.1
	EPOCH = 60

	def initiate_array(self):
		self.arrh_func = []
		self.arrsignoid = []
		self.arrerror = []
		self.arrprediction= []
		self.arrdelta = []
		self.arr_sumerror = []

	def random_theta(self):
		self.theta = [ [0.5, 0.1, 0.1, 0.9, 0.9] ]
		#self.theta = [ [0.1, 0.15, 0.2, 0.3, 0.4] ]

	def change_theta(self,rows):
		old = []
		for i in range(5):
			old.append(self.theta[rows][i])
		self.theta = [old]

	#def save_data(self, data):
	#	self.subdata.append(data)

	def get_x(self,i):
		for j in range(4):
			self.mini_array.append(float(self.m.group(j+1)))

	def get_class(self,i):
		if self.m.group(5) == 'Iris-setosa':
			self.mini_array.append(1)
		else:
			self.mini_array.append(0)

	def matching(self,data):
		p = re.compile(r'([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),([\w\-]+)')
		self.m = p.match(data)

	def get_data(self,start,end):
		self.subdata = []
		f = open('iris.txt','r')
		f1 = f.readlines()
		for i in range(start,end):
			self.mini_array = []
			self.matching(f1[i])	
			self.get_x(i)
			self.get_class(i)
			self.subdata.append(self.mini_array)
		f.close

	def convert_to_numpy(self,arr):
		return np.array(arr)

	def h_function_training(self,data,i):
		return data[i][0] * self.theta[i][0] + data[i][1] * self.theta[i][1] + data[i][2] * self.theta[i][2] + data[i][3] * self.theta[i][3] + self.theta[i][4]

	def h_function_validation(self,data,i):
		#print self.theta_validation
		return data[i][0] * self.theta_validation[0] + data[i][1] * self.theta_validation[1] + data[i][2] * self.theta_validation[2] + data[i][3] * self.theta_validation[3] + self.theta_validation[4]

	def signoid(self,i):
		return 1 / (1 + exp(-self.arrh_func[i]))
		
	def error(self,data,i):
		return (self.arrsignoid[i] - data[i][4])**2
		
	def prediction(self,i):
		if self.arrsignoid[i] < 0.5:
			return 0
		else:
			return 1
		
	def delta_bias(self,data,i):
		return 2*(self.arrsignoid[i] - data[i][4])*(1 - self.arrsignoid[i])*self.arrsignoid[i]

	def delta_theta(self,data,i):
		arr_function = []
		for j in range(4):
			function = self.delta_bias(data,i)*data[i][j]
			arr_function.append(function)
		arr_function.append(self.delta_bias(data,i))
		return arr_function
		
	def new_theta(self,i):
		arr_function = []
		for j in range(5):
			function = self.theta[i][j] - self.ALPHA * self.arrdelta[i][j]
			arr_function.append(function)
		return arr_function
		
	def sum_error(self):
		return np.average(self.arrerror)

	def h_to_prediction(self,data,j):
		formula = self.signoid(j)
		self.arrsignoid.append(formula)
		formula = self.error(data,j)
		self.arrerror.append(formula)
		formula = self.prediction(j)
		self.arrprediction.append(formula)

	def counting_error(self):
		#print self.convert_to_numpy(self.arrh_func)
		#print self.convert_to_numpy(self.arrsignoid)
		#print "ERROR KE EPOCH "+str(i+1)
		#print "==========================="
		self.arrerror = self.convert_to_numpy(self.arrerror)
		#print self.arrerror
		#print self.convert_to_numpy(self.arrprediction)
		#print self.convert_to_numpy(self.arrdelta)
		self.arr_sumerror.append(self.sum_error())
		return self.convert_to_numpy(self.arr_sumerror)


	def sgd(self,data):
		self.theta_validation = []	
		for j in range(data.shape[0]):
			formula = self.h_function_training(data,j)
			self.arrh_func.append(formula)
			self.h_to_prediction(data,j)
			formula = self.delta_theta(data,j)
			self.arrdelta.append(formula)
			formula = self.new_theta(j)
			self.theta.append(formula)
		self.theta_validation = self.theta[self.TRAINING_DATA.shape[0]]
		return self.counting_error()

	def validation(self,data):
		for j in range(data.shape[0]):
			formula = self.h_function_validation(data,j)
			self.arrh_func.append(formula)
			self.h_to_prediction(data,j)
		return self.counting_error()



	def main(self):
		error_training = np.zeros([self.EPOCH])
		error_validating = np.zeros([self.EPOCH])
		self.initiate_array()
		self.random_theta()
		print "DATA SET TRAINING"
		print "================="
		self.get_data(self.DATA_CLASS_1_TRAIN_START, self.DATA_CLASS_1_TRAIN_END)
		data_1 = self.convert_to_numpy(self.subdata).copy()
		self.get_data(self.DATA_CLASS_2_TRAIN_START, self.DATA_CLASS_2_TRAIN_END)
		data_2 = self.convert_to_numpy(self.subdata).copy()
		self.TRAINING_DATA = np.vstack([data_1,data_2])
		print self.TRAINING_DATA
		print "DATA SET VALIDATION"
		print "==================="
		self.get_data(self.DATA_CLASS_1_VALIDATION_START, self.DATA_CLASS_1_VALIDATION_END)
		data_1 = self.convert_to_numpy(self.subdata).copy()
		self.get_data(self.DATA_CLASS_2_VALIDATION_START, self.DATA_CLASS_2_VALIDATION_END)
		data_2 = self.convert_to_numpy(self.subdata).copy()
		self.VALIDATION_DATA = np.vstack([data_1,data_2])
		print self.VALIDATION_DATA
		print "THETA"
		print "=========="
		print self.theta
		for i in range(self.EPOCH):
			error_training[i] = self.sgd(self.TRAINING_DATA)
			self.initiate_array()
			#print self.convert_to_numpy(self.theta)
			error_validating[i] = self.validation(self.VALIDATION_DATA)
			self.initiate_array()
			self.change_theta(self.TRAINING_DATA.shape[0])
		print self.theta
		print self.theta_validation
		print "ERROR TRAINING"
		print "==========="
		print error_training
		print "ERROR VALIDATING"
		print "================"
		print error_validating

		plt.plot(error_training,'r--', error_validating, 'g--')
		plt.show()

a = SGD()
a.main()


