import xml.etree.ElementTree as ET
import scipy as sp
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import os,re
import Input_file_reader as IFR
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.interpolate import CubicSpline
import multiprocessing
import sys,time
import make_input_file
from scipy.linalg import cholesky
from scipy.optimize import shgo
from scipy.optimize import BFGS
from MechanismParser import Parser
import subprocess
#############################################################
###	   Uncertainty for arrhenius parameters		 ######
###	   of elementary reactions					  ######
#############################################################
def run(sample,data,length):
	A = UncertaintyExtractor(data)
	a1 =2*np.random.random_sample(1)-1
	a2 = 2*np.random.random_sample(1)-1
	a3 = 2*np.random.random_sample(1)-1
	A.populateValues(a1,a2,a3)
	A.getCovariance(flag=False)
	A.getUnCorrelated(flag = False)
	zeta = A.getB2Zeta(flag=True)
	#print(data)
	#zeta =UncertaintyExtractor(data).getExtremeCurves(sample)	
	del A
	print(zeta)
	
	return (sample,zeta,length)

def run_sampling_b_partial(sample, data, generator, length):
	"""
	Multiprocessing-compatible top-level function for parallel class-B
	partial-parameter sampling.  Called by workers.do_unsrt_b_partial.
	Generates ONE full-space ζ for data["param_indices"].
	No SLSQP — uses the analytical instance methods of UncertaintyExtractor.
	"""
	rng		   = np.random.default_rng()
	param_indices = data["param_indices"]
	A = UncertaintyExtractor(data)
	a1 = generator[0]
	a2 = generator[1]
	A.populateValues(a1,a2)
	A.getCovariance(flag=False)
	A.getUnCorrelated(flag=False)
	m = len(param_indices)
	if m = 3:
		zeta_list  = A.getB2Zeta(flag=True)
	
	elif m = 2:
		zeta_list	 = A.getClassB_partial(data, param_indices, 1, rng)
	
	else:
		zeta_list = A.getClassA_partial(param_indices, 1, rng)
	
	zeta = list(zeta_list[0]) if zeta_list else [0.0, 0.0, 0.0]
	del A
	return (sample, generator, zeta, length)


def run_sampling_c_partial(sample, data, generator, length):
	"""
	Multiprocessing-compatible top-level function for parallel class-C
	partial-parameter sampling.  Called by workers.do_unsrt_c_partial.
	Generates ONE full-space ζ for data["param_indices"].
	"""
	A = UncertaintyExtractor(data)
	A.getCovariance(flag=False)
	A.getUnCorrelated(flag=False)
	param_indices = data["param_indices"]
	rng		   = np.random.default_rng()
	zeta_list	 = A.getClassC_partial(param_indices, 1, rng)
	if not zeta_list:
		zeta_list = A.getClassA_partial(param_indices, 1, rng)
	zeta = list(zeta_list[0]) if zeta_list else [0.0, 0.0, 0.0]
	del A
	return (sample, generator, zeta, length)


class workers(object):
	def __init__(self,workers):
		#print("Initialized\n")
		self.pool = multiprocessing.Pool(processes=workers)
		self.progress = []
		self.parallized_zeta = []

	def callback(self,result):
		self.progress.append(result[0])
		self.parallized_zeta.append(result[1])
		sys.stdout.write("\t\t\r{:06.2f}% is complete".format(
			len(self.progress)/float(result[-1])*100))
		sys.stdout.flush()

	def callback_error(self, result):
		print('error', result)

	def do_job_async(self,data,sampling_points):
		for args in range(sampling_points):
			x = self.pool.apply_async(run, 
				  args=(1,data,sampling_points), 
				  callback=self.callback)
		self.pool.close()
		self.pool.join()
		print(x.get())
		self.pool.terminate()
		return self.parallized_zeta

	def do_unsrt_b_partial(self, data, sampling_points):
		"""
		Parallel class-B sampling for a partial parameter subset.
		
		Expects data to contain:
		  data["param_indices"]		– tuple of active parameter indices
		  data["generators_b_partial"] – array of shape (sampling_points, 2)

		Each worker call (run_sampling_b_partial) instantiates a fresh
		UncertaintyExtractor and generates ONE class-B zeta using the
		analytical 2×2-bisection (m=2) or 3×3-direct-solve (m=3) method.

		Returns (parallized_zeta, progress) mirroring the Worker interface.
		"""
		for args in range(sampling_points):
			self.pool.apply_async(
				run_sampling_b_partial,
				args=(1, data, data["generators_b_partial"][args], sampling_points),
				callback=self.callback,
				error_callback=self.callback_error)
		self.pool.close()
		self.pool.join()
		self.pool.terminate()
		return self.parallized_zeta, self.progress

	def do_unsrt_c_partial(self, data, sampling_points):
		"""
		Parallel class-C sampling for a partial parameter subset.

		Expects data to contain:
		  data["param_indices"]		– tuple of active parameter indices
		  data["generators_c_partial"] – array of shape (sampling_points, 2)

		Returns (parallized_zeta, progress) mirroring the Worker interface.
		"""
		for args in range(sampling_points):
			self.pool.apply_async(
				run_sampling_c_partial,
				args=(1, data, data["generators_c_partial"][args], sampling_points),
				callback=self.callback,
				error_callback=self.callback_error)
		self.pool.close()
		self.pool.join()
		self.pool.terminate()
		return self.parallized_zeta, self.progress

class UncertaintyExtractor(object):
	def __init__(self,data):
		self.data = data
		self.temperatures = data["temperatures"]
		self.uncertainties = data["uncertainties"]
		#cs = CubicSpline(self.temperatures,self.uncertainties)
		#self.temperatures = np.arange(self.temperatures[0],self.temperatures[-1],25)
		#self.uncertainties = cs(self.temperatures)
		self.ArrheniusParams = data["Arrhenius"]
		self.Theta = np.array([self.temperatures/self.temperatures,np.log(self.temperatures),-1/(self.temperatures)])
		self.M = 3.0/np.log(10.0)
		self.guess = np.array([-10.0,-10.0,0.5,200.0,10.0,5.0,1,1])
		self.guess_z = np.array([1,1,1])
		self.guess_z2 = np.array([10,10,100,200])
		self.parallized_zeta = []
		
		self.generator = []
		self.samples = []
		self.kleft_fact = None
		self.kright_fact = None
		self.kmiddle_fact = None
	
	def callback(self,result):
		self.progress.append(result[0])
		self.parallized_zeta.append(result[1])
		sys.stdout.write("\t\t\r{:06.2f}% is complete".format(len(self.progress)/float(result[-1])*100))
		sys.stdout.flush()
	def getPool(self,workers):
		#print("Entered pool")
		self.pool = multiprocessing.Pool(processes=workers)
	def do_job_async(self,sampling_points):
		#print("Entered Async\n")
		
		for args in range(sampling_points):
			self.pool.apply_async(run, 
				  args=(1,self.data,sampling_points), 
				  callback=self.callback)
		self.pool.close()
		self.pool.join()
		self.pool.terminate()	
		
	def getUncertFunc(self,L):
		func = [self.M*np.linalg.norm(np.dot(L.T,i)) for i in self.Theta.T]
		return np.asarray(func)
	
	def getZetaUnsrtKappaFunc(self,L,z):
		func = [(i.T.dot(L.dot(z))) for i in self.theta_for_kappa.T]
		return np.asarray(func)
	
	
	def getZetaUnsrtFunc(self,L,z):
		func = [(i.T.dot(L.dot(z))) for i in self.Theta.T]
		return np.asarray(func)	

	def obj_func(self,guess):
		z = guess
		cov = np.array([[z[0],0,0],[z[1],z[2],0],[z[3],z[4],z[5]]]);#cholesky lower triangular matric		
		f = (self.uncertainties - self.getUncertFunc(cov))/(self.uncertainties/self.M)
		obj = np.dot(f,f)
		return obj

	def const_func(self,guess):
		self.z = guess
		f = np.zeros(len(self.temperatures))
		f = self.uncertainties - self.getUncertFunc()
		return np.amin(f)

	def obj_func_zeta(self,guess):
		M = self.M
		T = self.temperatures
		cov = self.L
		Theta = np.array([T/T,np.log(T),-1/(T)])
		f = (self.uncertainties-self.getZetaUnsrtFunc(cov,guess))
		obj = np.dot(f,f)
		return obj
	
	def obj_func_zeta_b2(self,guess):
		M = self.M
		T = self.temperatures
		cov = self.L
		Theta = np.array([T/T,np.log(T),-1/(T)])
		f = (self.uncertainties-self.getZetaUnsrtFunc(cov,guess[0:-1]))
		obj = np.dot(f,f)
		return obj
		
	def obj_func_zeta_c2(self,guess):
		M = self.M
		T = self.temperatures
		cov = self.L
		Theta = np.array([T/T,np.log(T),-1/(T)])
		f = (self.Yu-self.getZetaUnsrtFunc(cov,guess))
		obj = np.dot(f,f)
		return obj
	
	def const_func_zeta_1(self,z):
		M = self.M
		T = self.temperatures
		cov = self.L
		Theta = np.array([T/T,np.log(T),-1/(T)])
		normTheta = np.sqrt((T/T)**2 + (np.log(T))**2 + (1/T)**2)
		unsrtFunc = M*np.linalg.norm(np.dot(cov.T,Theta))
		uncorrFunc = np.linalg.norm(np.dot(cov,guess))
		QLT = np.asarray(np.dot(Theta.T,np.dot(cov,guess))).flatten()
		f = (self.uncertainties - self.getZetaUnsrtFunc(cov,z))
		#print(np.dot(np.transpose(theta),np.dot(L1,eta)))
		obj = np.dot(f,f)
		return np.amin(f)

	def const_func_zeta_2(self,z):
		M = self.M
		T = self.temperatures[1:-1]
		cov = self.L
		Theta = np.array([T/T,np.log(T),-1/(T)])
		QtLZ = np.asarray([(i.dot(cov.dot(z))) for i in Theta.T])
		f = (self.uncertainties[1:-1]-QtLZ)
		return np.amin(f)
	
	def const_1_typeB_Zeta(self,z):
		M = self.M
		#T = self.temperatures
		P = self.ArrheniusParams
		T = self.temperatures[0]
		cov = self.L
		Pmin = self.P_min
		
		Theta = np.array([T/T,np.log(T),-1/(T)])
		k_min = Theta.dot(Pmin)
		QtLZ = (Theta.T.dot(cov.dot(z)))
		f = Theta.dot(P)-QtLZ
		return k_min - f
	
	def const_2_typeB_Zeta(self,z):
		M = self.M
		#T = self.temperatures
		T = self.const_T
		cov = self.L
		Pmax = self.P_max
		P = self.ArrheniusParams
		Theta = np.array([T/T,np.log(T),-1/(T)])
		k_max = Theta.dot(Pmax)
		QtLZ = (Theta.T.dot(cov.dot(z)))
		f = (Theta.dot(P)-QtLZ)
		return k_max - f
	
	def const_3_typeB_Zeta(self,z):
		M = self.M
		#T = self.temperatures
		T = self.temperatures[-1]
		cov = self.L
		Pmin = self.P_min
		P = self.ArrheniusParams
		Theta = np.array([T/T,np.log(T),-1/(T)])
		k_min = Theta.dot(Pmin)
		QtLZ = (Theta.T.dot(cov.dot(z)))  
		f = (Theta.dot(P)-QtLZ)
		return k_min - f
	
	def const_2_typeC_Zeta(self,z):
		M = self.M
		T = self.temperatures[-1]
		cov = self.L
		Pmax = self.P_max
		P = self.ArrheniusParams
		Theta = np.array([T/T,np.log(T),-1/(T)])
		k_max = Theta.dot(Pmax)
		QtLZ = (Theta.T.dot(cov.dot(z)))
		f = (Theta.dot(P)-QtLZ)
		return k_max - f
	
	
	def cons_derivative_b2(self,z):
		if self.kright_fact < 0 and self.kleft_fact<0:
			T = z[-1]
			guess = z[0:-1]
			P = self.ArrheniusParams
			cov = self.L
			P_max = P + np.asarray(np.dot(cov,self.kmiddle_fact*self.zeta.x)).flatten()
			theta = np.array([0*(T/T),1/T,1/T**2])
			dk_dt = theta.T.dot(P_max)
			dko_dt = theta.T.dot(P)
			dQtLZ_dt =(theta.T.dot(cov.dot(z[0:-1])))
			obj = dk_dt - dko_dt - dQtLZ_dt
		elif self.kright_fact >0 and self.kleft_fact>0:
			T = z[-1]
			guess = z[0:-1]
			P = self.ArrheniusParams
			cov = self.L
			P_min = P - np.asarray(np.dot(cov,self.kmiddle_fact*self.zeta.x)).flatten()
			theta = np.array([0,1/T,1/T**2])
			dk_dt = theta.T.dot(P_min)
			dko_dt = theta.T.dot(P)
			dQtLZ_dt =(theta.T.dot(cov.dot(z[0:-1])))
			obj = dk_dt - dko_dt - dQtLZ_dt
			
		elif self.kright_fact>0 and self.kleft_fact<0:
			T = z[-1]
			guess = z[0:-1]
			P = self.ArrheniusParams
			cov = self.L
			P_min = P + self.kmiddle_fact*np.asarray(np.dot(cov,self.zeta.x)).flatten()
			theta = np.array([0,1/T,1/T**2])
			dk_dt = theta.T.dot(P_min)
			dko_dt = theta.T.dot(P)
			dQtLZ_dt =(theta.T.dot(cov.dot(z[0:-1])))
			obj = dk_dt - dko_dt - dQtLZ_dt
		elif self.kright_fact<0 and self.kleft_fact>0:
			T = z[-1]
			guess = z[0:-1]
			P = self.ArrheniusParams
			cov = self.L
			P_max = P - self.kmiddle_fact*np.asarray(np.dot(cov,self.zeta.x)).flatten()
			theta = np.array([0,1/T,1/T**2])
			dk_dt = theta.T.dot(P_max)
			dko_dt = theta.T.dot(P)
			dQtLZ_dt =(theta.T.dot(cov.dot(z[0:-1])))
			obj = dk_dt - dko_dt - dQtLZ_dt
		else:
			T = z[-1]
			guess = z[0:-1]
			P = self.ArrheniusParams
			cov = self.L
			P_max = P + self.kmiddle_fact*np.asarray(np.dot(cov,self.zeta.x)).flatten()
			theta = np.array([0,1/T,1/T**2])
			dk_dt = theta.T.dot(P_max)
			dko_dt = theta.T.dot(P)
			dQtLZ_dt =(theta.T.dot(cov.dot(z[0:-1])))
			obj = dk_dt - dko_dt - dQtLZ_dt
		return obj
		
	def const_2_typeB2_Zeta(self,z):
		if self.kleft_fact >0 and self.kright_fact>0:
			M = self.M
			T = z[-1]
			cov = self.L
			P = self.ArrheniusParams
			Pmin = P - self.kmiddle_fact*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([T/T,np.log(T),-1/(T)]).astype(float)
			k_min = Theta.dot(Pmin)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = (Theta.dot(P)+QtLZ)
			obj = k_min - f
		elif self.kleft_fact <0 and self.kright_fact<0:
			M = self.M
			T = z[-1]
			cov = self.L
			P = self.ArrheniusParams
			Pmax = P + self.kmiddle_fact*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_max = Theta.dot(Pmax)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = (Theta.dot(P)+QtLZ)
			obj = k_max - f
		elif self.kleft_fact <0 and self.kright_fact>0:
			M = self.M
			T = z[-1]
			cov = self.L
			P = self.ArrheniusParams
			Pmax = P + self.kmiddle_fact*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_max = Theta.dot(Pmax)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = (Theta.dot(P)+QtLZ)
			obj = k_max - f
			"""
			M = self.M
			T = self.temperatures[1:-1]
			cov = self.L
			Theta = np.array([T/T,np.log(T),-1/(T)])
			QtLZ = np.asarray([(i.dot(cov.dot(z[0:-1]))) for i in Theta.T])
			f = (self.uncertainties[1:-1]-QtLZ)
			obj = np.amin(f)
			"""
		elif self.kleft_fact >0 and self.kright_fact<0:
			M = self.M
			T = z[-1]
			cov = self.L
			P = self.ArrheniusParams
			Pmin = P - self.kmiddle_fact*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_min = Theta.dot(Pmin)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = (Theta.dot(P)+QtLZ)
			obj = k_min - f
			"""
			M = self.M
			T = self.temperatures[1:-1]
			cov = self.L
			Theta = np.array([T/T,np.log(T),-1/(T)])
			QtLZ = np.asarray([(i.dot(cov.dot(z[0:-1]))) for i in Theta.T])
			f = (self.uncertainties[1:-1]-QtLZ)
			obj = np.amin(f)
			"""
		else:
			M = self.M
			T = z[-1]
			cov = self.L
			P = self.ArrheniusParams
			Pmin = P - self.kmiddle_fact*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_min = Theta.dot(Pmin)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = (Theta.dot(P)+QtLZ)
			obj = k_min - f
		return obj

	def const_1_typeB2_Zeta(self,z):
		if self.kleft_fact <0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[0]
			cov = self.L
			P_left = P - abs(self.kleft_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_left = Theta.dot(P_left)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		
		elif self.kleft_fact>0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[0]
			cov = self.L
			P_left = P + abs(self.kleft_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_left = Theta.dot(P_left)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		else:
			M = self.M
			P = self.ArrheniusParams
			#T = self.temperatures
			T = self.temperatures[0]
			cov = self.L
			P_right = P 
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_left = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		return k_left - f
	
	def const_1_typeC2_Zeta(self,z):
		if self.kleft_fact <0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[0]
			cov = self.L
			P_left = P - abs(self.kleft_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_left = Theta.dot(P_left)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		elif self.kleft_fact>0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[0]
			cov = self.L
			P_left = P + abs(self.kleft_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_left = Theta.dot(P_left)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		else:
			M = self.M
			P = self.ArrheniusParams
			#T = self.temperatures
			T = self.temperatures[0]
			cov = self.L
			P_right = P 
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_left = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		return k_left - f
	
	def populateValues(self,a1,a2):
		self.kleft_fact = a1
		self.kright_fact = a2
		self.kmiddle_fact = 1.0
		#print(f"In populate{a1},{a2}\n")
	def const_3_typeB2_Zeta(self,z):
		if self.kright_fact <0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[-1]
			cov = self.L
			P_right = P - abs(self.kright_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_right = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		elif self.kright_fact >0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[-1]
			cov = self.L
			P_right = P + abs(self.kright_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_right = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		else:
			M = self.M
			P = self.ArrheniusParams
			#T = self.temperatures
			T = self.temperatures[0]
			cov = self.L
			P_right = P 
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_right = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		return k_right - f
	
	def const_3_typeC2_Zeta(self,z):
		if self.kright_fact <0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[-1]
			cov = self.L
			P_right = P - abs(self.kright_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_right = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		elif self.kright_fact >0:
			M = self.M
			P = self.ArrheniusParams
			T = self.temperatures[-1]
			cov = self.L
			P_right = P + abs(self.kright_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
			Theta = np.array([1,np.log(T),-1/(T)])
			k_right = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		else:
			M = self.M
			P = self.ArrheniusParams
			#T = self.temperatures
			T = self.temperatures[0]
			cov = self.L
			P_right = P 
			Theta = np.array([T/T,np.log(T),-1/(T)])
			k_right = Theta.dot(P_right)
			QtLZ = (Theta.T.dot(cov.dot(z[0:-1])))
			f = Theta.dot(P)+QtLZ
		return k_right - f
	
	
	def getCovariance(self,flag = False):
		if flag == True:
			constraints = {'type': 'ineq', 'fun': self.const_func }
			self.const = [constraints]
			START_MUQ = time.time()
			self.solution = minimize(self.obj_func,self.guess,constraints=self.cons)
			STOP_MUQ = time.time()
			#print(f"\nMUQ-SAC method takes {STOP_MUQ-START_MUQ}\n")
		else:
			START_MUQ = time.time()
			self.solution = minimize(self.obj_func,self.guess,method="SLSQP")
			STOP_MUQ = time.time()
			#print(f"\nMUQ-SAC method takes {STOP_MUQ-START_MUQ}\n")
			
		#print(self.solution.x)
		self.L = np.array([[self.solution.x[0],0,0],[self.solution.x[1],self.solution.x[2],0],[self.solution.x[3],self.solution.x[4],self.solution.x[5]]]);#cholesky lower triangular matric
		cov1 = self.L
		cov2 = np.dot(self.L,self.L.T)
		#print(cov2)
		#print(np.exp(self.ArrheniusParams[0]),self.ArrheniusParams[1],self.ArrheniusParams[2])
		#print(self.temperatures[0],self.temperatures[-1])
		D,Q = np.linalg.eigh(cov2)
		self.A = Q.dot(sp.linalg.sqrtm(np.diag(D)))
		self.cov = self.L
	
	def getUnCorrelated(self,flag = False):
		self.getCovariance(flag = False)
		if flag == True:
			#con1 = {'type': 'ineq', 'fun': self.const_func_zeta_1}
			#con2 = {'type': 'ineq', 'fun': self.const_func_zeta_2}
			con3 = {'type': 'eq', 'fun': self.const_nmax}
			con4 = {'type': 'eq', 'fun': self.const_nmin}
			self.const_zeta = [con3,con4]
			zeta = minimize(self.obj_func_zeta,self.guess_z,constraints=self.const_zeta)
		else:
			zeta = minimize(self.obj_func_zeta,self.guess_z,method="Nelder-Mead")

		self.zeta = zeta
		P = self.ArrheniusParams
		self.P_max = P + np.asarray(np.dot(self.L,self.zeta.x)).flatten();
		self.P_min = P - np.asarray(np.dot(self.L,self.zeta.x)).flatten();
		self.kmax = self.Theta.T.dot(self.P_max)
		self.kmin = self.Theta.T.dot(self.P_min)
		self.kappa = self.Theta.T.dot(P)
		return zeta
	
	def getConstrainedUnsrtZeta(self,flag=False):
		if flag == True:
			con1 = {'type': 'eq', 'fun': self.const_1_typeB_Zeta}
			con2 = {'type': 'eq', 'fun': self.const_2_typeB_Zeta}
			con3 = {'type': 'eq', 'fun': self.const_3_typeB_Zeta}
			self.const_zeta = [con1,con2,con3]
			zeta_list = []
			obj_val = []
			alpha = []
			n = []
			epsilon = []
			
			for i,T in enumerate(self.temperatures):
				if T<self.temperatures[-1]-300 and T> self.temperatures[0]+300:
					self.const_T = T
					zeta = minimize(self.obj_func_zeta,self.guess_z,method="SLSQP",constraints=self.const_zeta)
					#zeta = minimize(self.obj_curved_zeta,self.guess_z,method="Nelder-Mead")
					zeta_list.append(zeta.x)
					alpha.append(zeta.x[0])
					n.append(zeta.x[1])
					epsilon.append(zeta.x[2])
					obj_val.append(abs(self.obj_func_zeta(zeta.x))+abs(self.const_1_typeB_Zeta(zeta.x))+abs(self.const_2_typeB_Zeta(zeta.x))+abs(self.const_3_typeB_Zeta(zeta.x)))
			#print(self.rIndex)
			#print(obj_val)
			alpha_square = [i**2 for i in alpha]
			n_square = [i**2 for i in n]
			epsilon_square = [i**2 for i in epsilon]
			index = obj_val.index(min(obj_val))
			
			return [zeta_list[index]],index,np.array([alpha[alpha_square.index(max(alpha_square))],n[n_square.index(max(n_square))],epsilon[epsilon_square.index(max(epsilon_square))]])
		else:
			con1 = {'type': 'eq', 'fun': self.const_1_typeB_Zeta}
			con2 = {'type': 'eq', 'fun': self.const_2_typeC_Zeta}
			con3 = {'type': 'ineq', 'fun': self.const_func_zeta_2}
			self.const_zeta = [con1,con2]
			zeta = minimize(self.obj_func_zeta,self.guess_z,constraints=self.const_zeta)
			return zeta
		#return zeta_list
	
	
	def getC2Zeta(self,flag):
		self.getUnCorrelated(flag=False)
		#self.kleft_fact = 0.1
		#self.kright_fact = -0.5
		if flag == True:
			if self.kleft_fact <0:
				M = self.M
				P = self.ArrheniusParams
				T = self.temperatures[0]
				cov = self.L
				P_left = P - abs(self.kleft_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
				Theta = np.array([T/T,np.log(T),-1/(T)])
				k0_left = Theta.dot(P)
				k_left = Theta.dot(P_left)
				
				
			elif self.kleft_fact>0:
				M = self.M
				P = self.ArrheniusParams
				T = self.temperatures[0]
				cov = self.L
				P_left = P + abs(self.kleft_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
				Theta = np.array([T/T,np.log(T),-1/(T)])
				k0_left = Theta.dot(P)
				k_left = Theta.dot(P_left)
				
				
			else:
				M = self.M
				P = self.ArrheniusParams
				#T = self.temperatures
				T = self.temperatures[0]
				cov = self.L
				P_right = P 
				Theta = np.array([T/T,np.log(T),-1/(T)])
				k0_left = Theta.dot(P)
				k_left = Theta.dot(P_right)
				
			
			if self.kright_fact <0:
				M = self.M
				P = self.ArrheniusParams
				T = self.temperatures[-1]
				cov = self.L
				P_right = P - abs(self.kright_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
				Theta = np.array([1,np.log(T),-1/(T)])
				k0_right = Theta.dot(P)
				k_right = Theta.dot(P_right)
				
				
			elif self.kright_fact >0:
				M = self.M
				P = self.ArrheniusParams
				T = self.temperatures[-1]
				cov = self.L
				P_right = P + abs(self.kright_fact)*np.asarray(np.dot(self.cov,self.zeta.x)).flatten()
				Theta = np.array([1,np.log(T),-1/(T)])
				k0_right = Theta.dot(P)
				k_right = Theta.dot(P_right)
				
				
			else:
				M = self.M
				P = self.ArrheniusParams
				#T = self.temperatures
				T = self.temperatures[0]
				
				cov = self.L
				P_right = P 
				Theta = np.array([T/T,np.log(T),-1/(T)])
				k0_right = Theta.dot(P)
				k_right = Theta.dot(P_right)
				
			
			"""
			Find slope
			------
			To find that we need the f(T) from the kappa_left and kappa_right
			"""
			T2 = self.temperatures[-1]
			FT2 = k_left-k0_left
			T1 = self.temperatures[0]
			FT1 = k_right-k0_right
			
			slope = (FT2 - FT1)/(T2-T1)
			constant = FT2-slope*(T2)
			
			self.Yu = slope*(self.temperatures) + constant
								
			#con1 = {'type': 'eq', 'fun': self.const_1_typeC2_Zeta}
			#con3 = {'type': 'eq', 'fun': self.const_3_typeC2_Zeta}
			#self.const_zeta = [con1,con3]
			#bnds = ((float("-inf"),float("inf")),(float("-inf"),float("inf")),(float("-inf"),float("inf")),(200,3500))
			#zeta = minimize(self.obj_func_zeta_b2,self.guess_z2,method="SLSQP",constraints=self.const_zeta,bounds=bnds)
			zeta = minimize(self.obj_func_zeta_c2,self.guess_z,method="SLSQP")
			#zeta = shgo(self.obj_func_zeta_b2,bnds,constraints=self.const_zeta)
			#zeta = minimize(self.obj_func_zeta_c2,self.guess_z2,method="trust-constr",constraints=self.const_zeta)
			#print(zeta)
		else:
			con5 = {'type': 'ineq', 'fun': self.cons_T}
			self.const_zeta = [con5]
			bnds = ((float("-inf"),float("inf")),(float("-inf"),float("inf")),(float("-inf"),float("inf")),(200,3500))
			zeta = minimize(self.obj_func_zeta_b2,self.guess_z2,method="SLSQP",constraints=self.const_zeta,bounds=bnds)
		return [zeta.x[0],zeta.x[1],zeta.x[2]]
	
	
	def _build_bound_constraints(self,temperatures, uncertainties, dk_z_at_T,
                              n_bound_pts: int = 20):
		"""
		Return a list of SLSQP 'ineq' dicts enforcing
		   |Δκ(Tᵢ)| ≤ |uncertainties[i]|  for a downsampled temperature grid.

		Two constraints per grid point:
		   uᵢ - Δκ(Tᵢ) ≥ 0   (upper bound)
		   uᵢ + Δκ(Tᵢ) ≥ 0   (lower bound)

		Parameters
		----------
		temperatures  : (N,) array of temperatures [K]
		uncertainties : (N,) per-temperature uncertainty magnitudes
		dk_z_at_T     : callable (T, z) → float  (Δκ at a single temperature)
		n_bound_pts   : number of grid points to sample (default 20)

		Returns
		-------
		bound_cons : list of constraint dicts
		"""
		stride = max(1, len(temperatures) // n_bound_pts)
		T_grid = temperatures[::stride]
		u_grid = np.abs(uncertainties[::stride])

		bound_cons = []
		for Ti, ui in zip(T_grid, u_grid):
			Ti, ui = float(Ti), float(ui)
			bound_cons.append({
			  'type': 'ineq',
			  'fun': lambda z, Ti=Ti, ui=ui:  ui - dk_z_at_T(Ti, z[:3])  # upper
			})
			bound_cons.append({
			  'type': 'ineq',
			  'fun': lambda z, Ti=Ti, ui=ui:  ui + dk_z_at_T(Ti, z[:3])  # lower
			})
		return bound_cons
	
	def getB2Zeta(self,flag): 
		self.getUnCorrelated(flag=False)
		if flag == True:
			con1 = {'type': 'eq', 'fun': self.const_1_typeB2_Zeta}
			con2 = {'type': 'eq', 'fun': self.const_2_typeB2_Zeta}
			con3 = {'type': 'eq', 'fun': self.const_3_typeB2_Zeta}
			con4 = {'type': 'eq', 'fun': self.cons_derivative_b2}
			
			def dk_z_at_T(Tv, z):
				return float(self._psac_theta_full(np.array([Tv]))[:, 0] @ self.L @ z)
			
			bound_constraints = self._build_bound_constraints(
			   self.temperatures, self.uncertainties, dk_z_at_T, n_bound_pts=20
		    )

			self.const_zeta = [con1,con2,con3,con4] + bound_constraints
			bnds = ((float("-1000"),float("1000")),(float("-1000"),float("1000")),(float("-1000"),float("1000")),(200,3500))
			
			zeta = shgo(self.obj_func_zeta_b2, bnds,
                       constraints=self.const_zeta, n=128, iters=2,
                       sampling_method='sobol',
                       minimizer_kwargs={
                           "method": "SLSQP",
                           "options": {
                               "maxiter": 50,
                               "ftol":    1e-7,
                               "maxfun":  100,
                           }
                       })

		else:
			con5 = {'type': 'ineq', 'fun': self.cons_T}
			self.const_zeta = [con5]
			bnds = ((float("-inf"),float("inf")),(float("-inf"),float("inf")),(float("-inf"),float("inf")),(200,3500))
			zeta = minimize(self.obj_func_zeta_b2,self.guess_z2,method="SLSQP",constraints=self.const_zeta,bounds=bnds)
		return [zeta.x[0:-1][0],zeta.x[0:-1][1],zeta.x[0:-1][2]]
		
	def obj_get_kappa(self,guess):
		cov = self.L
		opt_kappa = self.kappa
		QtLZ = self.kappa_0 + self.theta_for_kappa.T.dot(cov.dot(guess))
		f = opt_kappa - QtLZ
		obj = np.dot(f,f)
		return obj
	def _obj_delta_n(self,z):
			f = ((self.kappa_d_n-self.kappa_0)-np.array(self.theta_for_kappa.T.dot(self.L.dot(z))).flatten())
			obj = np.dot(f,f)
			return obj

	def constraint_d_n(self,z):
		return 2 - np.array(self.L.dot(z)).flatten()[1]	
	
	
	def getZeta_typeA(self,kappa):
		T = np.array([self.temperatures[0],(self.temperatures[0]+self.temperatures[-1])/2,self.temperatures[-1]])
		self.theta_for_kappa = np.array([T/T,np.log(T),-1/T])
		P = self.ArrheniusParams
		#print(P)
		self.kappa_0 = self.theta_for_kappa.T.dot(P)
		self.kappa_d_n = kappa
		#print(self.kappa_0)
		"""
		Constrained zeta
		"""
		guess = np.array([10,1,10])
		con1 = {'type': 'ineq', 'fun': self.constraint_d_n}
		cons = [con1]
		fit = minimize(self._obj_delta_n,guess,constraints=cons)
		zeta = fit.x
		
		"""
		Un-constrained zeta
		"""
		
		return zeta
	def getZeta_typeB(self,kappa):
		T = np.array([self.temperatures[0],(self.temperatures[0]+self.temperatures[-1])/2,self.temperatures[-1]])
		self.theta_for_kappa = np.array([T/T,np.log(T),-1/T])
		P = self.ArrheniusParams
		#print(P)
		self.kappa_0 = self.theta_for_kappa.T.dot(P)
		self.kappa_d_n = kappa
	
	def getZetaFromGen(self,generator):
		P = self.ArrheniusParams
		self.cov = self.getCovariance()
		self.zeta = self.getUnCorrelated(flag=False)
		self.unsrtFunc = self.getUncertFunc(self.cov)
		self.zetaUnsrt = self.getZetaUnsrtFunc(self.cov,self.zeta.x)
		zeta = self.zeta.x
		self.P_max = P + np.asarray(np.dot(self.cov,zeta)).flatten();
		self.P_min = P - np.asarray(np.dot(self.cov,zeta)).flatten();
		self.kmax = self.Theta.T.dot(self.P_max)
		self.kmin = self.Theta.T.dot(self.P_min)
		self.kappa = self.Theta.T.dot(P)
		#self.zeta_curved_type_B,index,zeta_lim = self.getConstrainedUnsrtZeta(flag=True)
		#self.zeta_curved_type_C = self.getConstrainedUnsrtZeta(flag=False)
		#self.zeta_curved_type_B1 = self.getConstrainedUnsrtZeta_typeB1(flag=True)
		self.generator = []
		self.samples = []
		
		self.kleft_fact = generator[0]
		self.kright_fact = generator[1]
		#self.kmiddle_fact = generator[2]
		zeta_B2 = self.getB2Zeta(flag=True)
		return zeta_B2
		
	def getExtremeCurves(self,tag,zeta_type,sample_points):
		P = self.ArrheniusParams
		self.cov = self.getCovariance()
		self.zeta = self.getUnCorrelated(flag=False)
		self.unsrtFunc = self.getUncertFunc(self.cov)
		self.zetaUnsrt = self.getZetaUnsrtFunc(self.cov,self.zeta.x)
		zeta = self.zeta.x
		self.P_max = P + np.asarray(np.dot(self.cov,zeta)).flatten();
		self.P_min = P - np.asarray(np.dot(self.cov,zeta)).flatten();
		self.kmax = self.Theta.T.dot(self.P_max)
		self.kmin = self.Theta.T.dot(self.P_min)
		self.kappa = self.Theta.T.dot(P)
		#self.zeta_curved_type_B,index,zeta_lim = self.getConstrainedUnsrtZeta(flag=True)
		#self.zeta_curved_type_C = self.getConstrainedUnsrtZeta(flag=False)
		#self.zeta_curved_type_B1 = self.getConstrainedUnsrtZeta_typeB1(flag=True)
		self.generator = []
		self.samples = []
		for i in range(int(sample_points)):
			a1 =2*np.random.random_sample(1)-1
			a2 = 2*np.random.random_sample(1)-1
			#a3 = 2*np.random.random_sample(1)-1
			#a3 = [1.0]
			#a1 =np.random.uniform(-1,1,1)
			#a2 = np.random.uniform(-1,1,1)
			#a3 = np.random.uniform(-1,1,1)
			
			generator.append([a1[0],a2[0]])
			self.kleft_fact = a1[0]
			self.kright_fact = a2[0]
			self.kmiddle_fact = abs(a1[0])
			zeta_B2 = self.getB2Zeta(flag=True)
			self.samples.append(zeta_B2)
		return self.zeta_B2  
	
	def getExtremeCurves_fast(self,sample_points):
		X = workers(100)
		zeta_list = X.do_job_async(self.data,sample_points)
		del X
		#print(self.parallized_zeta)
		return zeta_list
		
	def getUncorreationMatrix(self,tag):
		T = self.temperatures
		P = self.ArrheniusParams
		self.cov = self.getCovariance()
		self.zeta = self.getUnCorrelated(flag=False)
		self.unsrtFunc = self.getUncertFunc(self.cov)
		self.zetaUnsrt = self.getZetaUnsrtFunc(self.cov,self.zeta.x)
		self.P_max = P + np.asarray(np.dot(self.cov,self.zeta.x)).flatten();
		self.P_min = P - np.asarray(np.dot(self.cov,self.zeta.x)).flatten();
		self.kmax = self.Theta.T.dot(self.P_max)
		self.kmin = self.Theta.T.dot(self.P_min)
		self.kappa = self.Theta.T.dot(P)
		self.zeta_curved_type_B,index,zeta_lim = self.getConstrainedUnsrtZeta(flag=True)
		self.zeta_curved_type_C = self.getConstrainedUnsrtZeta(flag=False)
		self.kright_fact = -1
		self.kleft_fact = 1
		#self.zeta_class = self.getC2Zeta(flag=True)
		
		zeta = np.array([[self.zeta.x[0],self.zeta.x[1],self.zeta.x[2]],[-self.zeta_curved_type_B[0][0],-self.zeta_curved_type_B[0][1],-self.zeta_curved_type_B[0][2]],[self.zeta_curved_type_C.x[0],self.zeta_curved_type_C.x[1],self.zeta_curved_type_C.x[2]]])
		self.zeta_matrix = np.matrix(zeta)
		
		
		"""
		fig, axs = plt.subplots(2, 1, figsize=(15,20))

		for zeta in self.zeta_curved_type_A:
			temp_unsrtFunc = np.asarray([self.M*np.dot(i.T,np.dot(self.cov,zeta)) for i in self.Theta.T])
			axs[0].plot(self.temperatures,temp_unsrtFunc,'k--')
			axs[0].plot(self.temperatures,-temp_unsrtFunc,'k--')
			
		axs[0].set_xlabel('Temperature (K)')
		axs[0].set_ylabel('Uncertainity ($f$)')
		temp_unsrtFunc = np.asarray([np.dot(i.T,np.dot(self.cov,self.zeta.x)) for i in self.Theta.T])
		temp_unsrtFunc_C = np.asarray([np.dot(i.T,np.dot(self.cov,self.zeta_curved_type_C.x)) for i in self.Theta.T])

		axs[0].plot(self.temperatures,temp_unsrtFunc_C,'y--')
		axs[0].plot(self.temperatures,-temp_unsrtFunc_C,'y--')
		axs[0].plot(self.temperatures,self.unsrtFunc,'r--',label='present study (MUQ)')
		axs[0].plot(self.temperatures,-self.unsrtFunc,'r--')
		axs[0].plot(self.temperatures,self.zetaUnsrt,'b-',label='present study (zeta) (MUQ)')
		axs[0].plot(self.temperatures,-self.zetaUnsrt,'b-')
		axs[0].set_ylim(-2*max(self.unsrtFunc),2*max(self.unsrtFunc))
		axs[0].plot(self.temperatures,self.uncertainties,'go',label='Exp. data')
		plt.savefig(f"{tag}.pdf",bbox_inches="tight")
		"""
		
		return self.zeta_matrix,P,self.P_max,self.P_min,self.cov

	# ═══════════════════════════════════════════════════════════════════
	# Partial-parameter SAC  —  instance methods
	# All maths is self-contained.  Methods are instance methods so they
	# can be called as self._psac_*() from getClassB_partial etc. and also
	# from module-level multiprocessing workers via A._psac_*().
	# Pure-math helpers take explicit T/L_r/indices arguments so that the
	# Worker can instantiate a fresh UncertaintyExtractor, call
	# getCovariance + getUnCorrelated once, and then dispatch directly.
	# ═══════════════════════════════════════════════════════════════════

	_PSAC_MAX_DELTA_N = 2.0   # physical |Δn| bound (class-level constant)

	# ── basis / prior-uncertainty helpers ──────────────────────────────
	def _psac_theta_full(self, T_val):
		"""Full Arrhenius basis [1, ln T, -1/T] at scalar T."""
		return np.array([1.0, np.log(float(T_val)), -1.0 / float(T_val)])

	def _psac_theta_S(self, T_val, indices):
		"""Reduced basis vector for selected indices at scalar T."""
		return self._psac_theta_full(T_val)[list(indices)]

	def _psac_fp_S(self, T_val, L_r, indices):
		"""f_prior,S(T) = ||L_r^T θ_S(T)||_2 at scalar T."""
		return float(np.linalg.norm(L_r.T @ self._psac_theta_S(T_val, indices)))

	def _psac_fp_S_vec(self, T_arr, L_r, indices):
		"""Vectorised f_prior,S over a temperature array."""
		return np.array([self._psac_fp_S(t, L_r, indices) for t in T_arr])

	def _f_prior(self, T, L_full):
		"""f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂ for each T in array."""
		thS = self._psac_theta_full(T)
		return np.array([np.linalg.norm(L_full.T @ col) for col in thS.T])
	
	def delta_kappa(self,T, L_r, zeta_r, indices):
		"""Δκ_S(T) = Θ_S(T)^T L_r zeta_r."""
		return self._psac_theta_S(T, indices).T @ (L_r @ zeta_r)
	
	def _psac_dtheta_S_dT(self, T_val, indices):
		"""Analytical dθ_S/dT.  d/dT: 1→0, ln T→1/T, -1/T→1/T²."""
		return np.array([0.0, 1.0 / float(T_val), 1.0 / float(T_val)**2])[list(indices)]

	def _psac_fp_S_deriv(self, T_val, L_r, indices):
		"""Analytical df_prior,S/dT = (L_r^T θ_S)·(L_r^T dθ_S/dT) / f_prior,S."""
		LTth  = L_r.T @ self._psac_theta_S(T_val, indices)
		LTdth = L_r.T @ self._psac_dtheta_S_dT(T_val, indices)
		fp	= np.linalg.norm(LTth)
		return float(LTth @ LTdth) / fp if fp > 1e-30 else 0.0

	def _psac_dk_S(self, T_arr, L_r, zeta_r, indices):
		"""∆κ_S(T) = θ_S(T)^T L_r ζ_r over temperature array."""
		Lz = L_r @ zeta_r
		return np.array([float(self._psac_theta_S(t, indices) @ Lz) for t in T_arr])

	def _psac_has_sign_change(self, arr):
		"""Return True if arr changes sign at least once."""
		return bool(np.any(np.diff(np.sign(arr)) != 0))

	def _psac_enforce_dn(self, zeta_r, L_r, indices):
		"""Post-hoc scale ζ_r so |Δn| < _PSAC_MAX_DELTA_N when n is active."""
		if 1 not in indices:
			return zeta_r
		pos_n = list(indices).index(1)
		dn	= abs((L_r @ zeta_r)[pos_n])
		if dn > self._PSAC_MAX_DELTA_N:
			zeta_r = zeta_r * (self._PSAC_MAX_DELTA_N / (dn + 1e-30)) * 0.95
		return zeta_r

	# ── covariance helpers ──────────────────────────────────────────────
	def _psac_get_reduced_L(self, L_full, indices):
		"""
		Build the reduced Cholesky L_r from the principal submatrix of Σ.
		NEVER slice L_full directly.  Always: Σ = L L^T → Σ_r → chol(Σ_r).
		Returns (Σ_r, L_r).
		"""
		from scipy.linalg import cholesky as _chol
		Sigma   = L_full @ L_full.T
		idx	 = list(indices)
		Sigma_r = Sigma[np.ix_(idx, idx)]
		try:
			L_r = _chol(Sigma_r, lower=True)
		except Exception:
			eps = 1e-12 * max(float(np.trace(Sigma_r)), 1e-10) / max(len(idx), 1)
			L_r = _chol(Sigma_r + eps * np.eye(len(idx)), lower=True)
		return Sigma_r, L_r

	def _psac_reconstruct_full(self, zeta_r, indices, full_size=3):
		"""Embed reduced ζ_r into a full-length vector (zeros at inactive)."""
		zeta_full = np.zeros(full_size)
		for local_i, global_i in enumerate(indices):
			zeta_full[global_i] = zeta_r[local_i]
		return zeta_full

	# ── linear system solvers ───────────────────────────────────────────
	def _psac_solve_2x2(self, T1, T2, rhs1, rhs2, L_r, indices):
		"""Solve the 2×2 system (L_r^T θ_S at T1, T2) · ζ_r = [rhs1, rhs2]."""
		r1 = L_r.T @ self._psac_theta_S(T1, indices)
		r2 = L_r.T @ self._psac_theta_S(T2, indices)
		A  = np.vstack([r1, r2])
		if abs(np.linalg.det(A)) < 1e-14:
			return None
		try:
			return np.linalg.solve(A, np.array([rhs1, rhs2]))
		except np.linalg.LinAlgError:
			return None

	def _psac_solve_3x3(self, T_min, T_u, L_r, r1_fp, r3_fp, r3_fpd, indices):
		"""
		Solve the 3×3 system (C1 + C3 + C4) for ζ_r at a fixed T_u.
		Rows: [L_r^T θ_S(T_min)],  [L_r^T θ_S(T_u)],  [L_r^T dθ_S/dT(T_u)].
		"""
		row1 = L_r.T @ self._psac_theta_S(T_min, indices)
		row2 = L_r.T @ self._psac_theta_S(T_u,   indices)
		row3 = L_r.T @ self._psac_dtheta_S_dT(T_u, indices)
		A	= np.vstack([row1, row2, row3])
		b	= np.array([r1_fp, r3_fp, r3_fpd])
		if abs(np.linalg.det(A)) < 1e-14:
			return None
		try:
			return np.linalg.solve(A, b)
		except np.linalg.LinAlgError:
			return None

	# ── curve-type samplers ─────────────────────────────────────────────
	def _psac_class_A(self, T_arr,L_full, L_r, indices, rng, n_samples,max_attempts=10):
		"""
		Generate `n_samples` Class-A zeta_r vectors for the given parameter subset,
		with rejection sampling to enforce the fp uncertainty bounds at all temperatures.

		Algorithm
		---------
		1.  Compute f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂  for all T.
		2.  Form A_mat = Θ_S(T)^T L_r  (overdetermined system, shape N×m).
		3.  For each sample:
			a. Draw α ~ U(0, 1), draw sign ∈ {-1, +1}.
			   Set target b = sign·α·f_prior_S(T).
			   Solve zeta_r = pinv(A_mat) · b  (least-squares fit).
			b. Apply Δn constraint.
			c. Reject if |A_mat @ zeta_r| > fp at ANY temperature; retry.
		4.  Raise RuntimeError if max_attempts is exceeded for any single sample.

		Parameters
		----------
		T			: temperature array (K), shape (N,)
		L_full	   : full (3,3) Cholesky factor, used to compute fp
		L_r		  : (m, m) reduced Cholesky factor
		indices	  : tuple of active parameter indices, e.g. (0, 1) for (A, n)
		n_samples	: number of valid samples to generate
		rng		  : numpy Generator (created fresh if None)
		max_attempts : max draws per sample before raising an error (default 10)

		Returns
		-------
		zeta_list : list of n_samples zeta_r arrays, each shape (m,)
		"""
		
		fp	 = self._f_prior(T, L_full)
		thS	= np.array([self._psac_theta_S(t, indices) for t in T_arr])  # (N,m)
		A_mat  = thS @ L_r
		A_pinv = np.linalg.pinv(A_mat)								   # (m,N)
		out	= []
		for _ in range(n_samples):
			accepted = False
			for attempt in range(max_attempts):
				alpha_s = rng.uniform(0.00, 1.0)
				sign	= rng.choice([-1.0, 1.0])
				zr	  = A_pinv @ (sign * alpha_s * fp)
				zr	   = self._psac_enforce_dn(zr, L_r, indices)
				
				curve = A_mat @ zr					  # shape (N,)
				if np.all(np.abs(curve) <= fp):			 # passes at every T
					accepted = True
					break
			if not accepted:
				raise RuntimeError(
					f"Class-A sample {i}: could not find a valid sample within "
					f"{max_attempts} attempts. "
					f"Check L_r / indices consistency or increase max_attempts."
				)
			
			out.append(zr)
		return out

	def _psac_class_D_m2(self, T_arr, L_full, L_r, indices, rng, n_samples):
		"""
		Class-D for m=2: 2×2 solve at each candidate T_u, bisect on the
		C4 tangency residual g(T_u) = dΔκ/dT|_{T_u} - r3·f'_prior,S(T_u).
		Falls back to SLSQP when no sign change is found (flat f_prior).
		Returns list of n_samples ζ_r arrays.
		"""
		
		idx = list(indices)
		_, _, L_r = self._psac_get_reduced_L(L_full, idx)   # (m, m)

		T_min = float(T_arr[0])
		T_max = float(T_arr[-1])

		# f_prior from full L at all temperatures
		thfull	   = self._psac_theta_full(T_arr)		   # (3, N)
		LT_th		= L_full.T @ thfull				 # (3, N)
		f_prior_vals = np.linalg.norm(LT_th, axis=0)	 # (N,)
		fp_min	   = f_prior_vals[0]
		fp_max	   = f_prior_vals[-1]

		# Reduced theta at endpoints — extract rows manually
		thfull_min = self._psac_theta_full(np.array([T_min]))[:, 0]  # (3,)
		thfull_max = self._psac_theta_full(np.array([T_max]))[:, 0]  # (3,)
		thS_min	= thfull_min[idx]					   # (m,)
		thS_max	= thfull_max[idx]					   # (m,)

		# Build M (2x2) — computed once
		M = np.stack([
		   thS_min @ L_r,	# (m,) @ (m,m) = (m,)
		   thS_max @ L_r	 # (m,) @ (m,m) = (m,)
		], axis=0)			 # (2, m) = (2, 2)

		if abs(np.linalg.det(M)) < 1e-12:
		   raise ValueError(
			  f"M is singular for indices={indices}."
		   )

		M_inv = np.linalg.inv(M)			  # (2, 2), once

		results = []
		for k in range(n_samples):
		   accepted = False
		   for attempt in range(200):
			  r1 = float(rng.uniform(-1.0, 1.0))
			  r2 = float(rng.uniform(-1.0, 1.0))

			  d	  = np.array([r1 * fp_min,
								r2 * fp_max])
			  zeta_r = M_inv @ d				 # (2,)

			  dk = self.delta_kappa(
				 T_arr, L_r, zeta_r, indices
			  )

			  if np.all(
				 np.abs(dk) <= f_prior_vals + 1e-10
			  ):
				 accepted = True
				 break

		   if not accepted:
			  print(
				 f"  [!] Sample {k}: validation failed "
				 f"after 200 attempts for "
				 f"indices={indices}."
			  )

		   results.append(
			  self._psac_enforce_dn(
				 np.asarray(zeta_r), L_r, indices
			  )
		   )
		return results

	def _psac_class_B_m3_(self, T_arr, L_full, L_r, indices, rng, n_samples):
		"""
		Class-B for m=3: direct 3×3 linear solve (C1+C3+C4) at each T_u.
		Scans a shuffled 80-point grid and keeps the first solution whose
		Δκ_S changes sign.  >10^4× faster than SLSQP.
		Returns list of n_samples ζ_r arrays.
		"""
		T_min   = float(T_arr[0]);   T_max = float(T_arr[-1])
		Tu_grid = np.linspace(T_min * 1.04, T_max * 0.96, 80)
		out	 = []
		attempt = 0
		while len(out) < n_samples and attempt < n_samples * 30:
			attempt += 1
			r1 = rng.uniform(-0.99, 0.99)
			r3 = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
			for Tu in rng.permutation(Tu_grid):
				zr = self._psac_solve_3x3(
					T_min, float(Tu), L_r,
					r1 * self._psac_fp_S(T_min, L_r, indices),
					r3 * self._psac_fp_S(float(Tu), L_r, indices),
					r3 * self._psac_fp_S_deriv(float(Tu), L_r, indices),
					indices)
				if zr is None:
					continue
				if self._psac_has_sign_change(self._psac_dk_S(T_arr, L_r, zr, indices)):
					out.append(self._psac_enforce_dn(zr, L_r, indices))
					break
		return out
	
	def _psac_class_B_m3(self, data, T_arr, L_full, L_r, indices, rng, n_samples):
		"""
		Generate one Class-B sample using the ORIGINAL published SHGO/SLSQP
		method for m = 3  (all three Arrhenius parameters A, n, Ea active).

		Constraints C1–C4 are enforced on the full-space Δκ using L_full.
		SHGO is used as the primary solver; SLSQP is the fallback.

		Uncertainty bounds are enforced at two levels:
		  Layer 1 (optimizer)	   — inequality constraints |Δκ(Tᵢ)| ≤ uᵢ on a
									  20-point temperature grid, active during SHGO/SLSQP.
		  Layer 2 (post-processing) — radial scaling of zr if any residual violation
									  survives after _enforce_dn_constraint.

		This is the method published in the MUQ-SAC paper, extended with bound
		enforcement.

		Parameters
		----------
		nominal	   : (3,) nominal log-rate Arrhenius parameter vector
		T_arr        : (N,) array of temperatures in K
		L_full	   : (3, 3) lower-triangular Cholesky factor from MUQ
		uncertainties : (N,) array of uncertainty values
		indices	   : must be (0, 1, 2) for this method
		sub_dir	   : Path object for animation output (currently commented out)
		n_samples	 : number of samples to generate
		rng		   : numpy Generator for random r1, r2 draws

		Returns
		-------
		zeta_list : list of zeta_r arrays, each shape (3,)
		"""
		T_min	= float(T_arr[0])
		T_max	= float(T_arr[-1])
		L		= L_full
		zeta_unc = _compute_uncorrelated_direction(L, T_arr, uncertainties)

		# ── helper closures: rate curve values and derivatives ────────────────
		def dk_unc_at_T(Tv):
			return float(theta_full(np.array([Tv]))[:, 0] @ L @ zeta_unc)

		def dk_z_at_T(Tv, z):
			return float(theta_full(np.array([Tv]))[:, 0] @ L @ z)

		def ddk_unc_at_T(Tv):
			dth = np.array([0.0, 1.0 / Tv, 1.0 / Tv ** 2])
			return float(dth @ L @ zeta_unc)

		def ddk_z_at_T(Tv, z):
			dth = np.array([0.0, 1.0 / Tv, 1.0 / Tv ** 2])
			return float(dth @ L @ z)

		def mismatch_objective(z):
			Theta = np.array([T_arr / T_arr,
							  np.log(T_arr),
							  -1.0 / T_arr])
			QtLZ  = np.array([th @ L @ z[:3] for th in Theta.T])
			obj   = float(np.dot(uncertainties - QtLZ, uncertainties - QtLZ))
			return obj

		# ── nominal curve (full 3-param theta) ────────────────────────────────
		Theta_full = np.array([T_arr / T_arr,
							   np.log(T_arr),
							   -1.0 / T_arr])
		QtPo = np.array([th @ nominal for th in Theta_full.T])


		# ── Layer 1: build bound-inequality constraints (computed once) ────────
		# These enforce |Δκ(Tᵢ)| ≤ uᵢ on a 20-point grid inside the optimizer.
		bound_constraints = _build_bound_constraints(
			T_arr, uncertainties, dk_z_at_T, n_bound_pts=20
		)

		zeta_list = []
		for _ in range(n_samples):
			r1 = float(rng.uniform(-1.0, 1.0))
			r2 = float(rng.uniform(-1.0, 1.0))
			sign_C2, sign_C4, kmiddle = _determine_constraint_signs(r1, r2)

			# ── per-sample anchor points ───────────────────────────────────────
			_dk_min		= dk_unc_at_T(T_min)
			_dk_max		= dk_unc_at_T(T_max)
			idx_min		= int(np.argmin(np.abs(temperatures - T_min)))
			idx_max		= int(np.argmin(np.abs(temperatures - T_max)))

			# ── equality constraints C1–C4 ────────────────────────────────────
			def c1(z): return r1 * dk_unc_at_T(T_min) - dk_z_at_T(T_min, z[:3])
			def c3(z): return r2 * dk_unc_at_T(T_max) - dk_z_at_T(T_max, z[:3])
			def c2(z):
				Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
				return sign_C2 * kmiddle * dk_unc_at_T(Tu) - dk_z_at_T(Tu, z[:3])
			def c4(z):
				Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
				return sign_C4 * kmiddle * ddk_unc_at_T(Tu) - ddk_z_at_T(Tu, z[:3])

			# ── full constraint list: C1–C4 + bound inequalities ──────────────
			constraints = [
				{'type': 'eq', 'fun': c1},
				{'type': 'eq', 'fun': c2},
				{'type': 'eq', 'fun': c3},
				{'type': 'eq', 'fun': c4},
			] + bound_constraints						  # ← Layer 1 appended

			bounds = [(-1000, 1000)] * 3 + [(200, 3500)]
			x0	 = np.array([10, 10, 100, (T_min + T_max) / 2])

			t_sample = time.perf_counter()
			try:
				sol = shgo(mismatch_objective, bounds,
						   constraints=constraints, n=128, iters=2,
						   sampling_method='sobol',
						   minimizer_kwargs={
							   "method": "SLSQP",
							   "options": {
								   "maxiter": 50,
								   "ftol":	1e-7,
								   "maxfun":  100,
							   }
						   })
				zr = sol.x[:3]
			except Exception:
				try:
					sol = minimize(mismatch_objective, x0, method='SLSQP',
								   bounds=bounds, constraints=constraints,
								   options={'maxiter': 2000, 'ftol': 1e-9})
					zr = sol.x[:3]
				except Exception:
					zr = x0[:3]

			elapsed = time.perf_counter() - t_sample
			if elapsed > ORIG_TIME_CAP_S:
				print(f"	  [!] original_m3 sample took {elapsed:.1f}s "
					  f"(cap={ORIG_TIME_CAP_S}s)")

			# ── _enforce_dn_constraint (existing, preserves ||zeta|| norm) ────
			zr_dn = _enforce_dn_constraint(np.asarray(zr), L_full, (0, 1, 2))

			# ── Layer 2: post-processing bound safety net ─────────────────────
			# Radially scales zr_dn if any temperature point still violates
			# |Δκ(T)| ≤ uncertainty(T) after the optimizer + dn constraint.
			zr_final, max_ratio = _enforce_uncertainty_bounds(
				zr_dn, L_full, temperatures, uncertainties
			)
			if max_ratio > 1.0:
				print(f"	  [!] sample {_:03d}: residual bound violation "
					  f"ratio = {max_ratio:.4f} — scaled down in post-processing.")

			zeta_list.append(zr_final)
		return zeta_list
	
	def _psac_class_C(self, T_arr, L_r, indices, rng, n_samples):
		"""
		Class-C: direct least-squares fit to a linear crossing ramp target.

		f_c(T) = r1·f_prior_S(T_min) + [r2·f_prior_S(T_max) - r1·f_prior_S(T_min)]
				                 × (T - T_min) / (T_max - T_min)

		zeta_r = pinv(Θ_S^T L_r) · f_c(T)  — single closed-form solve per attempt,
		no iterative optimisation.  The recorded "objective" is the LS residual
		||f_c - Θ_S^T L_r zeta_r||² (should be near-zero for well-posed subsets).

		Parameters
		----------
		nominal       : (3,) nominal Arrhenius parameters (for viz baseline)
		T_arr  : (N,) temperature array in K
		L_full        : (3, 3) full Cholesky factor
		uncertainties : (N,) uncertainty array  (used for viz band only)
		indices       : tuple of active parameter indices (must have len >= 2)
		sub_dir       : Path-like, directory for per-sample animation output
		n_samples     : number of accepted samples to generate
		rng           : numpy Generator

		Returns
		-------
		zeta_list : list of zeta_r arrays, each shape (m,)
		"""
		
		T_min   = float(T_arr[0]);  T_max = float(T_arr[-1])
		inv_T     = 1.0 / T_arr
		inv_T_min = 1.0 / T_max    # note: T_max → smallest 1/T
		inv_T_max = 1.0 / T_min    # note: T_min → largest 1/T
		
		thfull	   = self._psac_theta_full(T_arr)		   # (3, N)
		LT_th		= L_full.T @ thfull				 # (3, N)
		f_prior_vals = np.linalg.norm(LT_th, axis=0)	 # (N,)
		fp_Tmin	   = f_prior_vals[0]
		fp_Tmax	   = f_prior_vals[-1]
		thS = np.array([self._psac_theta_S(t, indices) for t in T_arr])
		A_mat  = thS.T @ L_r                      # (N, m)  =  Φ_S L_r
		A_pinv = np.linalg.pinv(A_mat)
		zeta_list = []
		sample_idx = 0
		attempt    = 0
		while len(zeta_list) < n_samples and attempt < n_samples * 5:
			attempt += 1

			r1 = float(rng.uniform(-1.0, 1.0))
			r2 = float(rng.uniform(-1.0, 1.0))
			if np.sign(r1) == np.sign(r2):
				r2 = -r2

			# Build ramp target
			#fc = (r1 * fp_Tmin
			#      + (r2 * fp_Tmax - r1 * fp_Tmin) / (T_max - T_min)
			#      * (temperatures - T_min))           # (N,)

			fc = (r1 * fp_Tmin
			  + (r2 * fp_Tmax - r1 * fp_Tmin) / (inv_T_max - inv_T_min)
			  * (inv_T - inv_T_min))
			# Closed-form solve
			zeta_r = A_pinv @ fc                      # (m,)

			# LS residual — what the right panel tracks
			residual = fc - A_mat @ zeta_r            # (N,)
			obj      = float(np.dot(residual, residual))


			# ── accept / reject ───────────────────────────────────────────
			dk = delta_kappa(T_arr, L_r, zeta_r, indices)
			if not _has_sign_change(dk):
				continue
			
			zeta_r = self._psac_enforce_dn(
				 np.asarray(zeta_r), L_r, indices
			  )

			zeta_list.append(zeta_r)
			
		return zeta_list[:n_samples]

	def _psac_fsac(self, T_arr, L_r, indices, rng, n_samples):
		"""
		Fast SAC for partial parameters.
		m=1 → class-A (f-SAC is undefined for a single parameter).
		m=2 → 2×2 system at T_min + T_mid.
		m=3 → 3×3 system at T_min + T_mid + T_max.
		Midpoint value drawn to ensure opposite-sign anchors → crossover.
		Returns list of n_samples ζ_r arrays.
		"""
		T_min = float(T_arr[0]);  T_max = float(T_arr[-1])
		T_mid = 0.5 * (T_min + T_max)
		m	 = len(indices)
		fp_min = self._psac_fp_S(T_min, L_r, indices)
		fp_mid = self._psac_fp_S(T_mid, L_r, indices)
		fp_max = self._psac_fp_S(T_max, L_r, indices)
		out = []
		for _ in range(n_samples * 8):
			if len(out) >= n_samples:
				break
			r1	= rng.uniform(-1.0, 1.0)
			r2	= -r1 * rng.uniform(0.3, 1.0)
			r_mid = rng.uniform(-1.0, 1.0) * np.sign(r2)
			kl, km, kr = r1 * fp_min, r_mid * fp_mid, r2 * fp_max
			if m == 1:
				sub = self._psac_class_A(T_arr, L_r, indices, rng, 1)
				if sub:
					out.append(sub[0])
				continue
			elif m == 2:
				zr = self._psac_solve_2x2(T_min, T_mid, kl, km, L_r, indices)
			else:
				row1 = L_r.T @ self._psac_theta_S(T_min, indices)
				row2 = L_r.T @ self._psac_theta_S(T_mid, indices)
				row3 = L_r.T @ self._psac_theta_S(T_max, indices)
				A	= np.vstack([row1, row2, row3])
				if abs(np.linalg.det(A)) < 1e-14:
					continue
				try:
					zr = np.linalg.solve(A, np.array([kl, km, kr]))
				except np.linalg.LinAlgError:
					continue
			if zr is None:
				continue
			if m >= 2 and not self._psac_has_sign_change(
					self._psac_dk_S(T_arr, L_r, zr, indices)):
				continue
			out.append(self._psac_enforce_dn(zr, L_r, indices))
		return out[:n_samples]

	# ── public partial-parameter interface ─────────────────────────────

	# ── public partial-parameter interface ────────────────────────────

	def get_full_sigma_and_L(self):
		"""Return (Σ, L_full) from self.cov (3×3 Cholesky). 
		Requires getCovariance() to have been called first."""
		L	 = self.cov
		Sigma = L @ L.T
		return Sigma, L

	def get_reduced_cholesky(self, param_indices):
		"""
		Compute (Σ_r, L_r) via the principal submatrix of Σ = L L^T.
		param_indices : tuple/list of ints — 0=ln(A), 1=n, 2=ε=Ea/R.
		NEVER slices L directly; always reconstructs Σ first.
		"""
		_, L_full = self.get_full_sigma_and_L()
		return L_full, (self._psac_get_reduced_L(L_full, param_indices))

	def get_fsac_partial(self, param_indices, n_samples, rng=None):
		"""
		f-SAC sampling for a parameter subset.
		m=1 → class-A fallback.  m=2 → 2×2 system.  m=3 → 3×3 system.
		Returns list of full-space (length-3) ζ vectors.
		"""
		if rng is None:
			rng = np.random.default_rng()
		_, L_r  = self.get_reduced_cholesky(param_indices)
		zr_list = self._psac_fsac(self.temperatures, L_r, param_indices, rng, n_samples)
		return [self._psac_reconstruct_full(zr, param_indices) for zr in zr_list]

	def getClassA_partial(self, param_indices, n_samples, rng=None):
		"""Class-A samples for any subset (m=1,2,3). Returns full-space ζ vectors."""
		if rng is None:
			rng = np.random.default_rng()
		L_full, (_, L_r)  = self.get_reduced_cholesky(param_indices)
		zr_list = self._psac_class_A(self.temperatures, L_full, L_r, param_indices, rng, n_samples)
		return [self._psac_reconstruct_full(zr, param_indices) for zr in zr_list]

	def getClassB_partial(self, data, param_indices, n_samples, rng=None):
		"""
		Class-B samples for a parameter subset.
		m=1 → empty list (infeasible) Ultimately a class-A curves will be generated instead of 				 class-A curves.
		m=2 → Class-D curves will be generated
		m=3 → Class-B curves will be generated
		Returns full-space ζ vectors.
		"""
		#m = len(param_indices)
		#if m < 2:
		#	return []
		if rng is None:
			rng = np.random.default_rng()
		L_full, (_, L_r)  = self.get_reduced_cholesky(param_indices)
		#if m == 2:
		zr_list = self._psac_class_D_m2(self.temperatures, L_full, L_r, param_indices, rng, n_samples)
		#else:
		#	zr_list = self._psac_class_B_m3(data, self.temperatures, L_full, L_r, param_indices, rng, n_samples)
		return [self._psac_reconstruct_full(zr, param_indices) for zr in zr_list]

	def getClassC_partial(self, param_indices, n_samples, rng=None):
		"""
		Class-C samples for a parameter subset.
		m=1 → empty list (infeasible).
		m≥2 → linear-anchor pseudo-inverse LS.
		Returns full-space ζ vectors.
		"""
		if len(param_indices) < 2:
			return []
		if rng is None:
			rng = np.random.default_rng()
		_, L_r  = self.get_reduced_cholesky(param_indices)
		zr_list = self._psac_class_C(self.temperatures, L_r, param_indices, rng, n_samples)
		return [self._psac_reconstruct_full(zr, param_indices) for zr in zr_list]

	def reconstruct_full_zeta(self, zeta_r, param_indices, full_size=3):
		"""Embed reduced ζ_r back into a full-length vector (zeros at inactive)."""
		return self._psac_reconstruct_full(zeta_r, param_indices, full_size)

	# ── END NEW UncertaintyExtractor methods ──────────────────────────────────

class reaction(UncertaintyExtractor):
	def __init__(self, Element,mechPath,binary_files):
		#self.samap_executable = binary_files["samap_executable"]
		#self.jpdap_executable = binary_files["jpdap_executable"]
		self.rxn = self.classification = self.type = self.sub_type = self.exp_data_type = self.temperatures = self.uncertainties = self.branching  = self.branches = self.pressure_limit = self.common_temp = self.temp_limit = None
		self.selected = False
		self.linked_rIndex = None

		DATA = Parser(mechPath).mech
		RXN_LIST = Parser(mechPath).rxnList
		self.tag = Element.tag
		self.rxn = str(Element.attrib["rxn"])
		self.rIndex = str(Element.attrib["no"])
		#self.rxn_dict = IFR.MechParsing(mechPath).getKappa(self.rxn)
		#print(self.rIndex,IFR.MechParsing(mechPath).getArrhenius(self.rxn))
		if self.rxn in RXN_LIST:
			self.index = RXN_LIST.index(self.rxn)
		else:
			raise AssertionError(f"Rxn {self.rxn} not in the mechanism. Kindly check the uncertainty file that you have submitted !!\n")
		
		for item in Element:
			if item.tag == "class":
				self.classification = item.text
			if item.tag == "type":
				self.type = item.text
			if item.tag == "perturbation_type":
				self.perturbation_type = item.text
			if item.tag == "perturbation_factor":
				self.perturbation_factor = float(item.text)
			if item.tag == "sub_type":
				self.sub_type = item.attrib["name"]
				try:
					self.linked_rIndex = item.attrib["link"]
				except KeyError:
					self.linked_rIndex = None
				for subitem in item:
					if subitem.tag == "multiple":
						self.multiple = subitem.text.strip()
					if subitem.tag == "branching":
						self.branching = subitem.text
					if subitem.tag == "branches":
						self.branches = subitem.text
					if subitem.tag == "pressure_limit":
						self.pressure_limit = subitem.text
					if subitem.tag == "common_temp":
						self.common_temp = subitem.text
					if subitem.tag == "temp_limit":
						self.temp_limit = subitem.text
			
			if item.tag == "data_type":
				self.exp_data_type = item.text
			if item.tag == "file":
				self.exp_data_file = item.text
			if item.tag == "temp":
				#print(item.text)
				self.temperatures = np.asarray([float(i) for i in item.text.split(",")])
			if item.tag == "unsrt":
				self.uncertainties = np.asarray([float(i) for i in item.text.split(",")])
			
		if self.exp_data_type.split(";")[0] == "constant":
			if self.exp_data_type.split(";")[1] == "array":	
				self.temperatures = self.temperatures
				self.uncertainties = self.uncertainties
				
			elif self.exp_data_type.split(";")[1] == "end_points":
				self.temperatures = np.linspace(self.temperatures[0],self.temperatures[1],200)
				self.uncertainties = np.linspace(self.uncertainties[0],self.uncertainties[1],200)
		elif self.exp_data_type.split(";")[0] == "file":
			unsrt_file = open(str(self.exp_data_file),"r").readlines()
			unsrtData = [np.asfarray(i.strip("\n").strip("''").split(","),float) for i in unsrt_file]
			self.temperatures = np.asarray([i[0] for i in unsrtData])
			self.uncertainties = np.asarray([i[1] for i in unsrtData])
		
		if len(self.temperatures) != len(self.uncertainties):
			print("Error in unsrt data for {}".format(self.rxn))
	
		
		if self.type == "pressure_dependent" and self.pressure_limit.strip() != "":
			if self.pressure_limit == "High":
				self.rxn_Details = DATA["reactions"][self.index]
				#print()
				self.rxn_dict = self.rxn_Details["high-P-rate-constant"]
				self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
			else:
				self.rxn_Details = DATA["reactions"][self.index]
				self.rxn_dict = self.rxn_Details["low-P-rate-constant"]
				self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
			self.nametag = self.rxn+":"+self.pressure_limit
			
		elif self.type == "pressure_independent" and self.sub_type == "duplicate":
			if self.branches.strip() == "A":
				self.index = self.index
			else:
				self.index = self.index+1
			self.rxn_Details = DATA["reactions"][self.index]
			self.nametag = self.rxn+":"+self.branches
			self.rxn_dict = self.rxn_Details["rate-constant"]
			self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
		else:
			self.rxn_Details = DATA["reactions"][self.index]
			self.nametag = self.rxn
			self.rxn_dict = self.rxn_Details["rate-constant"]
			self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
		
		if self.branching == "True":
			self.branches = self.branches.strip('"').split(",")
			self.branches = [int(i)-1 for i in self.branches]
			#print(self.branches)
		#print(self.nametag)
		#print(self.classification)
		
		
		
		data = {}
		data["temperatures"] = self.temperatures
		data["uncertainties"] = self.uncertainties
		data["Arrhenius"] = self.nominal
		
		#uncertainty extractor
		super().__init__(data)
		self.zeta_Matrix,self.P,self.P_max,self.P_min,self.cov = self.getUncorreationMatrix(self.rIndex)
		self.solution = self.zeta
		self.cholskyDeCorrelateMat = self.L
		#print(self.rIndex,self.L.dot(self.L.T))
		self.activeParameters = [self.rIndex+'_A',self.rIndex+'_n',self.rIndex+'_Ea']
		self.perturb_factor = self.zeta.x
		self.selection = [1.0,1.0,1.0]
		#print(self.zeta.x)
		#print(self.L)
		"""
		if "JPDAP" not in os.listdir():
			os.mkdir("JPDAP")
			os.chdir("JPDAP")
			print(f"{self.rIndex}")
			start = time.time()
			self.input_dict = self.getJPDAP()
			stop = time.time()
			print(f"{stop-start}")
		else:	
			os.chdir("JPDAP")
			if str(self.rIndex) in os.listdir():
				print(f"{self.rIndex}")
				print("Uncertainty_analysis is done!!")
				os.chdir(str(self.rIndex))
				self.input_dict = self.readJPDAP()
			else:
				print(f"{self.rIndex}")
				start = time.time()
				self.input_dict = self.getJPDAP()
				stop = time.time()
				print(f"{stop-start}")
			
		os.chdir("..")
		"""
		#print(f"{self.rIndex}")
		#print(f"{self.cholskyDeCorrelateMat}")
		#print(f"{self.zeta.x}")
		if "factor" in self.perturbation_type:
			#self.perturb_factor =  [min(self.uncertainties),0,0]
			#self.solution = 1.0
			#self.getUncorreationMatrix = np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
			#self.cholskyDeCorrelateMat = np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
			#self.zeta_Matrix = 1
			#self.activeParameters = [self.rIndex+'_A',self.rIndex+'_n',self.rIndex+'_Ea']
			#self.selection = [1.0,0.0,0.0]
			self.perturb_factor =  [min(self.uncertainties)]
			self.solution = 1.0
			#self.getUncorreationMatrix = np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
			self.cholskyDeCorrelateMat = np.array([min(self.uncertainties)])
			self.zeta_Matrix = 1
			self.activeParameters = [self.rIndex+'_A']
			self.selection = [1.0,0.0,0.0]
		#print(f"{self.rIndex}= {self.zeta_Matrix}")
		"""
		file_unsrt = open("Reaction_detail_nominal.csv","a+")
		string_rates = f"{self.rIndex},"
		for i in self.rxn_dict:
			string_rates+=f"{i},"
		string_rates+="\n"
		file_unsrt.write(string_rates)
		file_unsrt.close()
		
		file_mat = open("cholesky.csv","a+")
		string_cholesky = f"{self.rIndex},"
		for i in list(self.cholskyDeCorrelateMat):
			for j in i:
				string_cholesky+=f"{j},"
		string_cholesky+="\n"
		file_mat.write(string_cholesky)
		file_mat.close()
		
		file_zeta = open("rxn_zeta_data.csv","a+")
		string_zeta = f"{self.rIndex},"
		for i in self.zeta.x:
			string_zeta+=f"{i},"
		string_zeta+="\n"
		file_zeta.write(string_zeta)
		file_zeta.close()
		"""	
#public function to get the uncertainity values for discrete temperatures
	def getDtList(self):
		self.Unsrt_dict = {}
		key = ["Tag","Solution","Class","Type","Perturbation_type","Sub_type","Branch_boolen","Branches","Pressure_limit","Common_temp","temp_list","Nominal","Exp_input_data_type","priorCovariance","Perturb_factor","Basis_vector","Uncertainties","Temperatures","unsrt_func","Data_key"]
		values = [self.tag,self.solution,self.classification,self.type,self.perturbation_type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.rxn_dict,self.exp_data_type,self.cholskyDeCorrelateMat,self.perturb_factor,self.zeta.x,self.uncertainties,self.temperatures,self.unsrtFunc,""]
		for i,element in enumerate(key):
			self.Unsrt_dict[element] = values[i]
		return self.Unsrt_dict
	
	def getKappaMax(self,T):
		T = np.array([self.temperatures[0],(self.temperatures[0]+self.temperatures[-1])/2,self.temperatures[-1]])
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_max)).flatten()
	
	def extract_submatrix_with_vectors(self,L, x, y):
		"""
		Extracts a submatrix using x and y vectors based on their non-zero values.

		Parameters:
			L (ndarray): Cholesky decomposed covariance matrix (NumPy array).
			x (ndarray): Row selector vector (selection_matrix,non-zero indicates rows to include).
			y (ndarray): Column selector vector (Zeta,non-zero indicates columns to include).

		Returns:
			ndarray: The extracted submatrix.
		"""
		Sigma = L.dot(L.T)
		
		row_indices = np.nonzero(x)[0]  # Indices of non-zero elements in x
		col_indices = np.nonzero(y)[0]  # Indices of non-zero elements in y
		return Sigma,Sigma[np.ix_(row_indices, col_indices)]
	
	def perturbe_partial_values(self,po,L,x,zeta):
		Sigma,Sigma_reduce = self.extract_submatrix_with_vectors(L,x,zeta)
		p,Lr = self.process_with_cholesky(Sigma_reduce,po,zeta)
		return Sigma,Sigma_reduce,p,Lr
	
	def process_with_cholesky(self,cov_matrix_reduced, p_o, y):
		"""
		Performs Cholesky decomposition and updates p_o based on the reduced cov_matrix and y.

		Parameters:
			cov_matrix_reduced (ndarray): The reduced covariance matrix.
			p_o (ndarray): The initial vector with values for non-zero indices of y.
			y (ndarray): The vector with non-zero entries indicating the indices to update.

		Returns:
			ndarray: Updated vector p.
		"""
		# Cholesky decomposition of the reduced covariance matrix
		L_reduced = np.linalg.cholesky(cov_matrix_reduced)

		# Select p_o and y_ corresponding to non-zero indices of y
		non_zero_indices = np.nonzero(y)[0]
		p_o_reduced = np.copy(p_o)[non_zero_indices]
		y_reduced = y[non_zero_indices]

		# Perform the update p = p_o + L_reduced @ y_
		p_reduced = p_o_reduced + L_reduced @ y_reduced

		# Create the final vector p
		p = np.copy(p_o)
		p[non_zero_indices] = p_reduced

		return p,L_reduced

	
	def readJPDAP(self):
		input_dict = {}
		#input_dict["samples"] = self.sim_
		#input_dict["samples_skipped"] = int(0.1*self.sim_)
		#input_dict["Random_seed"] = 1
		#input_dict["sampling_method"] = "SOBOL"
		#input_dict["sampling_distribution"] = "NORMAL"
		#input_dict["equidistant_T"] = 100
		#input_dict["T_begin"] = data = self.rxnUnsert[i].temperatures[0]
		#input_dict["T_end"] = self.rxnUnsert[i].temperatures[-1]
		input_dict["L"] = 0
		input_dict["len_temp_data"] = len(self.temperatures)
		string_unsrt_data =""
		for index,k in enumerate(self.temperatures):
			string_unsrt_data+=f"{k} {self.uncertainties[index]} \n"
		input_dict["temperature_unsrt_data"] = string_unsrt_data
		input_dict["alpha"] = np.exp(self.rxn_dict[0])
		input_dict["n"] = self.rxn_dict[1]
		input_dict["n_min"] = self.rxn_dict[1]-2
		input_dict["n_max"] = self.rxn_dict[1]+2
		input_dict["epsilon"] = self.rxn_dict[2]
		L = self.cholskyDeCorrelateMat
		#input_dict["covariance_matrix"] = str(L.dot(L.T)).strip("[]").replace("[","").replace("]","")
		input_dict["uncertainty_type"] = "2slnk"
		if len(self.activeParameters) == 3:
			input_dict["uncertain_parameters"] = "AnE"
		else:
			input_dict["uncertain_parameters"] = "A"
		Nagy_covariance_matrix = ""
		file_name_jpdap = "jpdap_data_"+str(self.rIndex)+".txt_fit_minRMSD.txt"
		Nagy_covariance_matrix = open(file_name_jpdap,"r").readlines()[5:8]
		covariance_matrix = ""
		cov_float = []
		for n in Nagy_covariance_matrix:
			covariance_matrix+=str(n)
			cov_float.append(np.asfarray(n.strip("''").strip("\n").split(),float))
		#print(covariance_matrix)
		#print(cov_float)
		#print(type(np.asarray(cov_float)))
		os.chdir("..")
		input_dict["cov_float"] = np.asarray(cov_float)
		input_dict["covariance_matrix"] = covariance_matrix.strip("\n")
		return input_dict
	
	def getJPDAP(self):
		input_dict = {}
		#input_dict["samples"] = self.sim_
		#input_dict["samples_skipped"] = int(0.1*self.sim_)
		#input_dict["Random_seed"] = 1
		#input_dict["sampling_method"] = "SOBOL"
		#input_dict["sampling_distribution"] = "NORMAL"
		#input_dict["equidistant_T"] = 100
		#input_dict["T_begin"] = data = self.rxnUnsert[i].temperatures[0]
		#input_dict["T_end"] = self.rxnUnsert[i].temperatures[-1]
		input_dict["L"] = 0
		input_dict["len_temp_data"] = len(self.temperatures)
		string_unsrt_data =""
		for index,k in enumerate(self.temperatures):
			string_unsrt_data+=f"{k} {self.uncertainties[index]} \n"
		input_dict["temperature_unsrt_data"] = string_unsrt_data
		input_dict["alpha"] = np.exp(self.rxn_dict[0])
		input_dict["n"] = self.rxn_dict[1]
		input_dict["n_min"] = self.rxn_dict[1]-2
		input_dict["n_max"] = self.rxn_dict[1]+2
		input_dict["epsilon"] = self.rxn_dict[2]
		L = self.cholskyDeCorrelateMat
		#input_dict["covariance_matrix"] = str(L.dot(L.T)).strip("[]").replace("[","").replace("]","")
		input_dict["uncertainty_type"] = "2slnk"
		if len(self.activeParameters) == 3:
			input_dict["uncertain_parameters"] = "AnE"
		else:
			input_dict["uncertain_parameters"] = "A"
		#input_rxn_dict[i] = input_dict
		string_dict = {}
		jpdap_instring = make_input_file.create_JPDAP_input(input_dict)
		"""
		Run: JPDAP code
		"""
		os.mkdir(f"{self.rIndex}")
		os.chdir(f"{self.rIndex}")
		file_jpdap = open("jpdap_data_"+str(self.rIndex)+".txt","w").write(jpdap_instring)
		run_jpdap_string = f"""#!/bin/bash
{self.jpdap_executable} jpdap_data_{self.rIndex}.txt &> out"""
		file_print_run_jpdap = open("run_jpdap","w").write(run_jpdap_string)
		subprocess.call(["chmod","+x",'run_jpdap'])
		start_Jpdap = time.time()
		subprocess.call(["./run_jpdap"])
		stop_Jpdap = time.time()
		print(f"\n\tJPDAP code took {stop_Jpdap-start_Jpdap}s to execute\n")
		Nagy_covariance_matrix = ""
		file_name_jpdap = "jpdap_data_"+str(self.rIndex)+".txt_fit_minRMSD.txt"
		Nagy_covariance_matrix = open(file_name_jpdap,"r").readlines()[5:8]
		covariance_matrix = ""
		for n in Nagy_covariance_matrix:
			covariance_matrix+=str(n)
		#print(covariance_matrix)
		os.chdir("..")
		input_dict["covariance_matrix"] = covariance_matrix.strip("\n")
		return input_dict
	
	def getKappaMin(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_min)).flatten()
	
	def getMean(self):
		return self.P
	
	def getNominal(self,T):
		T = np.array([self.temperatures[0],(self.temperatures[0]+self.temperatures[-1])/2,self.temperatures[-1]])
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P)).flatten()
	
	def getCov(self):
		return self.cov
	
	def getAllData(self):
		if len(self.branches.split(","))==1:
			b1 = self.branches.split(",")[0]
			b2 = ""
			b3 = ""
		elif len(self.branches.split(",")) >1:
			b1 = self.branches.split(",")[0]
			b2 = self.branches.split(",")[1]
			if len(self.branches.split(",")) >2:
				b3 = self.branches.split(",")[3]
			else:
				b3 = ""
		else:
			b1 = ""
			b2 = ""
			b3 = ""
		exp_data_type = self.exp_data_type.split(";")[0]
		exp_format = self.exp_data_type.split(";")[1]
		Log_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.classification,self.type,self.sub_type,self.branching,b1,b2,b3,self.pressure_limit,self.common_temp,self.temp_limit,exp_data_type,exp_format,self.nametag)
		exp_unsrt_string = ""
		solver_log = "######################\n{}######################\n\t\t{}\n\n".format(self.nametag,self.solution)
		calc_cholesky = self.cholskyDeCorrelateMat
		zeta_string = "{}".format(self.zeta)
		for i in range(len(self.temperatures)):
			exp_unsrt_string += "{}\t{}\n".format(self.temperatures[i],self.uncertainties[i])
		return Log_string,exp_unsrt_string,solver_log,calc_cholesky,zeta_string
 
#public function to get the temperature range for the uncertainity of a perticular reaction
	def getTempRange(self):
		return self.temperatures[0], self.temperatures[-1]
	
	def getTemperature(self):
		return self.temperatures
	
	def getRxnType(self):
		return self.type,self.branching,self.branches

#public function to get the zeta values for a perticulat reaction
	def getData(self):	
		return self.zeta.x
	
	def zetaValues(self):
		return self.zeta.x
#public function to get the cholesky decomposed matrix for normalization of variables
		
	def getCholeskyMat(self):
		return self.cholskyDeCorrelateMat


#############################################################
###	   center broadening factors for showing		######
###		  decomposition and recombination of		###### 
###		  pressure dependent reactions			  ######
#############################################################
class PLOG_Interconnectedness(UncertaintyExtractor):
	def __init__(self,plog_object_list,index,count):
		"""
		Creates the class for the PLOG reactions
		
		"""
		self.selected = False
		self.linked_rIndex = None
		parent = plog_object_list[0]
		parent2 = plog_object_list[1]
		self.rxn = parent.rxn
		self.classification = parent.classification
		self.tag = parent.tag
		self.rIndex = str(parent.rIndex.split(":")[0])+":"+str(index)
		#print(self.rxn)
		fraction = int(count+1)
		alpha = float(int(index)/int(fraction))
		#print(self.rIndex)
		self.index = parent.index
		#print(self.index)
		self.rxn_Details = parent.rxn_Details
		#self.perturbation_factor = parent.perturbation_factor
		self.perturbation_type = parent.perturbation_type
		self.type = parent.type
		self.sub_type = parent.sub_type
		self.exp_data_type = "Interpolation"
		self.temperatures = parent.temperatures
		self.uncertainties = parent.uncertainties
		self.branching  = parent.branching
		self.branches = parent.branches
		self.pressure_limit = "PLOG_"+str(index)
		self.common_temp = parent.common_temp
		self.temp_limit = parent.temp_limit
		self.rxn_dict = self.rxn_Details["rate-constants"][index]
		self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
		
		
		
		if self.type == "pressure_dependent" and self.pressure_limit.strip() != "" and self.classification == "PLOG":
			#print(self.pressure_limit)
			self.nametag = str(self.rxn)+":"+str(self.pressure_limit)
		
		elif self.type == "pressure_dependent" and self.pressure_limit.strip() != "" and self.classification == "PLOG-Duplicate":
			#print(self.pressure_limit)
			self.nametag = str(self.rxn)+":"+str(self.branches)+"-"+str(self.pressure_limit)
		
		elif self.type == "pressure_independent" and self.sub_type == "duplicate":
			self.nametag = str(self.rxn)+":"+str(self.branches)
		else:
			self.nametag = self.rxn
		
		#print(self.nametag)
		#print(self.classification)
		"""
		Interpolation for uncertainty 
		"""
		
		
		data = {}
		data["temperatures"] = self.temperatures
		data["uncertainties"] = self.uncertainties
		data["Arrhenius"] = self.nominal
		
		LOW = None
		HIGH = None
		if "High" in parent.rIndex:
			HIGH = parent.uncertainties
			LOW = parent2.uncertainties
		else:
			HIGH = parent2.uncertainties
			LOW = parent.uncertainties
		#print(alpha)
		interpolation = alpha*LOW + (1-alpha)*HIGH
		#print(LOW)
		#print(HIGH)
		#print(interpolation)
		super().__init__(data)
		self.zeta_Matrix,self.P,self.P_max,self.P_min,self.cov = self.getUncorreationMatrix(self.rIndex)
		self.solution = self.zeta.x
		self.cholskyDeCorrelateMat = self.L
		self.activeParameters = [self.rIndex+'_A',self.rIndex+'_n',self.rIndex+'_Ea']
		self.selection = [1.0,1.0,1.0]
		self.perturb_factor = self.zeta.x
		
		"""
		For perturbing only A-factor
		"""
		if "factor" in self.perturbation_type:
			self.perturb_factor =  [min(self.uncertainties)]
			self.solution = 1.0
			#self.getUncorreationMatrix = np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
			self.cholskyDeCorrelateMat = np.array([min(self.uncertainties)])
			self.zeta_Matrix = 1
			self.activeParameters = [self.rIndex+'_A']
			self.selection = [1.0,0.0,0.0]
		
	def getKappaMin(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_min)).flatten()
	
	def getMean(self):
		return self.P
	
	def getNominal(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P)).flatten()
	
	def getCov(self):
		return self.cholskyDeCorrelateMat
	
	def getKappaMax(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_max)).flatten()
		
class PLOG(UncertaintyExtractor):
		
	def __init__(self, Element,mechPath,binary_files):
		"""
		for both types of PLOG
		PLOG 
		
		PLOG-DUPLICATE
		
		
		"""
		self.rxn = self.classification = self.type = self.sub_type = self.exp_data_type = self.temperatures = self.uncertainties = self.branching  = self.branches = self.pressure_limit = self.common_temp = self.temp_limit = None
		#super().__init__(Element,mechPath,binary_files)
		DATA = Parser(mechPath).mech
		RXN_LIST = Parser(mechPath).rxnList
		self.selected = False
		self.linked_rIndex = None
		
		self.tag = Element.tag
		self.rxn = str(Element.attrib["rxn"])
		#print(self.rxn)
		self.rIndex = str(Element.attrib["no"])
		
		if self.rxn in RXN_LIST:
			self.index = RXN_LIST.index(self.rxn)
		else:
			raise AssertionError("Rxn not in the mechanism. Kindly check the uncertainty file that you have submitted !!\n")
		
		#self.rxn_Details = DATA["reactions"][self.index]
		#print(self.rxn_Details)
		
		#self.rxn_dict = DATA["Reactions"]["rate-constants"]		
		for item in Element:
			if item.tag == "class":
				self.classification = item.text
			if item.tag == "type":
				self.type = item.text
			if item.tag == "perturbation_type":
				self.perturbation_type = item.text
			if item.tag == "perturbation_factor":
				self.perturbation_factor = float(item.text)
			if item.tag == "sub_type":
				self.sub_type = item.attrib["name"]
				try:
					self.linked_rIndex = item.attrib["link"]
				except KeyError:
					self.linked_rIndex = None
				
				for subitem in item:
					if subitem.tag == "multiple":
						self.branching = subitem.text
					if subitem.tag == "branches":
						self.branches = subitem.text
					if subitem.tag == "pressure_limit":
						self.pressure_limit = subitem.text.strip()
					if subitem.tag == "common_temp":
						self.common_temp = subitem.text
					if subitem.tag == "temp_limit":
						self.temp_limit = subitem.text
			
			if item.tag == "data_type":
				self.exp_data_type = item.text
			if item.tag == "file":
				self.exp_data_file = item.text
			if item.tag == "temp":
				#print(item.text)
				self.temperatures = np.asarray([float(i) for i in item.text.split(",")])
			if item.tag == "unsrt":
				self.uncertainties = np.asarray([float(i) for i in item.text.split(",")])
			
		if self.exp_data_type.split(";")[0] == "constant":
			if self.exp_data_type.split(";")[1] == "array":	
				self.temperatures = self.temperatures
				self.uncertainties = self.uncertainties
				
			elif self.exp_data_type.split(";")[1] == "end_points":
				self.temperatures = np.linspace(self.temperatures[0],self.temperatures[1],200)
				self.uncertainties = np.linspace(self.uncertainties[0],self.uncertainties[1],200)
		elif self.exp_data_type.split(";")[0] == "file":
			unsrt_file = open(str(self.exp_data_file),"r").readlines()
			unsrtData = [np.asfarray(i.strip("\n").strip("''").split(","),float) for i in unsrt_file]
			self.temperatures = np.asarray([i[0] for i in unsrtData])
			self.uncertainties = np.asarray([i[1] for i in unsrtData])
		
		if len(self.temperatures) != len(self.uncertainties):
			print("Error in unsrt data for {}".format(self.rxn))
	
		if self.type == "pressure_dependent" and self.pressure_limit.strip() != "" and self.classification == "PLOG":
			#print(self.pressure_limit)
			self.nametag = str(self.rxn)+":"+str(self.pressure_limit)
		
		elif self.type == "pressure_dependent" and self.pressure_limit.strip() != "" and self.classification == "PLOG-Duplicate":
			#print(self.pressure_limit)
			self.nametag = str(self.rxn)+":"+str(self.branches)+"-"+str(self.pressure_limit)
		
		elif self.type == "pressure_independent" and self.sub_type == "duplicate":
			self.nametag = str(self.rxn)+":"+str(self.branches)
		else:
			self.nametag = self.rxn
		
		#print(self.nametag)
		#print(self.classification)
		if self.classification == "PLOG-Duplicate":
			if self.branches == "A":
				self.index = self.index
			else:
				self.index = self.index+1
			
			self.rxn_Details = DATA["reactions"][self.index]
			if self.pressure_limit == "Low":
				self.rxn_dict = DATA["reactions"][self.index]["rate-constants"][0]
			elif self.pressure_limit == "High":
				self.rxn_dict = DATA["reactions"][self.index]["rate-constants"][-1]
			else:
				raise AssertionError(f"Please give a valid input for pressure_limit in the uncertainty file for PLOG!! The reaction is question is \n{self.rxn_details}\n")
			#print(self.rxn_dict)
			self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
			
			
		else:
			self.rxn_Details = DATA["reactions"][self.index]
			if self.pressure_limit == "Low":
				self.rxn_dict = DATA["reactions"][self.index]["rate-constants"][0]
			elif self.pressure_limit == "High":
				self.rxn_dict = DATA["reactions"][self.index]["rate-constants"][-1]
			else:
				raise AssertionError(f"Please give a valid input for pressure_limit in the uncertainty file for PLOG!! The reaction is question is \n{self.rxn_details}\n")
				
			#print(self.rxn_dict)
			self.nominal = [np.log(self.rxn_dict["A"]),self.rxn_dict["b"],self.rxn_dict["Ea"]/1.987]
		
		
		data = {}
		data["temperatures"] = self.temperatures
		data["uncertainties"] = self.uncertainties
		data["Arrhenius"] = self.nominal
		
		"""
		For perturbing all the three reactions
		"""
		super().__init__(data)
		self.zeta_Matrix,self.P,self.P_max,self.P_min,self.cov = self.getUncorreationMatrix(self.rIndex)
		self.solution = self.zeta
		self.cholskyDeCorrelateMat = self.L
		self.activeParameters = [self.rIndex+'_A',self.rIndex+'_n',self.rIndex+'_Ea']
		self.perturb_factor = self.zeta.x
		self.selection = [1.0,1.0,1.0]
		"""
		For perturbing only A-factor
		"""
		if "factor" in self.perturbation_type:
			self.perturb_factor =  [min(self.uncertainties)]
			self.solution = 1.0
			#self.getUncorreationMatrix = np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
			self.cholskyDeCorrelateMat = np.array([min(self.uncertainties)])
			self.zeta_Matrix = 1
			self.activeParameters = [self.rIndex+'_A']
			self.selection = [1.0,0.0,0.0]
		#print(f"{self.rIndex}= {self.zeta_Matrix}")
		
	def getKappaMin(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_min)).flatten()
	
	def getMean(self):
		return self.P
	
	def getNominal(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P)).flatten()
	
	def getCov(self):
		return self.cholskyDeCorrelateMat
	
	def getKappaMax(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_max)).flatten()
		
class fallOffCurve:
	def __init__(self, Element,mechPath):
		self.rxn = self.classification = self.type = self.sub_type = self.exp_data_type = self.temperatures = self.uncertainties = None
		
		self.rxn = Element.attrib["rxn"]
		self.tag = Element.tag	
		self.foc_dict = IFR.MechParsing(mechPath).getFocData(self.rxn)[0]
		self.selected = False
		self.linked_rIndex = None
		
		#print(self.rxn_dict)
		for item in Element:
			if item.tag == "class":
				self.classification = item.text
			if item.tag == "type":
				self.type = item.text
			if item.tag == "sub_type":
				self.sub_type = item.attrib["name"]
				try:
					self.linked_rIndex = item.attrib["link"]
				except KeyError:
					self.linked_rIndex = None
				for subitem in item:
					if subitem.tag == "multiple":
						self.branching = subitem.text
					if subitem.tag == "branches":
						self.branches = subitem.text
					if subitem.tag == "pressure_limit":
						self.pressure_limit = subitem.text
					if subitem.tag == "common_temp":
						self.common_temp = subitem.text
					if subitem.tag == "temp_limit":
						self.temp_limit = subitem.text
			
			if item.tag == "data_type":
				self.exp_data_type = item.text
			if item.tag == "temp":
				#print(item.text)
				self.temperatures = np.asarray([float(i) for i in item.text.split(",")])
			if item.tag == "unsrt":
				self.uncertainties = np.asarray([float(i) for i in item.text.split(",")])
			
		if self.exp_data_type.split(";")[0] == "constant":
			if self.exp_data_type.split(";")[1] == "array":	
				self.temperatures = self.temperatures
				self.uncertainties = self.uncertainties
				
			elif self.exp_data_type.split(";")[1] == "end_points":
				self.temperatures = np.linspace(self.temperatures[0],self.temperatures[1],20)
				self.uncertainties = np.linspace(self.uncertainties[0],self.uncertainties[1],20)
		
		
#		for item in Element:
#			if item.tag == "class":
#				self.classification = item.text
#			if item.tag == "type":
#				self.type = item.text
#			if item.tag == "sub_type":
#				self.sub_type = item.attrib["name"]
#				for subitem in item:
#					if subitem.tag == "branch":
#						self.branching = subitem.text
#					if subitem.tag == "branches":
#						self.branches = subitem.text
#					if subitem.tag == "pressure_limit":
#						self.pressure_limit = subitem.text
#					if subitem.tag == "common_temp":
#						self.common_temp = subitem.text
#					if subitem.tag == "temp_limit":
#						self.temp_limit = subitem.text
#			
#			if item.tag == "data_type":
#				self.exp_data_type = item.text
#			if item.tag == "temp":
#				self.temperatures = np.asarray([float(i) for i in item.text.split(",")])
#			if item.tag == "unsrt":
#				self.uncertainties = np.asarray([float(i) for i in item.text.split(",")])
#			
#		if self.exp_data_type.split(";")[0] == "constant":
#			if self.exp_data_type.split(";")[1] == "array":
#				self.temperatures = self.temperatures
#				self.uncertainties = self.uncertainties
#			elif self.exp_data_type.split(";")[1] == "end_points":
#				self.temperatures = np.linspace(self.temperatures[0],self.temperatures[1],20)
#				self.uncertainties = np.linspace(self.uncertainties[0],self.uncertainties[1],20)
		
		self.nametag = self.rxn+":"+self.sub_type

		L11 = self.uncertainties[0]/3
		L = np.array([L11]) #Cholesky_lower_triangular_matrix
		self.cholskyDeCorrelateMat = L
		#print(L)
		self.zeta = 1
		
		self.T = self.temperatures
		self.f = self.getUncertainty(self.T)		
		self.solution = self.cholskyDeCorrelateMat
		
		
#public function to get the uncertainity values for discrete temperatures
	def getAllData(self):
		Log_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.classification,self.type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.exp_data_type,self.nametag)
		exp_unsrt_string = ""
		solver_log = "{}\n{}\n".format(self.nametag,self.solution)
		calc_cholesky = self.cholskyDeCorrelateMat
		zeta_string = "{}".format(self.zeta)
		for i in range(len(self.temperatures)):
			exp_unsrt_string += "{}\t{}\n".format(self.temperatures[i],self.uncertainties[i])
		return Log_string,exp_unsrt_string,solver_log,calc_cholesky,zeta_string

	def getDtList(self):
		self.Unsrt_dict = {}
		key = ["Tag","Solution","Class","Type","Sub_type","Branch_boolen","Branches","Pressure_limit","Common_temp","temp_list","Nominal","Exp_input_data_type","priorCovariance","Basis_vector","Uncertainties","Temperatures","unsrt_func","Data_key"]
		values = [self.tag,self.solution,self.classification,self.type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.foc_dict,self.exp_data_type,self.cholskyDeCorrelateMat,self.zeta,self.uncertainties,self.temperatures,self.f,""]
		for i,element in enumerate(key):
			self.Unsrt_dict[element] = values[i]
		return self.Unsrt_dict
	
	def getUncertainty(self,T):
		L11 = self.uncertainties[0]/3
		
		Foc_unsrt = 3*np.sqrt((L11*(T/T))**2)
		return Foc_unsrt
 
#public function to get the temperature range for the uncertainity of a perticular reaction
	def getTempRange(self):
		return self.temperatures[0], self.temperatures[-1]
	def getTemperature(self):
		return self.temperatures
	def getRxnType(self):
		
		return self.type,self.IsBranching,self.branchRxn
	def fit_zeta(self,T,z):
		L11 = self.uncertainties[0]/3
		func =z*L11*(T/T)
		
		return func

	def getZeta(self):
		#printZeta = "{}\t{}\t{}\n".format(self.zeta.x[0],self.zeta.x[1],self.zeta.x[2])
		#printL ="{}".format(self.cholskyDeCorrelateMat)
		#printL+="\n"
		#fileZeta = open('../zetaValues.txt','a')
		#fileL = open('../Cholesky.txt','a')
		#fileZeta.write(printZeta)
		#fileL.write(printL)
		#fileZeta.close()
		#fileL.close()
		#print("\n{}\n".format(self.zeta.x));
		return self.zeta
	
	def zetaValues(self):
		return self.zeta
#public function to get the cholesky decomposed matrix for normalization of variables
	
	def getCholeskyMat(self):
		return self.cholskyDeCorrelateMat

	def getKappaMax(self,T):
		T = np.asarray(T)
		theta = np.array([T/T,np.log(T),-1/T])
		return np.asarray(theta.T.dot(self.P_max)).flatten()

#############################################################
###	   Uncertainty for heat capacities			  ######
###	   of kinetic species						   ######
#############################################################


class thermodynamic:
	def __init__(self, Element,thermo_loc):
		self.species = self.classification = self.type = self.sub_type =  self.branching = self.branches  = self.pressure_limit  = self. common_temp = self.temp_limit= None
		self.tag = Element.tag
		IFRT = IFR.ThermoParsing(thermo_loc)
				
		self.exp_data_type = {}
		self.temperatures = {}
		self.uncertainties = {}
		self.cholskyDeCorrelateMat = {}
		self.zeta = {}
		self.species = Element.attrib["species"]
		self.nominal = {}
		self.selected = False
		self.linked_rIndex = None
		
		for item in Element:
			#print(item.tag)
			if item.tag == "class":
				self.classification = item.text
				continue
			if item.tag == "type":
				self.type = item.text
				continue
			if item.tag == "sub_type":
				self.sub_type = item.attrib["name"]
				try:
					self.linked_rIndex = item.attrib["link"]
				except KeyError:
					self.linked_rIndex = None
				for subitem in item:
					if "multiple" in subitem.tag:
						self.branching = str(subitem.text)
						continue
					if "branches" in subitem.tag:
						self.branches = str(subitem.text)
						continue
						
					if "pressure_limit" in subitem.tag:
						self.pressure_limit = str(subitem.text)
						continue
					if "common_temp" in subitem.tag:
						self.common_temp = str(subitem.text)
						continue
					if  "temp_limit" in subitem.tag :
						self.temp_limit = str(subitem.text)
						#print(self.temp_limit)
						if self.temp_limit == "Low":
		
							self.thermo_dict = IFR.ThermoParsing(thermo_loc).getThermoLow(self.species)
						else:
							self.thermo_dict = IFR.ThermoParsing(thermo_loc).getThermoHigh(self.species)
						continue
				continue
				
		for i in self.branches.split(","):
			
			for item in Element:
				if item.tag == "data_type_"+str(i):
					self.exp_data_type[str(i)] = item.text
					continue
					
				if item.tag == "temp_"+str(i):
					#print(item.text)
					self.temperatures[str(i)] = np.asarray([float(j) for j in item.text.split(",")])
					continue
						
				if item.tag == "unsrt_"+str(i):
					self.uncertainties[str(i)] = np.asarray([float(i) for i in item.text.split(",")])
					continue
			
		for i in self.exp_data_type:
			if self.exp_data_type[str(i)].split(";")[0] ==  "constant":
				if self.exp_data_type[str(i)].split(";")[1] == "array":
					continue
				elif self.exp_data_type[str(i)].split(";")[1] == "end_points":
					self.temperatures[str(i)] = np.linspace(self.temperatures[str(i)][0],self.temperatures[str(i)][1],20)
					self.uncertainties[str(i)] = np.linspace(self.uncertainties[str(i)][0],self.uncertainties[str(i)][1],20)
					continue

			elif self.exp_data_type[str(i)].split(";")[0] == "percentage":
				if self.exp_data_type[str(i)].split(";")[1] == "array":
					a = self.thermo_dict[str(i)]
					func = IFRT.function(str(i),a,self.temperatures[str(i)])
					y = self.uncertainties[str(i)]
					#print(y)						
					self.uncertainties[str(i)] = np.asarray(np.dot(y,func)).flatten()
					continue

				elif self.exp_data_type[str(i)].split(";")[1] == "end_points":

					self.temperatures[str(i)] = np.linspace(self.temperatures[str(i)][0],self.temperatures[str(i)][1],20)
					self.uncertainties[str(i)] = np.linspace(self.uncertainties[str(i)][0],self.uncertainties[str(i)][1],20)

					a = self.thermo_dict[str(i)]
					func = IFRT.function(str(i),a,self.temperatures[str(i)])
					#print(str(i),a,self.temperatures[str(i)])
					#print(func)
					y = self.uncertainties[str(i)]						
					#print(y*func)
					self.uncertainties[str(i)] = np.asarray((y*func)).flatten()
					continue
			
		
			
		self.doUnsrtAnalysis()
		for i in self.branches.split(","):
			self.cholskyDeCorrelateMat[str(i)],self.zeta[str(i)] = self.unsrt(str(i))
		self.nametag = self.species+":"+self.temp_limit	
			
		self.corelate_block = block_diag(*(self.cholskyDeCorrelateMat["Hcp"],self.cholskyDeCorrelateMat["h"],self.cholskyDeCorrelateMat["e"]))
		
		self.f = self.getUncertainty(self.temperatures)
		
		
		
		
	def getAllData(self):
		b1 = self.branches.split(",")[0]
		b2 = self.branches.split(",")[1]
		if len(self.branches.split(",")) >2:
			b3 = self.branches.split(",")[3]
		else:
			b3 = ""
		exp_data_type = self.exp_data_type["Hcp"].split(";")[0]
		exp_format = self.exp_data_type["Hcp"].split(";")[1]
		Log_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.classification,self.type,self.sub_type,self.branching,b1,b2,b3,self.pressure_limit,self.common_temp,self.temp_limit,exp_data_type,exp_format,self.nametag)
		exp_unsrt_string = ""
		solver_log = "######################\n{}######################\n\t\t{}\n\n".format(self.nametag,self.solution)
		calc_cholesky = self.corelate_block
		zeta_string = "{}\t{}\t{}\n".format(self.zeta["Hcp"],self.zeta["h"],self.zeta["e"])
		#print(self.temperatures)
		#print(self.uncertainties)
		for i in range(len(self.temperatures["Hcp"])):
			exp_unsrt_string += "{}\t{}\t{}\t{}\t{}\t{}\n".format(self.temperatures["Hcp"][i],self.uncertainties["Hcp"][i],self.temperatures["h"][i],self.uncertainties["h"][i],self.temperatures["e"][i],self.uncertainties["e"][i])
		string_2 = "temp\tHcp\ttemp\th\ttemp\te\t\n"
		file_unsrt = open("./Data/"+self.nametag+"_usrtData.log","w")
		file_unsrt.write(string_2+"\n"+exp_unsrt_string)
		file_unsrt.close()
		return Log_string,exp_unsrt_string,solver_log,calc_cholesky,zeta_string
	
	
	def getDtList(self):
		self.Unsrt_dict = {}
		key = ["Tag","Solution","Class","Type","Sub_type","Branch_boolen","Branches","Pressure_limit","Common_temp","temp_list","Nominal","Exp_input_data_type","priorCovariance","Basis_vector","Uncertainties","Temperatures","unsrt_func","Data_key"]
		values = [self.tag,self.solution,self.classification,self.type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.thermo_dict,self.exp_data_type,self.cholskyDeCorrelateMat,self.zeta,self.uncertainties,self.temperatures,self.f,"Hcp,h,e"]		
		
		for i,element in enumerate(key):
			self.Unsrt_dict[element] = values[i]
		return self.Unsrt_dict
		
	def doUnsrtAnalysis(self):
		T = self.temperatures[self.branches.split(",")[0]]
		guess = 0.001*np.ones(17)
		self.solution = minimize(self.uncertPriorObjective,guess,method="Nelder-Mead",options={'maxiter': 100000, 'maxfev': 100000, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 1E-05, 'fatol': 1E-05, 'adaptive': True})
		Tscale = 5000
		#print(self.solution)
		L11 = self.solution.x[0]
		L21 = self.solution.x[1]/Tscale
		L22 = self.solution.x[2]/Tscale
		L31 = self.solution.x[3]/Tscale**2
		L32 = self.solution.x[4]/Tscale**2
		L33 = self.solution.x[5]/Tscale**2
		L41 = self.solution.x[6]/Tscale**3
		L42 = self.solution.x[7]/Tscale**3
		L43 = self.solution.x[8]/Tscale**3
		L44 = self.solution.x[9]/Tscale**3
		L51 = self.solution.x[10]/Tscale**4
		L52 = self.solution.x[11]/Tscale**4
		L53 = self.solution.x[12]/Tscale**4
		L54 = self.solution.x[13]/Tscale**4
		L55 = self.solution.x[14]/Tscale**4
		L66 = self.solution.x[15]
		L77 = self.solution.x[16]
		self.Lcp = np.array([[L11,L21,L31,L41,L51],[0,L22,L32,L42,L52],[0,0,L33,L43,L53],[0,0,0,L44,L54],[0,0,0,0,L55]])
		self.LH = np.array([L66])
		self.LS = np.array([L77])
		
		self.cholskyDeCorrelateMat_cp = np.matrix(self.Lcp.T)
		self.cholskyDeCorrelateMat_H  = np.matrix(self.LH.T)
		self.cholskyDeCorrelateMat_S  = np.matrix(self.LS.T)
		
		
		theta_cp = np.array([T/T,T,T**2,T**3,T**4])
		theta_H = np.array([T/T])
		theta_S = np.array([T/T])
		
		#Find zeta values
		guess_zeta = 0.01*np.array([1,1,1,1,1,1,1])
		self.zeta = minimize(self.obj_zeta,guess_zeta)
		self.zeta_cp = self.zeta.x[0:5]
		self.zeta_h = self.zeta.x[5]
		self.zeta_s = self.zeta.x[6]
	
	def unsrt(self,index):
		if index == "Hcp":
			L = self.Lcp
			z = self.zeta_cp
		if index == "h":
			L = self.LH
			z = self.zeta_h
		if index == "e":
			L = self.LS
			z = self.zeta_s
		return L,z
		
		
	def func_4(self,T,L11,L12,L22,L13,L23,L33,L14,L24,L34,L44,L15,L25,L35,L45,L55):
		unsrt = 9*((L55*T**4)**2+(L44*T**3+L55*T**4)**2+(L33*T**2+L34*T**3+L35*T**4)**2+(L22*T+L23*T**2+L24*T**3+L25*T**4)**2+(L11+L12*T+L13*T**2+L14*T**3+L15*T**4)**2)
		return unsrt
	def func_zeta(self,T,L0,L1,L2,L3,L4):
		z = np.array([L0,L1,L2,L3,L4])
		L11 = self.solution[0]
		L12 = self.solution[1]
		L22 = self.solution[2]
		L13 = self.solution[3]
		L23 = self.solution[4]
		L33 = self.solution[5]
		L14 = self.solution[6]
		L24 = self.solution[7]
		L34 = self.solution[8]
		L44 = self.solution[9]
		L15 = self.solution[10]
		L25 = self.solution[11]
		L35 = self.solution[12]
		L45 = self.solution[13]
		L55 = self.solution[14]
		fdiff = ((z[0]*L11)+(z[0]*L12+z[1]*L22)*T+(z[0]*L13+z[1]*L23+z[2]*L33)*T**2+(z[0]*L14+z[1]*L24+z[2]*L34+z[3]*L44)*(T**3)+(L15*z[0]+L25*z[1]+L35*z[2]+L45*z[3]+L55*z[4])*T**4)	
		return fdiff

	def getUncertainty(self,T): 
		unsrt = {}
		Tscale = 5000
		T = T["Hcp"]
		L11 = self.solution.x[0]
		L21 = self.solution.x[1]/Tscale
		L22 = self.solution.x[2]/Tscale
		L31 = self.solution.x[3]/Tscale**2
		L32 = self.solution.x[4]/Tscale**2
		L33 = self.solution.x[5]/Tscale**2
		L41 = self.solution.x[6]/Tscale**3
		L42 = self.solution.x[7]/Tscale**3
		L43 = self.solution.x[8]/Tscale**3
		L44 = self.solution.x[9]/Tscale**3
		L51 = self.solution.x[10]/Tscale**4
		L52 = self.solution.x[11]/Tscale**4
		L53 = self.solution.x[12]/Tscale**4
		L54 = self.solution.x[13]/Tscale**4
		L55 = self.solution.x[14]/Tscale**4
		L66 = self.solution.x[15]
		L77 = self.solution.x[16]
		# Truncation at 3 sigma
		unsrt_cp = 3*np.sqrt((L55*T**4)**2+(L44*T**3+L55*T**4)**2+(L33*T**2+L43*T**3+L53*T**4)**2+(L22*T+L32*T**2+L42*T**3+L52*T**4)**2+(L11+L21*T+L31*T**2+L41*T**3+L51*T**4)**2)
		unsrt_H = 3*np.sqrt((L66*(T/T))**2)
		unsrt_S = 3*np.sqrt((L77*(T/T))**2)
		unsrt["Hcp"] = unsrt_cp
		unsrt["h"] = unsrt_H
		unsrt["e"] = unsrt_S
		return unsrt
	
	def uncertPriorObjective (self,guess):
		Tscale = 5000
		R = 8.314
		Z= guess
		#z = np.array([Z[15],Z[16],Z[17],Z[18],Z[19]])
		L11 =Z[0]
		L21 =Z[1]
		L22 =Z[2]
		L31 =Z[3]
		L32 =Z[4]
		L33 =Z[5]
		L41 =Z[6]
		L42 =Z[7]
		L43 =Z[8]
		L44 =Z[9]
		L51 =Z[10]
		L52 =Z[11]
		L53 =Z[12]
		L54 =Z[13]
		L55 =Z[14]
		L66=Z[15]
		L77=Z[16]
		Lcp = np.array([[L11,L21,L31,L41,L51],[0,L22,L32,L42,L52],[0,0,L33,L43,L53],[0,0,0,L44,L54],[0,0,0,0,L55]])
		Lh = np.array([L66])
		Ls = np.array([L77])

		if "h" in self.sub_type:
			Y_h = self.uncertainties["h"]
			thetaH = np.array([T/T])
			sigma_H = 9*(np.dot(Lh,thetaH))**2

		if "e" in self.sub_type:
			Y_s = self.uncertainties["e"]
			thetaS = np.array([T/T])
			sigma_S = 9*(np.dot(Ls,thetaS))**2

		if "Hcp" in self.sub_type:
			Y_cp = self.uncertainties["Hcp"]
			thetaCP = np.array([T/T,T,T**2,T**3,T**4])
			unsrt = 9*((L55*T**4)**2+(L44*T**3+L55*T**4)**2+(L33*T**2+L43*T**3+L53*T**4)**2+(L22*T+L32*T**2+L42*T**3+L52*T**4)**2+(L11+L21*T+L31*T**2+L41*T**3+L51*T**4)**2)

		T = self.temperatures[self.branches.split(",")[0]]/Tscale


		unsrt = 9*((L55*T**4)**2+(L44*T**3+L55*T**4)**2+(L33*T**2+L43*T**3+L53*T**4)**2+(L22*T+L32*T**2+L42*T**3+L52*T**4)**2+(L11+L21*T+L31*T**2+L41*T**3+L51*T**4)**2)


		if "Hcp" in self.sub_type:
			residual_cp = (Y_cp-np.sqrt(unsrt))/(Y_cp/3)
		else:
			residual_cp = np.array([0])
		if "h" in self.sub_type:
			residual_h =(Y_h-np.sqrt(sigma_H))/(Y_h/3)
		else:
			residual_h =  np.array([0])
		if "e" in self.sub_type:
			residual_s =(Y_s-np.sqrt(sigma_S))/(Y_s/3)
		else:
			residual_s = np.array([0])
	
		obj = np.dot(residual_cp.T,residual_cp)+np.dot(residual_h.T,residual_h)+np.dot(residual_s.T,residual_s)
		return obj
	
	def obj_zeta(self,guess):
		T = self.temperatures[self.branches.split(",")[0]]
		#T = np.linspace(1000,5000,100)
		Tscale =5000
		z = np.ones(7)
		L11 = self.solution.x[0]
		L21 = self.solution.x[1]/Tscale
		L22 = self.solution.x[2]/Tscale
		L31 = self.solution.x[3]/Tscale**2
		L32 = self.solution.x[4]/Tscale**2
		L33 = self.solution.x[5]/Tscale**2
		L41 = self.solution.x[6]/Tscale**3
		L42 = self.solution.x[7]/Tscale**3
		L43 = self.solution.x[8]/Tscale**3
		L44 = self.solution.x[9]/Tscale**3
		L51 = self.solution.x[10]/Tscale**4
		L52 = self.solution.x[11]/Tscale**4
		L53 = self.solution.x[12]/Tscale**4
		L54 = self.solution.x[13]/Tscale**4
		L55 = self.solution.x[14]/Tscale**4
		L66 = self.solution.x[15]
		L77 = self.solution.x[16]
		z[0] = guess[0]
		z[1] = guess[1]
		z[2] = guess[2]
		z[3] = guess[3]
		z[4] = guess[4]
		z[5] = guess[5]
		z[6] = guess[6]
		
		zetaFunc_cp = ((z[0]*L11)*(T/T)+(z[0]*L21+z[1]*L22)*T+(z[0]*L31+z[1]*L32+z[2]*L33)*T**2+(z[0]*L41+z[1]*L42+z[2]*L43+z[3]*L44)*(T**3)+(L51*z[0]+L52*z[1]+L53*z[2]+L54*z[3]+L55*z[4])*T**4)
		zetaFunc_H = (T/T)*z[5]*L66
		zetaFunc_S = (T/T)*z[6]*L77
		
		
		if "Hcp" in self.sub_type:
			residual_cp =self.uncertainties["Hcp"]-zetaFunc_cp
		else:
			residual_cp = 0
		if "h" in self.sub_type:
			residual_H = self.H_uncertainties["h"]-zetaFunc_H
		else:
			residual_H = 0
		if "e" in self.sub_type:
			residual_S = self.S_uncertainties["e"]-zetaFunc_S
		else:
			residual_S = 0	
		
		obj = np.dot(residual_cp,residual_cp)+np.dot(residual_H,residual_H)+np.dot(residual_S,residual_S)
		return obj
		
#############################################################
###	   Uncertainty analysis for fallOffCurves,	  ######
###	   enthalpy, entropy and transport			  ######
###	   properties for kinetic species			   ######
#############################################################		
#Temperature independent uncetainties
#zeta = 3
#sigma_p = L11
	
class transport:
	def __init__(self, Element,transport_loc):
		self.species = self.classification = self.type = self.sub_type  = self. branching = self.branches  = self.pressure_limit  = self. common_temp = self.temp_limit =  None
		self.tag = Element.tag
		self.exp_data_type = {}
		self.temperatures = {}
		self.uncertainties = {}
		self.cholskyDeCorrelateMat = {}
		self.zeta = {} 
		self.species = Element.attrib["species"]
		self.selected = False
		self.linked_rIndex = None
		self.trans_dict = IFR.TransportParsing(transport_loc).getTransportData(self.species)
		
		for item in Element:
			#print(item.tag)
			if item.tag == "class":
				self.classification = item.text
				continue
			if item.tag == "type":
				self.type = item.text
				continue
			if item.tag == "sub_type":
				self.sub_type = item.attrib["name"]
				try:
					self.linked_rIndex = item.attrib["link"]
				except KeyError:
					self.linked_rIndex = None
				for subitem in item:
					if "multiple" in subitem.tag:
						self.branching = str(subitem.text)
						continue
					if "branches" in subitem.tag:
						self.branches = str(subitem.text)
						continue
						
					if "pressure_limit" in subitem.tag:
						self.pressure_limit = str(subitem.text)
						continue
					if "common_temp" in subitem.tag:
						self.common_temp = str(subitem.text)
						continue
					if  "temp_limit" in subitem.tag :
						self.temp_limit = str(subitem.text)
						
						continue
				continue
					
		for i in self.branches.split(","):
			
			for item in Element:
				if item.tag == "data_type_"+str(i):

					self.exp_data_type[str(i)] = item.text

					continue
					
				if item.tag == "temp_"+str(i):
					#print(item.text)
					self.temperatures[str(i)] = np.asarray([float(j) for j in item.text.split(",")])
					continue
						
				if item.tag == "unsrt_"+str(i):
					self.uncertainties[str(i)] = np.asarray([float(i) for i in item.text.split(",")])
					continue
		
		
		for i in self.exp_data_type:
			
			if self.exp_data_type[str(i)].split(";")[0] ==  "constant":
				
				if self.exp_data_type[str(i)].split(";")[1] == "array":
				
					continue
				
				elif self.exp_data_type[str(i)].split(";")[1] == "end_points":
				
					self.temperatures[str(i)] = np.linspace(self.temperatures[str(i)][0],self.temperatures[str(i)][1],20)
				
					self.uncertainties[str(i)] = np.linspace(self.uncertainties[str(i)][0],self.uncertainties[str(i)][1],20)
				
					continue

			elif self.exp_data_type[str(i)].split(";")[0] == "percentage":
				
				if self.exp_data_type[str(i)].split(";")[1] == "array":
				
					a = self.trans_dict[str(i)]
					y = self.uncertainties[str(i)]						
					self.uncertainties[str(i)] = y*a
					
					continue

				elif self.exp_data_type[str(i)].split(";")[1] == "end_points":

					self.temperatures[str(i)] = np.linspace(self.temperatures[str(i)][0],self.temperatures[str(i)][1],20)
					self.uncertainties[str(i)] = np.linspace(self.uncertainties[str(i)][0],self.uncertainties[str(i)][1],20)

					a = self.trans_dict[str(i)]
					y = self.uncertainties[str(i)]						
					self.uncertainties[str(i)] = y*a
					continue
				
			
			
		
		for i in self.branches.split(","):
			self.cholskyDeCorrelateMat[str(i)],self.zeta[str(i)] = self.unsrt(str(i))
			
			
		self.nametag = self.species	
		self.f = self.getUncertainty(self.temperatures)
		'''
		fig = plt.figure()
		plt.xlabel('Temperature (K)')
		plt.ylabel('Uncertainity ($\sigma$)')
		plt.title( string+'{}'.format(self.name), fontsize = 10)		
		plt.plot(self.temperatures,self.uncertainties,'o',label='exp uncertainties');
		plt.ylim(0,2*max(self.uncertainties[0],self.uncertainties[-1]))	
		plt.legend()
		my_path = os.getcwd()
		plt.savefig(my_path+'/Plots/TDeptUnsrt/'+self.name+'.png')
		
		'''	
		self.solution = self.cholskyDeCorrelateMat
		self.corelate_block = block_diag(*(self.cholskyDeCorrelateMat["LJe"],self.cholskyDeCorrelateMat["LJs"]))
	
	
	
	def getDtList(self):
		self.Unsrt_dict = {}
		key = ["Tag","Solution","Class","Type","Sub_type","Branch_boolen","Branches","Pressure_limit","Common_temp","temp_list","Nominal","Exp_input_data_type","priorCovariance","Basis_vector","Uncertainties","Temperatures","unsrt_func","Data_key"]
		values = [self.tag,self.solution,self.classification,self.type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.trans_dict,self.exp_data_type,self.cholskyDeCorrelateMat,self.zeta,self.uncertainties,self.temperatures,self.f,"LJe,LJs"]
		
		for i,element in enumerate(key):
			self.Unsrt_dict[element] = values[i]
		return self.Unsrt_dict
		
	def getAllData(self):
		b1 = self.branches.split(",")[0]
		b2 = self.branches.split(",")[1]
		if len(self.branches.split(",")) >2:
			b3 = self.branches.split(",")[3]
		else:
			b3 = ""
		exp_data_type = self.exp_data_type["LJe"].split(";")[0]
		exp_format = self.exp_data_type["LJe"].split(";")[1]
		Log_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.classification,self.type,self.sub_type,self.branching,b1,b2,b3,self.pressure_limit,self.common_temp,self.temp_limit,exp_data_type,exp_format,self.nametag)
		exp_unsrt_string = ""
		solver_log = "######################\n{}######################\n\t\t{}\n\n".format(self.nametag,self.solution)
		calc_cholesky = self.corelate_block
		zeta_string = "{}\t{}\n".format(self.zeta["LJe"],self.zeta["LJs"])
		#print(self.temperatures)
		#print(self.uncertainties)
		for i in range(len(self.temperatures["LJe"])):
			exp_unsrt_string += "{}\t{}\t{}\t{}\n".format(self.temperatures["LJe"][i],self.uncertainties["LJe"][i],self.temperatures["LJs"][i],self.uncertainties["LJs"][i])
		string_2 = "temp\tLJe\ttemp\tLJs\n"
		file_unsrt = open("./Data/"+self.nametag+"_usrtData.log","w")
		file_unsrt.write(string_2+"\n"+exp_unsrt_string)
		file_unsrt.close()
		return Log_string,exp_unsrt_string,solver_log,calc_cholesky,zeta_string
	
	
	def getUncertainty(self,T): 
		unsrt = {}
		T1 = T["LJe"]
		T2 = T["LJs"]
		unsrt_LJe = np.sqrt((float(self.uncertainties["LJe"][0])*(T1/T1))**2)
		unsrt_LJs = np.sqrt((float(self.uncertainties["LJs"][0])*(T2/T2))**2)
		unsrt["LJe"] = unsrt_LJe
		unsrt["LJs"] = unsrt_LJs
		return unsrt
	
	def unsrt(self,index):
		if index == "LJe":
			L = self.uncertainties[index][0]
			z = 1
		if index == "LJs":
			L = self.uncertainties[index][0]
			z = 1
		#print(L)
		return L,z		
	
class collision:
	def __init__(self, Element,string,mechanism_loc):
		self.rxn = self.classification = self.type = self.sub_type = self. branching = self.branches  = self.pressure_limit  = self. common_temp = self.temp_limit= None
		
		self.tag = Element.tag
		self.cholskyDeCorrelateMat = {}
		self.zeta = {}
		self.exp_data_type = {}
		self.temperatures = {} 
		self.uncertainties = {}
		self.nominal = {}
		self.rxn = Element.attrib["rxn"]
		self.selected = False
		self.linked_rIndex = None
		
		for item in Element:
			#print(item.tag)
			if item.tag == "class":
				self.classification = item.text
				continue
			if item.tag == "type":
				self.type = item.text
				continue
			if item.tag == "sub_type":
				self.sub_type = item.attrib["name"]
				
				try:
					self.linked_rIndex = item.attrib["link"]
				except KeyError:
					self.linked_rIndex = None
				for subitem in item:
					if "multiple" in subitem.tag:
						self.branching = str(subitem.text)
						continue
					if "branches" in subitem.tag:
						self.branches = str(subitem.text)
						self.m_dict = IFR.MechParsing(mechanism_loc).getThirdBodyCollisionEff(self.rxn,self.branches)
						continue
						
					if "pressure_limit" in subitem.tag:
						self.pressure_limit = str(subitem.text)
						continue
					if "common_temp" in subitem.tag:
						self.common_temp = str(subitem.text)
						continue
					if  "temp_limit" in subitem.tag :
						self.temp_limit = str(subitem.text)
						
						continue
				continue
					
		for i in self.branches.split(","):
			
			for item in Element:
				if item.tag == "data_type_"+str(i):

					self.exp_data_type[str(i)] = item.text

					continue
					
				if item.tag == "temp_"+str(i):
					#print(item.text)
					self.temperatures[str(i)] = np.asarray([float(j) for j in item.text.split(",")])
					continue
						
				if item.tag == "unsrt_"+str(i):
					self.uncertainties[str(i)] = np.asarray([float(i) for i in item.text.split(",")])
					continue
		
		
		for i in self.exp_data_type:
			
			if self.exp_data_type[str(i)].split(";")[0] ==  "constant":
				
				if self.exp_data_type[str(i)].split(";")[1] == "array":
				
					continue
				
				elif self.exp_data_type[str(i)].split(";")[1] == "end_points":
				
					self.temperatures[str(i)] = np.linspace(self.temperatures[str(i)][0],self.temperatures[str(i)][1],20)
				
					self.uncertainties[str(i)] = np.linspace(self.uncertainties[str(i)][0],self.uncertainties[str(i)][1],20)
				
					continue

			elif self.exp_data_type[str(i)].split(";")[0] == "percentage":
				
				if self.exp_data_type[str(i)].split(";")[1] == "array":
				
					a = self.trans_dict[str(i)]
					y = self.uncertainties[str(i)]						
					self.uncertainties[str(i)] = y*a
					
					continue

				elif self.exp_data_type[str(i)].split(";")[1] == "end_points":

					self.temperatures[str(i)] = np.linspace(self.temperatures[str(i)][0],self.temperatures[str(i)][1],20)
					self.uncertainties[str(i)] = np.linspace(self.uncertainties[str(i)][0],self.uncertainties[str(i)][1],20)

					a = self.trans_dict[str(i)]
					y = self.uncertainties[str(i)]						
					self.uncertainties[str(i)] = y*a
					continue
				
			
			
		
		for i in self.branches.split(","):
			self.cholskyDeCorrelateMat[str(i)],self.zeta[str(i)] = self.unsrt(self.uncertainties[str(i)])
	
		
		self.nametag = self.rxn+":"+self.sub_type
		self.solution = self.cholskyDeCorrelateMat
		self.f = self.getUncertainty()
		#print(self.m_dict)
		'''
		fig = plt.figure()
		plt.xlabel('Temperature (K)')
		plt.ylabel('Uncertainity ($\sigma$)')
		plt.title( string+'{}'.format(self.name), fontsize = 10)		
		plt.plot(self.temperatures,self.uncertainties,'o',label='exp uncertainties');
		plt.ylim(0,2*max(self.uncertainties[0],self.uncertainties[-1]))	
		plt.legend()
		my_path = os.getcwd()
		plt.savefig(my_path+'/Plots/TDeptUnsrt/'+self.name+'.png')
		
		'''	
	def getDtList(self):
		self.Unsrt_dict = {}
		key = ["Tag","Solution","Class","Type","Sub_type","Branch_boolen","Branches","Pressure_limit","Common_temp","temp_list","Nominal","Exp_input_data_type","priorCovariance","Basis_vector","Uncertainties","Temperatures","unsrt_func","Data_key"]
		values = [self.tag,self.solution,self.classification,self.type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.m_dict,self.exp_data_type,self.cholskyDeCorrelateMat,self.zeta,self.uncertainties,self.temperatures,self.f,self.branches]
		for i,element in enumerate(key):
			self.Unsrt_dict[element] = values[i]
		return self.Unsrt_dict
	
	def getAllData(self):
		Log_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.classification,self.type,self.sub_type,self.branching,self.branches,self.pressure_limit,self.common_temp,self.temp_limit,self.exp_data_type,self.nametag)
		exp_unsrt_string = ""
		solver_log = "{}\n{}\n".format(self.nametag,self.sol)
		calc_cholesky = self.cholskyDeCorrelateMat
		zeta_string = "{}".format(self.zeta)
		for i in range(len(self.temperatures)):
			exp_unsrt_string += "{}\t{}\n".format(self.temperatures[i],self.uncertainties[i])
		return Log_string,exp_unsrt_string,solver_log,calc_cholesky,zeta_string
		
	def getUncertainty(self): 
		unsrt = {}
		
		for i in self.branches.split(","):
			#print(i)
			T = self.temperatures[str(i)]
			unsrt_i = np.sqrt((float(self.uncertainties[str(i)][0])*(T/T))**2)
			unsrt[str(i)] = unsrt_i
		return unsrt
	def unsrt(self,unsrt):
		L = np.array([unsrt[0]])
		zeta = 1
		return L,zeta
	
#Parent class to host the uncertainty data for all the reactions. Has instances of reaction class as members		
class uncertaintyData:
	def __init__(self,pathDictionary,binary_files,unsrt_type=None):
	
		self.xmlPath = pathDictionary["uncertainty_data"]
		self.mechPath = pathDictionary["mechanism"]
		self.thermoPath = pathDictionary["thermo_file"]
		self.transportPath = pathDictionary["trans_file"]
		
		self.Reactionlist = []
		self.interConnectedlist = []
		self.PlogRxnlist = []
		self.focList = []
		self.mList = []
		self.thermoList = []
		self.transportList = []
		self.PlogRxnIndex = {}
		self.unsrt_data = {}
		self.reactionUnsrt = {}
		self.focUnsrt = {}
		self.plogUnsrt = {}
		self.Plog_index = {}
		self.trdBodyUnsrt = {}
		self.interconnectedRxn = {}
		#self.Plog_additional_Rxn = {}
		self.thermoUnsrt = {}
		self.transportUnsrt = {}
		self.tree = ET.parse(self.xmlPath)
		self.root = self.tree.getroot()
		count_rxn = 0
		count_foc = 0
		count_m = 0
		count_thermo = 0
		count_transport = 0
		
		for child in self.root:
			if child.tag == 'reaction':
				r = reaction(child,self.mechPath,binary_files)
				self.Reactionlist.append(r.nametag)
				self.reactionUnsrt[r.nametag] = r
				self.unsrt_data[r.nametag] = r
				count_rxn +=1
			if child.tag == 'PLOG':
				p = PLOG(child,self.mechPath,binary_files)
				self.PlogRxnlist.append(p.rIndex)
				self.PlogRxnIndex[p.rIndex] = p.nametag
				self.Plog_index[p.rIndex] = p.index
				self.plogUnsrt[p.nametag] = p
				self.unsrt_data[p.nametag] = p
				self.Reactionlist.append(p.nametag)
				count_rxn +=1
			
			if child.tag == "PLOG-Interconnectedness":
				plog_object_list = []
				list_of_PLOG_rxn = None
				#print(self.PlogRxnIndex)
				#print(self.Plog_index)
				for item in child:
					if item.tag == "InterConnectedRxns":				
						list_of_PLOG_rxn = item.text.split(",")
						
					if item.tag == "RxnCount":
						count = int(item.text)
				#print(list_of_PLOG_rxn)
				for i in list_of_PLOG_rxn:
					if i in self.PlogRxnIndex:
						#print(i)
						plog_object_list.append(self.plogUnsrt[self.PlogRxnIndex[i]])
					else:
						raise AssertionError(f"Invalid connected reactions are identified. Please check {item.text}")
				for i in range(count):
					index = i+1
					
					q = PLOG_Interconnectedness(plog_object_list,index,count)
					self.interconnectedRxn[q.nametag] = q
					self.interConnectedlist.append(q.nametag)
					self.unsrt_data[q.nametag] = q
					count_rxn +=1
					
			if child.tag == 'fallOffCurve':
				foc = fallOffCurve(child,self.mechPath)
				self.focList.append(foc.nametag)
				self.focUnsrt[foc.nametag] = foc
				self.unsrt_data[foc.nametag] = foc
				count_foc +=1
			if child.tag == 'thermo':
				th = thermodynamic(child,self.thermoPath)
				self.thermoList.append(th.nametag)
				self.thermoUnsrt[th.nametag] = th
				self.unsrt_data[th.nametag] = th
				count_thermo +=1
			if child.tag == 'collisionEff':
				string = "Unsrt for third bodies\n collision efficiencies [M]:  "
				m = collision(child,string,self.mechPath)
				self.mList.append(m.nametag)
				self.trdBodyUnsrt[m.nametag] = m
				self.unsrt_data[m.nametag] = m
				count_m +=1
			if child.tag == 'transport':
				tr = transport(child,self.transportPath)
				self.transportUnsrt[tr.nametag] = tr
				self.transportList.append(tr.nametag)
				self.unsrt_data[tr.nametag] = tr
				count_transport +=1
				#print(self.transportList)
		if unsrt_type !="opt":
			print("\n\n{} Reactions are selected for optimization\n".format(count_rxn))
			print("{} Fall-off (center broadening factors) of reactions {} are selected for optimization\n\n".format(count_foc,self.focList))
			print("{} third body collision efficiency's are selected for optimization\n\n".format(count_m))
			print("{} thermo-chemical parameters are selected for optimization\n\n".format(count_thermo))
			print("{} transport parameters are selected for optimization\n\n".format(count_transport))
			
	def extract_uncertainty(self):
		#print(self.root)
		return self.unsrt_data#,self.reactionUnsrt,self.plogUnsrt,self.interconnectedRxn, self.focUnsrt, self.trdBodyUnsrt, self.thermoUnsrt, self.transportUnsrt, self.Reactionlist,self.PlogRxnlist,self.interConnectedlist,self.focList,self.mList,self.thermoList,self.transportList
	
