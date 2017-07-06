import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import MySQLdb
from detect_peaks import detect_peaks
from scipy.signal import find_peaks_cwt
from scipy.integrate import simps
from scipy import integrate
from numpy import trapz
from kmeans import k_means_function

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x)
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return(b_0, b_1)

#data_angular = np.genfromtxt('dados_SVM_so_angular_resultante_dinamico.csv')
data_angular = np.genfromtxt('dados_SVM_so_angular.csv')
data_linear = pd.read_csv('dados.csv')
#data_linear = np.genfromtxt('dados.csv',delimiter=',',names=True,dtype=None)

data_new = data_angular[:10]
print(data_new)
x = np.linspace(0, len(data_new), len(data_new),  endpoint=True, )
indexes_maiorpeak = detect_peaks(data_new, mph=0.04, mpd=300)
#print(indexes_maiorpeak)
indexes_variospeaks = find_peaks_cwt(data_new, np.arange(1, 25))
#print(indexes_variospeaks)


#Calcular area depois do spyke de maior intensidade

dados_integral = data_new[(indexes_maiorpeak):]
print(dados_integral)

# Compute the area using the composite trapezoidal rule.
#The argument dx=1 indicates that the spacing of the data along the x axis is 1 units.

area_integral = trapz(dados_integral, dx=1)

print("area_integral: ", area_integral)

if area_integral < 200:

	data_new_linear = data_linear[:10]
	Y = k_means_function(np.array(data_new_linear)) 
	#fazer media por linha para fazer a predicao
	if Y == 0:
        	print('Estatico - Deitado')
	
	if Y == 1:
        	print('Estatico - Sentado')
	
	if Y == 2:
		print('Estatico - Em Pe')
	#print("Estatico")
else:
# Compute Linear Regression

	print('Dinamico')
	x_regression = x[(indexes_maiorpeak):]
	b = estimate_coef(x_regression, dados_integral)

	print("Estimated coefficients Falling:\nb_0 = {}  \
        	 \nb_1 = {}".format(b[0], b[1]))
	# predicted response vector
	y_pred = b[0] + b[1]*x_regression
	print("y_pred: ", y_pred)

	if b[1] > 0.5:
		print ('Dinamico-Caiu')


