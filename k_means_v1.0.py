import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import MySQLdb
from detect_peaks import detect_peaks
from scipy.signal import find_peaks_cwt
from scipy.integrate import simps
from scipy import integrate
from numpy import trapz

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

data_angular = np.genfromtxt('dados_SVM_so_angular_resultante_dinamico.csv')

x = np.linspace(0, len(data_angular), len(data_angular),  endpoint=True, )
indexes_maiorpeak = detect_peaks(data_angular, mph=0.04, mpd=300)
indexes_variospeaks = find_peaks_cwt(data_angular, np.arange(1, 25))

#Calcular area depois do spyke de maior intensidade

dados_integral = data_angular[(indexes_maiorpeak):]
# Compute the area using the composite trapezoidal rule.
#The argument dx=1 indicates that the spacing of the data along the x axis is 1 units.

area_integral = trapz(dados_integral, dx=1)

print("area_integral: ", area_integral)


# Compute Linear Regression

x_regression = x[(indexes_maiorpeak):]
b = estimate_coef(x_regression, dados_integral)

print("Estimated coefficients Falling:\nb_0 = {}  \
         \nb_1 = {}".format(b[0], b[1]))
# predicted response vector
y_pred = b[0] + b[1]*x_regression

print("y_pred: ", y_pred)

broken_df = pd.read_csv('dados.csv')
mat = broken_df.as_matrix()


kmeans = KMeans(n_clusters=4,random_state=3425).fit(mat)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(labels)
X=np.array([0.5632324, 0.0695801, 0.9223633])
Y=kmeans.predict(X)
if Y == 0:
	print('andando')

if Y == 1:
        print('deitado')


if Y == 2:
        print('sentado')


if Y == 3:
        print('Em Pe')


