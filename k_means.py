import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import MySQLdb
from detect_peaks import detect_peaks
from scipy.signal import find_peaks_cwt
from scipy.integrate import simps
from scipy import integrate
from numpy import trapz

conmysql = MySQLdb.connect(host='localhost',
		user='root',
		passwd='admin',
		db="db_Cotidiano"
		)

curs = conmysql.cursor()

curs.execute("SELECT * FROM tabela")
#Continuar exercicio colocando dados do csv no mysql para fazer as condicionais de movimento dinamico ou estatico

data_angular = curs.fetchall ()
#print(len(data_angular))
#print(len (dados_SVM_so_angular_resultante_dinamico.csv))
#data_angular = list (data_angular)
data_angular = np.array(data_angular)
print(len(data_angular))


#np.array(data_angular).dump(open('array.npy', 'wb'))
#data_angular = np.load(open('array.npy', 'rb'))

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

x = np.linspace(0, len(data_angular)/2, len(data_angular)/2,  endpoint=True, )
indexes_maiorpeak = detect_peaks(data_angular, mph=None, mpd=1)
#dados = np.genfromtxt('dados_SVM_so_angular_resultante_dinamico.csv')
#indexes_maiorpeak = detect_peaks(dados, mph=None, mpd=1)

indexes_variospeaks = find_peaks_cwt(dados, np.arange(1, 25))
#Calcular area depois do spyke de maior intensidade

dados_integral = data_angular[(indexes_maiorpeak):]
# Compute the area using the composite trapezoidal rule.
#The argument dx=1 indicates that the spacing of the data along the x axis is 1 units.

area_integral = trapz(dados_integral, dx=1)

print("area_andando: ", area_integral)


# Compute Linear Regression

x_regression = x[(indexes_maiorpeak):]
b = estimate_coef(x_regression, dados_integral)

print("Estimated coefficients Falling:\nb_0 = {}  \
         \nb_1 = {}".format(b_caiu[0], b_caiu[1]))
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

