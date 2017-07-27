import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import MySQLdb
import time
import sys
import smbus
import MySQLdb
import math
from detect_peaks import detect_peaks
from scipy.signal import find_peaks_cwt
from scipy.integrate import simps
from scipy import integrate
from numpy import trapz
from kmeans import k_means_function


# Power management registers
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

#Declare g's arrays
gx = []
gy = []
gz = []
ax = []
ay = []
az = []
g_resultante = []
a_resultante = []

bus = smbus.SMBus(1) # or bus = smbus.SMBus(1) for Revision 2 boards
address = 0x68       # This is the address value read via the i2cdetect command

# Now wake the 6050 up as it starts in sleep mode
bus.write_byte_data(address, power_mgmt_1, 0)

def read_byte(adr):
    return bus.read_byte_data(address, adr)

def read_word(adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr+1)
    val = (high << 8) + low
    return val

def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def dist(a,b):
    return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)

def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)

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

for i in range(0,22):
        time.sleep(0.1)
        gyro_xout = read_word_2c(0x43)
        gyro_yout = read_word_2c(0x45)
        gyro_zout = read_word_2c(0x47)

	
        gyro_xout_scaled = gyro_xout / 131
        gyro_yout_scaled = gyro_yout / 131
        gyro_zout_scaled = gyro_zout / 131

	gx.append(gyro_xout)
	gy.append(gyro_yout)
	gz.append(gyro_zout)
	
	g_resultante.append(math.sqrt(math.pow(gyro_xout_scaled,2)+math.pow(gyro_yout_scaled,2)+math.pow(gyro_zout_scaled,2)))
	#print "g_resultante: ", g_resultante 

        #print "gyro_xout : ", (gyro_xout / 131)
        #print "gyro_yout : ", (gyro_yout / 131)
        #print "gyro_zout : ", (gyro_zout / 131)

        accel_xout = read_word_2c(0x3b)
        accel_yout = read_word_2c(0x3d)
        accel_zout = read_word_2c(0x3f)

        accel_xout_scaled = accel_xout / 16384.0
        accel_yout_scaled = accel_yout / 16384.0
        accel_zout_scaled = accel_zout / 16384.0

	ax.append(accel_xout_scaled)
	ay.append(accel_yout_scaled)
        az.append(accel_zout_scaled)

	a_resultante.append(math.sqrt(math.pow(accel_xout_scaled,2)+math.pow(accel_yout_scaled,2)+math.pow(accel_zout_scaled,2)))

        #print "accel_xout: ", accel_xout_scaled
        #print "accel_yout: ", accel_yout_scaled
        #print "accel_zout: ", accel_zout_scaled
	time.sleep(0.5)

ax_mean = np.mean(ax)
ay_mean = np.mean(ay)
az_mean = np.mean(az)

a_resultante = [ax_mean, ay_mean, az_mean]
print"a_resultante: ", a_resultante 

conmysql = MySQLdb.connect(host= "localhost",
                  user="root",
                  passwd="admin",
                  db="db_Cotidiano")
curs = conmysql.cursor() 

#data_angular = np.genfromtxt('dados_SVM_so_angular_resultante_dinamico.csv')
#data_angular = np.genfromtxt('dados_SVM_so_angular.csv')
#data_linear = pd.read_csv('/home/pi/dados.csv')
#data_linear = np.genfromtxt('dados.csv',delimiter=',',names=True,dtype=None)

#data_new = data_angular[:10]
#print(data_new)
x = np.linspace(0, len(g_resultante), len(g_resultante),  endpoint=True, )
indexes_maiorpeak = detect_peaks(g_resultante, mph=0.04, mpd=300)
#print(indexes_maiorpeak)
indexes_variospeaks = find_peaks_cwt(g_resultante, np.arange(1, 25))
#print(indexes_variospeaks)


#Calcular area depois do spyke de maior intensidade

dados_integral = g_resultante[(indexes_maiorpeak):]
print(dados_integral)

# Compute the area using the composite trapezoidal rule.
#The argument dx=1 indicates that the spacing of the data along the x axis is 1 units.

area_integral = trapz(dados_integral, dx=1)

print("area_integral: ", area_integral)

count_Deitado = 0
count_Sentado = 0
count_Andando = 0
count_EmPe = 0
count_Caiu = 0


if area_integral < 200:
	#Mean = np.array(a_resultante).mean
	
	#data_new_linear = data_linear[:10]
	Y = k_means_function(a_resultante) 
	#fazer media por linha para fazer a predicao
	if Y == 0:
        	print('Estatico - Deitado')
		posicao = "Deitado"
		count_Deitado += 1 
	        curs.execute ("""INSERT INTO posicao_Deitado (Valor) VALUES (%s)""",(count_Deitado))
	if Y == 1:
        	print('Estatico - Sentado')
		posicao = "Sentado"
		count_Sentado += 1
                curs.execute ("""INSERT INTO posicao_Sentado (Valor) VALUES (%s)""",(count_Sentado))
	if Y == 2:
		print('Estatico - Em Pe')
		posicao = "Em pe"
		count_EmPe +=1 
                curs.execute ("""INSERT INTO posicao_EmPe (Valor) VALUES (%s)""",(count_EmPe))

	conmysql.commit()
        conmysql.close()
	#print("Estatico")
else:
# Compute Linear Regression

	posicao = "Andando"
	count_Andando += 1
	curs.execute("""INSERT INTO tabela_posicao VALUES (%s)""",(posicao))
        curs.execute ("""INSERT INTO posicao_Andando (Valor) VALUES (%s)""",(count_Andando))
	conmysql.commit()
        conmysql.close()

	x_regression = x[(indexes_maiorpeak):]
	b = estimate_coef(x_regression, dados_integral)
	
	print("Estimated coefficients Falling:\nb_0 = {}  \
        	 \nb_1 = {}".format(b[0], b[1]))
	# predicted response vector
	y_pred = b[0] + b[1]*x_regression
	print("y_pred: ", y_pred)

	if b[1] > 0.5:
		posicao = "Caiu"
		count_Caiu += 1
		curs.execute('''INSERT INTO tabela_posicao VALUES (%s)''',(posicao))
	        curs.execute ("""INSERT INTO posicao_Caiu (Valor) VALUES (%s)""",(count_Caiu))
	        conmysql.commit()
        	conmysql.close()

		print ('Dinamico-Caiu')


