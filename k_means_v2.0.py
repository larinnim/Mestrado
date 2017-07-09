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

while 1:
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

        #print "accel_xout: ", accel_xout_scaled
        #print "accel_yout: ", accel_yout_scaled
        #print "accel_zout: ", accel_zout_scaled
	time.sleep(0.5)

print "Angular gx: ", gx
print "Linear  ax: ", ax
 
#data_angular = np.genfromtxt('dados_SVM_so_angular_resultante_dinamico.csv')
#data_angular = np.genfromtxt('dados_SVM_so_angular.csv')
data_linear = pd.read_csv('/home/pi/dados.csv')
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


