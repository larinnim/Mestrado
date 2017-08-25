import csv
import MySQLdb
from UnicodeSupportForCsv import UnicodeWriter


con_andando = MySQLdb.Connection(db='db_Andando', host='localhost',           
user='root', passwd='admin')

crsr_andando = con_andando.cursor()

con_deitado = MySQLdb.Connection(db='db_Deitado', host='localhost',
user='root', passwd='admin')

crsr_deitado = con_deitado.cursor()

con_sentado = MySQLdb.Connection(db='db_Sentado', host='localhost',
user='root', passwd='admin')

crsr_sentado = con_sentado.cursor()

con_em_pe = MySQLdb.Connection(db='db_Em_Pe', host='localhost',
user='root', passwd='admin')

crsr_em_pe = con_em_pe.cursor()

crsr_andando.execute("SELECT * FROM tabela")
crsr_deitado.execute("SELECT * FROM tabela")
crsr_sentado.execute("SELECT * FROM tabela")
crsr_em_pe.execute("SELECT * FROM tabela")

with open(r'/home/pi/host/dados.csv', 'wb') as csvfile:
    uw = UnicodeWriter(
        csvfile, delimiter=',',
        quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in crsr_andando.fetchall():
        uw.writerow([unicode(col) for col in row])
    for row in crsr_deitado.fetchall():
        uw.writerow([unicode(col) for col in row])
    for row in crsr_sentado.fetchall():
        uw.writerow([unicode(col) for col in row])
    for row in crsr_em_pe.fetchall():
        uw.writerow([unicode(col) for col in row])

	
