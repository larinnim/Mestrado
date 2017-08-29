# Mestrado
** Limpar todos os bancos de dados
** rm dados.csv [ cd /host]



1) Verificar no Software qual o IP SERVER_IP está atualmente sendo utilizado.[Aplicativo e raspberry na mesma rede]
	Rede configurada no aplicativo: 192.168.0.5
2) No raspberry digitar: cd /host
			python pythonserver_Sentado.py
			python pythonserver_Deitado.py
			python pythonserver_Em_Pe.py
			python pythonserver_Andando.py
			Ao mesmo tempo que clica em calibrar no Android

	Nesse momento todas as informações serão enviadas para o banco de dados

3) Executar python populate_db.py para criar o arquivo "dados.csv" contendo as informações de calibração

4) sudo service cron start - A cada minuto o cron executa o algoritmo "k_means_v4.0.py"


k_means_v4.0.py:
	- realizar a leitura do sensor e armazenar em um array
	- identificar o index da maior magnitude lida pelo sensor através da função "detect_peaks()"
	- encontrar todos os peaks/spikes do vetor através da função "find_peaks_cwt"
	- separa o array a partir do valor de maior magnitude
	- realiza a integral da área remanescente através da função trapezoidal, dx = 1
	- condicional se o valor da integral < 150 para determinar que o movimento é estático
		Sim: invoca o algoritmo para fazer a predição da posição, passando como parâmetro
			o vetor de aceleração linear. O valor retornado pela função é a posição estática
		Não: realiza regressão linear e identifica os valores dos coeficientes
			Condicional se é queda: o coeficiente angular > 0.5 e diferença entre o os dois maiores peaks > 70

Algoritmos para apresentar:

K_Means_4cluster_linear_teste_1.py: Apresenta o algoritmo com clusterização de todas as posições inclusive cair
K_Means_4cluster_linear_teste.py: Apresenta o algoritmo com clusterização de todas as posições exceto cair
k_Means.py: A curva entre Andando e Caiu


	


 

