# SmartPlugs
Making a system for smart houses that analyse what type of equipment is connected to a plug, Later it will have more implementations 

########
HOW TO USE
########
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Output
0 - COMPUTER
1 - LAMP
2 - COMPUTER + LAMP
3 - SCREEN
4 - COMPUTER + SCREEN
5 - LAMP + SCREEN ----
6 - All Togeth

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%  FOLDERS %%%%%%%%%%%%%%%

flaskApp - folder que tem guardado a applicacao em flaskApp
la dentro tem tambem na pasta statics/imgs/trainedModels o plot da accuracy e lost do ultimo treino da rede
node/lastwave tem a ultima onda que foi enviada do node
O node e um dispositivo de tomada que foi ligado

/imgs
pasta que guarda as imagens geradas para treino e teste

/powerEng 
pasta que guarda documentos relativamente ao artigo

/trainingWeights
pasta que guarda os pesos da rede neural

/ArtigosRelevantes
pasta onde estao guardados alguns artigos Relevantes

%%%%%%%%%%%%%%%%%%%%%% FILES %%%%%%%%%%%%%%%%%%%%%%%%%
accesspoint.py ficheiro que aciona o AP ainda nao funciona 100%

client.py ficheiro que simula ser um cliente ou seja um node 

dataframe_test.pkl dataframe que guarda informacao sobre ondas de teste para a rede neural

dataframe_train.pkl dataframe que guarda informacao sobre ondas de treino,{label, valor_real, valorEficaz}

new_df.pkl dataframe que guarda info {label, array_spectrogram} e esta que vai ser lida pela CNN

neuralNet.py ficheiro python que tem a rede neural, gerador de ondas, e preparacao do dataset

wavePreparation.py script para preparacao da onda tem varias funcoes de backend

server.py script que tem o server

nodes.json guarda info ip add, id , do node e da sua ultima onda gerada
