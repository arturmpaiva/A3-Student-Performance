import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('student_performance_prediction.csv')

# Criando uma cópia dos dados para manipulação, mantendo os dados originais intactos
copyData = data.copy()

# Tratar variáveis numéricas de valor nulo, substituindo pelo valor da média
count = 0
for variavel in copyData.columns:
    if count == 0:
        count += 1
    elif count < 4:
        copyData[variavel] = copyData[variavel].fillna(copyData[variavel].mean())
        count += 1
    
# Colocar variáveis infinitas como Nulos
#copyData['Study Hours per Week'] = copyData['Study Hours per Week'].replace([np.inf, -np.inf], np.nan)

# Tratar variáveis não numéricas assinalando valores binários (variáveis categóricas)
copyData['Passed'] = copyData['Passed'].map({'Yes': 1, 'No': 0})
copyData['Participation in Extracurricular Activities'] = copyData['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Transformar variáveis categóricas em uma matriz de variáveis dummy e depois num array numpy
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(copyData[['Parent Education Level']]).toarray()

# Visualizar as categorias da variável em colunas
feature_labels = ohe.categories_

# Transformando o array de categorias em uma lista simples
feature_labels = np.array(feature_labels).ravel()

# Converte o array feature_array em um data frame e define o nome das colunas com os rótulos de feature_labels
features = pd.DataFrame(feature_array, columns = feature_labels)

# Juntando a variavel dummy e as colunas OHE para a base de dados original e deleta a original
dataNew = pd.concat([copyData, features], axis=1)
dataNew = dataNew.drop(columns='Parent Education Level', axis=1)

# Colocar a coluna 'Passed' como a última
colunas = [col for col in dataNew.columns if col != 'Passed']

# Adiciona a coluna 'Passed' no final
colunas.append('Passed')

# Reordenamos o DataFrame com base nas novas colunas
dataNew = dataNew[colunas]

# Visualizando o resultado
# print(dataNew.head())

# Definindo as variáveis explicativas e a variável alvo
y = dataNew.Passed
x = dataNew.drop(columns='Passed', axis=1)
# print(y)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
# print(x.shape, x_train.shape, x_test.shape)

# Linear Regression
reg = LinearRegression()
# reg.fit(x_train.values, y_train.values)

# Prevendo pelos dados treinados
train_data_pred = reg.predict(x_train.values)

# R squared value
r2_train = metrics.r2_score(y_train, train_data_pred)
#print('r2 squared value: ', r2_train)




# Histograma
#plt.figure(figsize=(8,6))
#sns.histplot(copyData['Study Hours per Week'], bins=15, kde=True, color='blue')
#plt.title('Distribuição das Horas de Estudo por Semana')
#plt.xlabel('Horas de Estudo por Semana')
#plt.ylabel('Frequência')
#plt.show()

# Gráfico de plot
#sns.relplot(
#    data=copyData,
#    x="Previous Grades", y="Study Hours per Week", col="Passed", hue="Attendance Rate")
#plt.show()

# Distribuição da variável de frequência
#sns.displot(copyData['Attendance Rate'])
#plt.title('Frequência')
#plt.show()

# Distribuição da variável de nível de escolaridade dos pais
#educationLevelGraph = sns.countplot(x='Parent Education Level', data=copyData)
#plt.title('Escolaridade dos pais')

# Mostrar valores exatos para cada nível de escolaridade
#for p in educationLevelGraph.patches:
#    educationLevelGraph.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
#                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
#                textcoords='offset points')
#plt.show()

