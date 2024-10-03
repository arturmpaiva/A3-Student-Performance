import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('student_performance_prediction.csv')

# Criando uma cópia dos dados para manipulação, mantendo os dados originais intactos
copyData = data.copy()

#       TRATANDO    DADOS

# Eliminar valores nulos, substituindo pelo valor da média nas variáveis numéricas, por "No" para a variável Participação de Atividades
# Extracurriculares e por "High School" para a variável Nível de Escolaridade dos Pais
count = 0
for variavel in copyData.columns:
    if count == 0:
        count += 1
    elif count < 4:
        copyData[variavel] = copyData[variavel].fillna(copyData[variavel].mean())
        count += 1
    elif count == 4:
        copyData[variavel] = copyData[variavel].fillna('No')
        count += 1
    elif count == 5:
        copyData[variavel] = copyData[variavel].fillna('High School')
        count += 1

# Tratar variáveis não numéricas assinalando valores binários (variáveis categóricas)
copyData['Passed'] = copyData['Passed'].map({'Yes': 1, 'No': 0})
copyData['Participation in Extracurricular Activities'] = copyData['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Tirar o "S" no início dos students ID para facilitar a leitura da variável
copyData['Student ID'] = copyData['Student ID'].astype(str)
copyData2 = copyData[copyData['Student ID'].str.match(r'S\d{5}')]
copyData2.loc[:, 'Student ID'] = copyData2['Student ID'].apply(lambda x: re.sub(r'S', '', x))
#print(copyData['Student ID'], copyData2['Student ID'])


# TRANSFORMAR   NUM     DATAFRAME

# Transforma variáveis categóricas em uma matriz de variáveis dummy e depois num array numpy
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(copyData2[['Parent Education Level']]).toarray()

# Visualiza as categorias da variável em colunas
feature_labels = ohe.categories_

# Transforma o array de categorias em uma lista simples  
feature_labels = np.array(feature_labels).ravel()

# Converte o array feature_array em um data frame e define o nome das colunas com os rótulos de feature_labels
features = pd.DataFrame(feature_array, columns = feature_labels)

# Junta as colunas OHE com a base de dados original em uma nova e deleta a original
dataNew = pd.concat([copyData2, features], axis=1)
dataNew = dataNew.drop(columns='Parent Education Level', axis=1)

# Passar coluna 'Passed' para última
colunas = [col for col in dataNew.columns if col != 'Passed']

# Adiciona a coluna 'Passed' no final
colunas.append('Passed')

# Reordenamos o DataFrame com base nas novas colunas
dataNew = dataNew[colunas]

# Apagar os nulos da coluna Passed
dataNew.dropna(subset=['Passed'], inplace=True)

#   DEFINIR     VARIÁVEIS     EXPLICATIVAS  E   ALVO

# Definindo as variáveis explicativas e a variável alvo
y = dataNew.Passed
x = dataNew.drop(columns=['Student ID', 'Passed'], axis=1)
#print(y)
#print(x)

# Separando as variáveis em de treino(80%) e de teste(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

# Checar a regressão linear
reg = LinearRegression()
reg.fit(x_train.values, y_train.values)

# Prevendo pelos dados treinados
train_data_pred = reg.predict(x_train.values)

# R squared value
r2_train = metrics.r2_score(y_train, train_data_pred)
print('r2 squared value: ', r2_train)

test_data_pred = reg.predict(x_test.values)
r2_test = metrics.r2_score(y_test, test_data_pred)
print('r2 squared value: ', r2_test)

# Gerar o gráfico
sns.displot(y_train - train_data_pred, kde=True)

# Adicionar o título
plt.title('Residual: ', size=18)

# Adicionar rótulos exatos nas barras (contagem)
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

# Exibir o gráfico
plt.show()
# Residual: diferencia entre a variável alvo e a predição
#sns.displot(y_train - train_data_pred)
#plt.title('Residual: ', size=18)
#plt.show()


# Histograma
#plt.figure(figsize=(8,6))
#sns.histplot(dataNew['Study Hours per Week'], bins=15, kde=True, color='blue')
#plt.title('Distribuição das Horas de Estudo por Semana')
#plt.xlabel('Horas de Estudo por Semana')
#plt.ylabel('Frequência')
#plt.show()

# Gráfico de plot
#sns.relplot(
#    data=dataNew,
#    x="Previous Grades", y="Study Hours per Week", col="Passed", hue="Attendance Rate")
#plt.show()

# Distribuição da variável de frequência
#sns.displot(dataNew['Attendance Rate'])
#plt.title('Frequência')
#plt.show()

# Distribuição da variável de nível de escolaridade dos pais
#educationLevelGraph = sns.countplot(x='Parent Education Level', data=dataNew)
#plt.title('Escolaridade dos pais')

# Mostrar valores exatos para cada nível de escolaridade
#for p in educationLevelGraph.patches:
#    educationLevelGraph.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
#                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
#                textcoords='offset points')
#plt.show()



# DO LOGISTIC REGRESSION, TREE MODELS, KNM MODELS, SUPPORT VECTOR, RANDOM FOREST