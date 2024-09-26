import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('student_performance_prediction.csv')
# Criando uma cópia dos dados para manipulação, mantendo os dados originais intactos
copyData = data.copy()

# Tratar valores numéricos nulos, trocando o nulo pelo valor da média
count = 0
for variavel in copyData.columns:
    if count == 0:
        count += 1
    elif count < 4:
        copyData[variavel] = copyData[variavel].fillna(copyData[variavel].mean())
        count += 1

    
# Colocar valores infinitos como Nulos
#copyData['Study Hours per Week'] = copyData['Study Hours per Week'].replace([np.inf, -np.inf], np.nan)

# Assinalando valores para dados não numéricos (variáveis categóricas)
copyData['Passed'] = copyData['Passed'].map({'Yes': 1, 'No': 0})
copyData['Participation in Extracurricular Activities'] = copyData['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})
# Nível de escolaridade dos pais é uma variável categórica nominal
dummies = pd.get_dummies(copyData['Parent Education Level'])
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(copyData[['Parent Education Level']]).toarray()
# Visualizar as categorias da variável em colunas
feature_labels = ohe.categories_
# Criando um array
feature_labels = np.array(feature_labels).ravel()
# Transformando os arrays em um data frame
features = pd.DataFrame(feature_array, columns = feature_labels)

# Juntando a variavel dummy e as colunas OHE para a base de dados original
data_new = pd.concat([copyData, dummies, features], axis=1)
data_new = data_new.drop(columns='Parent Education Level', axis=1)

print(data_new.head())


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

