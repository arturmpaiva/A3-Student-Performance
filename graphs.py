import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('student_performance_prediction.csv')


# Tratar valores faltantes, se houver (exemplo: preencher com a média)
data['Study Hours per Week'] = data['Study Hours per Week'].fillna(data['Study Hours per Week'].mean())
# Colocar valores infinitos como Nulos
data['Study Hours per Week'] = data['Study Hours per Week'].replace([np.inf, -np.inf], np.nan)

(data.head())

(data.info())

(data.describe())

# Criando uma cópia dos dados para manipulação, mantendo os dados originais intactos
copyData = data.copy()

# Assinalando valores para dados não numéricos
copyData['Passed'] = copyData['Passed'].map({'Yes': 1, 'No': 0})
copyData['Participation in Extracurricular Activities'] = copyData['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})

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
sns.displot(copyData['Attendance Rate'])
plt.title('Frequência')
plt.show()



