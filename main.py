import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregar o dataset
df = pd.read_csv('iris.csv', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Visualizar dados
sns.pairplot(df, hue='species')
plt.show()

# Codificar rótulos
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Separar características e rótulos
X = df.drop('species', axis=1)
y = df['species']

# Normalizar dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=5)

# Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {test_acc:.2f}')
