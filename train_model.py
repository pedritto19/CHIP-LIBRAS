import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Carregar dataset
csv_file = "gestures_dataset.csv"
df = pd.read_csv(csv_file, encoding="utf-8")

# Garantir que estamos usando apenas as 65 features corretas
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])  # Remover timestamp
if "flipped" in df.columns:
    df = df.drop(columns=["flipped"])  # Remover flag de espelhamento

# Separar features (X) e rótulos (y)
X = df.iloc[:, :-1].values  # Features corretas: 63 coordenadas + movimento
y = df.iloc[:, -1].values   # Classe do gesto

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Avaliação do modelo
accuracy = clf.score(X_test, y_test)
print(f"🎯 Precisão do modelo: {accuracy:.2f}")

# Salvar modelo treinado
model_file = "gestures_model.pkl"
with open(model_file, "wb") as f:
    pickle.dump(clf, f)

print(f"✅ Modelo treinado e salvo em {model_file}!")
