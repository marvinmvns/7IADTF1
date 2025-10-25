
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Carregar o dataset
df = pd.read_csv('./dataset/heart_attack_prediction_indonesia.csv')

# Separar features e target
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

# Identificar variáveis categóricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Codificar variáveis categóricas
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# Normalização das features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Treinar o melhor modelo (Random Forest)
best_model = RandomForestClassifier(random_state=42, n_estimators=100)
best_model.fit(X_train_scaled, y_train)

# Salvar o modelo, scaler e label encoders
with open('artifacts/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('artifacts/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('artifacts/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
    
with open('artifacts/feature_names.pkl', 'wb') as f:
    pickle.dump(X_encoded.columns.tolist(), f)

print("Modelo, scaler e label encoders salvos com sucesso em 'artifacts/'")
