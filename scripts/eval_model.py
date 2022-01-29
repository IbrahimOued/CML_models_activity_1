import pickle
import json
from numpy import average
import pandas as pd
from config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
Config.MODELS_V2_PATH.mkdir(parents=True, exist_ok=True)


df_x_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
df_y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

# Nous allons scaler les données pour la normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(df_x_test)
y_true = df_y_test.pop('Class')

# Chargement du modèle
model = pickle.load(open(str(Config.MODELS_V2_PATH/"model_v2.pk"), mode='rb'))
# Prediction
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='micro')
recall_score = recall_score(y_true, y_pred, average='micro')
f1_score = f1_score(y_true, y_pred, average='micro')
with open(str(Config.METRICS_V2_FILE_PATH), mode='w') as f:
    json.dump(dict(accuracy=accuracy, precision=precision, f1_score=f1_score), f)
