import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

df_x_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
df_y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

# Nous allons scaler les données pour la normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(df_x_train)
y_train = df_y_train.pop('Class')

model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle.dump(model, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))
