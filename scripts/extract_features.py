import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
df_test = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))

# Nous allons procéder à un peu de préparation de données
# Replacing NaN with default valuesb
df_train = df_train.fillna(value=-1)

# Dropping columns
del df_train["Track Name"]

# Typecasting
df_train["Artist Name"] = df_train["Artist Name"].astype(str)

# Initializing Encoder
number = preprocessing.LabelEncoder()

# Encoding
df_train["Artist Name"] = number.fit_transform(df_train["Artist Name"])


# Columns used as predictors
X = df_train.drop(["Class"], axis=1)
y = df_train["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=Config.RANDOM_SEED, test_size=Config.TEST_SIZE)


# Enregistrement des features pour train et test
X_train.to_csv(
    str(Config.FEATURES_PATH / "train_features.csv"), index=None)
X_test.to_csv(
    str(Config.FEATURES_PATH / "test_features.csv"), index=None)

# Enregistrement des labels pour train et test
y_train.to_csv(
    str(Config.FEATURES_PATH / "train_labels.csv"), index=None)
y_test.to_csv(
    str(Config.FEATURES_PATH / "test_labels.csv"), index=None)
