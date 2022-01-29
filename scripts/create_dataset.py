# Télécharger le dataset depuis un GDrive
# Split en train et test
# Enregister dans "assets/data"


from sklearn.model_selection import train_test_split
import gdown
import pandas as pd
import numpy as np

from config import Config

# Set seed
np.random.seed(Config.RANDOM_SEED)

# ./assets/original_datasets et ./assets/data
Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Telecharger notre fichier
gdown.download("https://drive.google.com/uc?id=1-CumMzbuVqwxtoy_QeTKjjAvnqLAZ9gl",
               str(Config.ORIGINAL_DATASET_FILE_PATH))

# Charger le dataset original
df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))

df_train, df_test = train_test_split(
    df, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED)


df_train.to_csv(str(Config.DATASET_PATH/"train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH/"test.csv"), index=None)
