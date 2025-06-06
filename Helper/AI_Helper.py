from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Get_Dataset(path_of_dataset, name_of_dataset) -> pd.DataFrame:
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(path_of_dataset, file_name=name_of_dataset)
    dataset = pd.read_csv(name_of_dataset)
    return dataset


