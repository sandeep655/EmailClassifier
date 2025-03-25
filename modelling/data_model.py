import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # If chained labels exist, use them.
        if "y_chain" in df.columns:
            # Filter based on Type2 (y2) to ensure enough instances.
            y_series = pd.Series(df["y2"])
            good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
            mask = y_series.isin(good_y_value)
            X_good = X[mask]
            y_chain = df["y_chain"][mask]
            y_chain1 = df["y_chain1"][mask]
            y_chain2 = df["y_chain2"][mask]
            y_chain3 = df["y_chain3"][mask]
            # Further filter: keep only rows whose y_chain class has at least 2 instances.
            chain_counts = y_chain.value_counts()
            good_chain_values = chain_counts[chain_counts >= 2].index
            mask2 = y_chain.isin(good_chain_values)
            X_good = X_good[mask2]
            y_chain = y_chain[mask2]
            y_chain1 = y_chain1[mask2]
            y_chain2 = y_chain2[mask2]
            y_chain3 = y_chain3[mask2]
            new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
            (self.X_train, self.X_test,
            self.y_chain_train, self.y_chain_test,
            self.y_chain1_train, self.y_chain1_test,
            self.y_chain2_train, self.y_chain2_test,
            self.y_chain3_train, self.y_chain3_test) = train_test_split(
                X_good, y_chain, y_chain1, y_chain2, y_chain3,
                test_size=new_test_size, random_state=0, stratify=y_chain)
            self.embeddings = X_good
                        # Reset indices so that integer-based indexing works
            self.y_chain_train = self.y_chain_train.reset_index(drop=True)
            self.y_chain_test = self.y_chain_test.reset_index(drop=True)
            self.y_chain1_train = self.y_chain1_train.reset_index(drop=True)
            self.y_chain1_test = self.y_chain1_test.reset_index(drop=True)
            self.y_chain2_train = self.y_chain2_train.reset_index(drop=True)
            self.y_chain2_test = self.y_chain2_test.reset_index(drop=True)
            self.y_chain3_train = self.y_chain3_train.reset_index(drop=True)
            self.y_chain3_test = self.y_chain3_test.reset_index(drop=True)

        else:
            y = df.y.to_numpy()
            y_series = pd.Series(y)
            good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
            if len(good_y_value) < 1:
                print("None of the classes have more than 3 records: Skipping ...")
                self.X_train = None
                return
            y_good = y[y_series.isin(good_y_value)]
            X_good = X[y_series.isin(good_y_value)]
            new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good)
            self.y = y_good
            self.classes = good_y_value
            self.embeddings = X_good
            
    def get_embeddings(self):
        return self.embeddings

    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train

