import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y  # Placeholder for chained mode.
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        # Train using chained label if available.
        if hasattr(data, 'y_chain_train'):
            self.mdl.fit(data.X_train, data.y_chain_train)
        else:
            self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        # If chained labels exist, evaluate stage-by-stage.
        if hasattr(data, 'y_chain_test'):
            total_score = 0
            detailed_results = []
            N = len(self.predictions)
            for i in range(N):
                true_chain = data.y_chain_test[i].split('+')
                pred_chain = self.predictions[i].split('+')
                instance_details = f"Instance {i+1}:\n"
                instance_details += f"  True Labels: Type2: {true_chain[0]}"
                if len(true_chain) > 1:
                    instance_details += f", Type3: {true_chain[1]}"
                if len(true_chain) > 2:
                    instance_details += f", Type4: {true_chain[2]}\n"
                else:
                    instance_details += "\n"
                    
                instance_details += f"  Predicted Labels: Type2: {pred_chain[0]}"
                if len(pred_chain) > 1:
                    instance_details += f", Type3: {pred_chain[1]}"
                if len(pred_chain) > 2:
                    instance_details += f", Type4: {pred_chain[2]}\n"
                else:
                    instance_details += "\n"
                    
                score = 0
                # Stage 1: Evaluate Type2.
                if pred_chain[0] == true_chain[0]:
                    score = 1
                    instance_details += "  Stage 1 (Type2): Correct\n"
                else:
                    instance_details += "  Stage 1 (Type2): Incorrect\n"
                    instance_details += "  Final Accuracy for Instance: 0%\n"
                    detailed_results.append(instance_details)
                    continue  # Stop evaluation for this instance.
                    
                # Stage 2: Evaluate Type2+Type3.
                if len(true_chain) > 1 and len(pred_chain) > 1:
                    if pred_chain[1] == true_chain[1]:
                        score = 2
                        instance_details += "  Stage 2 (Type2+Type3): Correct\n"
                    else:
                        instance_details += "  Stage 2 (Type2+Type3): Incorrect\n"
                        instance_details += f"  Final Accuracy for Instance: {score/len(true_chain)*100:.0f}%\n"
                        detailed_results.append(instance_details)
                        total_score += score/len(true_chain)
                        continue
                        
                # Stage 3: Evaluate Type2+Type3+Type4.
                if len(true_chain) > 2 and len(pred_chain) > 2:
                    if pred_chain[2] == true_chain[2]:
                        score = 3
                        instance_details += "  Stage 3 (Type2+Type3+Type4): Correct\n"
                    else:
                        instance_details += "  Stage 3 (Type2+Type3+Type4): Incorrect\n"
                instance_score = score / len(true_chain)
                instance_details += f"  Final Accuracy for Instance: {instance_score*100:.0f}%\n"
                detailed_results.append(instance_details)
                total_score += instance_score
            
            overall_chained_accuracy = total_score / N
            #print("\n".join(detailed_results))
            print("Overall Chained Accuracy: {:.2%}".format(overall_chained_accuracy))
            print("\nFull Classification Report for chained label (y_chain):")
            print(classification_report(data.y_chain_test, self.predictions, zero_division=0))
        else:
            print(classification_report(data.y_test, self.predictions, zero_division=0))


    def data_transform(self) -> None:
        ...

