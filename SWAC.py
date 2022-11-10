import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from os import path
import pandas as pd


DATA_DIR = "/Users/localuser/Desktop/SWAC Code"

pitches = pd.read_csv(path.join(DATA_DIR, 'k_prob.csv'))

input = [pitches.head(210637).iloc[:,16],pitches.head(210637).iloc[:,17]]
output = pitches.head(210637).iloc[10]

for i in range (1):
    for j in range (210637):   #210637
        if(output[j:1] == "called_strike"):
           output[j:1] = 1
        elif (output[j:1] == "ball"):
            output[j:1] = 0

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(input,output)
model.predict_proba(input)


 
