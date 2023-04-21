import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path






pitches = pd.read_csv(r'C:/Users/gosho/OneDrive/Documents/SWAC Project/k_prob.csv')

#to_drop = ['game_pk','game_date','at_bat_number','pitch_number','pitch_type','pitcher_name','pitcher','batter','catcher','stand','p_throws','balls','strikes','broadcast']
#pitches.drop(to_drop, inplace=True, axis=1)
pitches.description[pitches['description'] == 'called_strike'] = 0
pitches.description[pitches['description'] == 'ball'] = 1
pitches['description'].value_counts()
pitches.dropna(inplace=True, axis=0) # drop all non number values

#This is the normalisation section
new_top = pitches['sz_top'].mean()
new_bot = pitches['sz_bot'].mean()

normalised_values = []

for index, row in pitches.iterrows():
    if row['plate_z'] > row['sz_top']:
        normalised_values.append(row['plate_z'] - row['sz_top'] + new_top)
    
    elif row['plate_z'] < row['sz_bot']:
        normalised_values.append(row['plate_z'] / row['sz_bot'] * new_bot)

    else:
        normalised_values.append((row['plate_z']-row['sz_bot']) / (row['sz_top']-row['sz_bot']) * (new_top-new_bot) + new_bot)

pitches.insert(19,'Normalised_Values',normalised_values, True)
#end of section

p_pos = ['plate_x', 'Normalised_Values','sz_top','sz_bot','zone']
X = pitches[p_pos] # Features
y = pitches.description # Target variable
y = y.astype("str")

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size= 0.10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))



from sklearn.model_selection import cross_val_score

print("Array of scores for the Kfold = 10 validation")
print(cross_val_score(classifier,X,y, cv = 10))

print("Average score for K = 10 KFold validation")
print(np.mean(cross_val_score(classifier,X,y, cv = 10)))

proba = classifier.predict_proba(X)
print(proba)

ball_proba = []
for i in proba:
    ball_proba.append(i[1])
pitches.insert(20,'Ball_Probability',ball_proba, True)



