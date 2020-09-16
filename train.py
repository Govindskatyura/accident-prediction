import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 
import seaborn as sns
from  sklearn.ensemble import RandomForestClassifier
sns.set(style="whitegrid", font_scale=1.5)

train_df = pd.read_csv("Train_data.csv")
test_df = pd.read_csv("Test_Data.csv")
train_df.drop("Unnamed: 0",axis=1,inplace=True)
test_df.drop("Unnamed: 0",axis=1,inplace=True)
train_df.drop("gender",axis=1,inplace= True)
test_df.drop("gender",axis=1,inplace= True)

code = LabelEncoder()
registrationMode_label_encoder = LabelEncoder()
planName_label_encoder = LabelEncoder()
clientType_label_encoder = LabelEncoder()
country_label_encoder = LabelEncoder()
# label_encoder object knows how to understand word labels. 

train_df['code']= code.fit_transform(train_df['code'])
train_df['registrationMode']= registrationMode_label_encoder.fit_transform(train_df['registrationMode'])
train_df['planName']= planName_label_encoder.fit_transform(train_df['planName'])
train_df['clientType']= clientType_label_encoder.fit_transform(train_df['clientType'])
country_label_encoder.fit(train_df['country'].append(test_df['country']))

train_df['country'] = country_label_encoder.transform(train_df['country'])
test_df['code'] = code.transform(test_df['code']) 
test_df['registrationMode'] = registrationMode_label_encoder.transform(test_df['registrationMode']) 
test_df['planName'] = planName_label_encoder.transform(test_df['planName'])
test_df['clientType'] = clientType_label_encoder.transform(test_df['clientType']) 
test_df['country'] = country_label_encoder.transform(test_df['country'])

X = train_df.drop('accident',axis=1)
y = train_df['accident']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#######################################################
###########    wighted logistic regression  ###########
#######################################################

weights = dict(zip((y_train.value_counts()/X_train.shape[0]).index,(y_train.value_counts()/X_train.shape[0]).values))

# define model
lg2 = LogisticRegression(class_weight=weights)
# fit it
lg2.fit(X_train,y_train)
# test
y_pred = lg2.predict(X_test)
# performance
fp = open("log.txt","w")
fp.write(f'Accuracy Score: {accuracy_score(y_test,y_pred)} \n')
fp.write(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)} \n')
fp.write(f'Area Under Curve: {roc_auc_score(y_test, y_pred)} \n')
fp.write(f'Recall score: {recall_score(y_test,y_pred)} \n')
importance = lg2.coef_[0]
# summarize feature importance
plt.figure(figsize=(10,8))
plt.title("Feature Importance")
plt.xlabel("features")
plt.ylabel("importance to predict  classes 0 or 1")
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("logistic_regression.png")

#######################################################
##########     Random Forest classifier      ##########
#######################################################
clf = RandomForestClassifier(n_estimators=10, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
fp.write("##"*50)
fp.write("\n Random Forest classifier\n")
fp.write('Classification Report:')
fp.write(classification_report(y_test, y_pred))
# summarize feature importance
importance = clf.feature_importances_
plt.figure(figsize=(10,8))
plt.title("Feature Importance")
plt.xlabel("features")
plt.ylabel("importance to predict  classes 0 or 1")
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("Random_Forest_classifier.png")
