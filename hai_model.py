import pandas as pd
from TaPR_pkg import etapr
from sklearn.ensemble import RandomForestClassifier

# Prepare train data
train_data1 = pd.read_csv("hai_rho/train1.csv", sep=";", engine="python")
train_data2 = pd.read_csv("hai_rho/train2.csv", sep=";", engine="python")

total_train_data = pd.concat([train_data1, train_data2], axis=0).reset_index(drop=True)

# Prepare test data
test_data1 = pd.read_csv("hai_rho/test1.csv", sep=";", engine="python")
test_data2 = pd.read_csv("hai_rho/test2.csv", sep=";", engine="python")

total_test_data = pd.concat([test_data1, test_data2], axis=0).reset_index(drop=True)

# Setup data for detection model to predict attack
LABEL_COLUMN = "attack"
FEATURE_COLUMN = total_train_data.columns.drop(["time", "attack", "attack_P1", "attack_P2", "attack_P3"])

train_x = total_train_data.loc[:,FEATURE_COLUMN]
train_y = total_train_data.loc[:,[LABEL_COLUMN]]

test_x = total_test_data.loc[:,FEATURE_COLUMN]
test_y = total_test_data.loc[:,[LABEL_COLUMN]]

# Fit model to train data
RF_Model = RandomForestClassifier(n_estimators=1024)
RF_Model.fit(train_x, train_y)
print("RF model fitted")

# Predict result for test data
rf_result = RF_Model.predict(test_x)
print(rf_result)

# Evaluate test result
TaPR = etapr.evaluate_haicon(anomalies=test_y, predictions=rf_result)
print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
