import joblib
import numpy as np

loaded_clf = joblib.load("Project1_F2025.joblib")

testdata = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])

#testdata = np.delete(testdata, 2, axis=1)
predictions = loaded_clf.predict(testdata)
print(predictions)