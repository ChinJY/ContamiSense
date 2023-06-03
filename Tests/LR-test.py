import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
a = np.array([[3],[4],[7],[9]])
b = np.array([0,1,1,1])

model = LogisticRegression(solver='liblinear', C=10, random_state=0)

model.fit(x, y)

print(model.predict(a))
print(model.score(a, b))