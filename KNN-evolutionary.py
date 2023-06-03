import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from statistics import mean
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV

#Specify features to be used
cols = "D,I,M"

#Specify training datasets
x_train = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Train-excl-29.xlsx',
    usecols=cols
)
y_train = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Train-excl-29.xlsx',
    usecols="U"
)

y_train = np.ravel(y_train)

#Specify test datasets
x_test_5 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-5.xlsx',
    usecols=cols
)
y_test_5 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-5.xlsx',
    usecols='U'
)

x_test_13 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-13.xlsx',
    usecols=cols
)
y_test_13 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-13.xlsx',
    usecols='U'
)

x_test_29 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-29.xlsx',
    usecols=cols
)
y_test_29 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-29.xlsx',
    usecols='U'
)

x_test_43 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-43.xlsx',
    usecols=cols
)
y_test_43 = pd.read_excel(
    io='C:/Users/CJY/OneDrive - Singapore Institute Of Technology/SIT Year 3/Capstone/Logbook/Data files/Exclude/Val-excl-43.xlsx',
    usecols='U'
)

y_test_5 = np.ravel(y_test_5)
y_test_13 = np.ravel(y_test_13)
y_test_29 = np.ravel(y_test_29)
y_test_43 = np.ravel(y_test_43)

# Define the parameter grid for KNeighborsClassifier
param_grid = {
    'n_neighbors': np.arange(1, 31),
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
}

model = KNeighborsClassifier(n_neighbors=3)

# Configure the cross-validation procedure
cv = StratifiedKFold(n_splits=2)

# Initialize the EvolutionaryAlgorithmSearchCV
es = EvolutionaryAlgorithmSearchCV(
    estimator=model,
    params=param_grid,
    scoring='accuracy',
    cv=cv,
    verbose=1,
    population_size=50,
    gene_mutation_prob=0.10,
    gene_crossover_prob=0.5,
    tournament_size=3,
    generations_number=2,
    n_jobs=4,
)

# Fit the EvolutionaryAlgorithmSearchCV to the training data
es.fit(x_train, y_train)

# Get the probabilities
y_pred_proba_5 = es.predict_proba(x_test_5)
y_pred_proba_13 = es.predict_proba(x_test_13)
y_pred_proba_29 = es.predict_proba(x_test_29)
y_pred_proba_43 = es.predict_proba(x_test_43)

# Calculate the FPR, TPR, and threshold using roc_curve
fpr_5, tpr_5, threshold = roc_curve(y_test_5, y_pred_proba_5[:, 1])
fpr_13, tpr_13, threshold = roc_curve(y_test_13, y_pred_proba_13[:, 1])
fpr_29, tpr_29, threshold = roc_curve(y_test_29, y_pred_proba_29[:, 1])
fpr_43, tpr_43, threshold = roc_curve(y_test_43, y_pred_proba_43[:, 1])

# Calculate the AUC
roc_auc_5 = auc(fpr_5, tpr_5)
roc_auc_13 = auc(fpr_13, tpr_13)
roc_auc_29 = auc(fpr_29, tpr_29)
roc_auc_43 = auc(fpr_43, tpr_43)

print("5: " + str(format(roc_auc_5, ".4f")))
print("13: " + str(format(roc_auc_13, ".4f")))
print("29: " + str(format(roc_auc_29, ".4f")))
print("43: " + str(format(roc_auc_43, ".4f")))
AUCs = [roc_auc_5, roc_auc_13, roc_auc_29, roc_auc_43]
print("Average: " + str(format(mean(AUCs), ".4f")))

# Plot the ROC curve
plt.figure()
plt.plot(fpr_43, tpr_43, color='darkorange', lw=2, label='ROC curve (AUC = %0.4f)' % roc_auc_43)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()