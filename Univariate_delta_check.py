import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

#Specify datasets
df_5 = pd.read_excel(
    io='../Data files/Exclude/Reformatted/5.xlsx',
    usecols="L:O,X"
)
df_13 = pd.read_excel(
    io='../Data files/Exclude/Reformatted/13.xlsx',
    usecols="L:O,X"
)
df_29 = pd.read_excel(
    io='../Data files/Exclude/Reformatted/29.xlsx',
    usecols="L:O,X"
)
df_43 = pd.read_excel(
    io='../Data files/Exclude/Reformatted/43.xlsx',
    usecols="L:O,X"
)

tprs_5 = []
tprs_13 = []
tprs_29 = []
tprs_43 = []
fprs_5 = []
fprs_13 =[]
fprs_29 = []
fprs_43 = []

tpr_list = [tprs_5,tprs_13,tprs_29,tprs_43]
fpr_list = [fprs_5,fprs_13,fprs_29,fprs_43]
Dataframes = [df_5,df_13,df_29,df_43]
mean_fpr = np.linspace(0, 1, 101)

Analyte = 'Potassium Diff'

for j, frame in enumerate(Dataframes):
    max_value = frame[Analyte].max().max()
    min_value = frame[Analyte].min().min()
    for i in np.linspace(min_value, max_value, 100):
        # Find indices of values greater than i in analyte column
        positive_indices = frame[frame[Analyte] >= i].index.tolist()
        negative_indices = frame[frame[Analyte] < i].index.tolist()

        predicted_positive = frame.iloc[positive_indices, [4]]
        predicted_negative = frame.iloc[negative_indices, [4]]

        true_positive = (predicted_positive == 1).sum().sum()
        false_positive = (predicted_positive == 0).sum().sum()
        true_negative = (predicted_negative == 0).sum().sum()
        false_negative = (predicted_negative == 1).sum().sum()

        tpr = true_positive/(true_positive + false_negative)
        fpr = false_positive/(false_positive + true_negative)

        tpr_list[j].append(tpr)
        fpr_list[j].append(fpr)

fprs_5.reverse()
tprs_5.reverse()
interp_tpr_5 = np.interp(mean_fpr, fprs_5, tprs_5)

# print(mean_fpr)
# print()
# print(interp_tpr_5)
# print(interp_tpr_5[1])
# print(interp_tpr_5)

AUC_5 = auc(fprs_5, tprs_5)
AUC_5_interp = auc(mean_fpr, interp_tpr_5)
AUC_13 = auc(fprs_13, tprs_13)
AUC_29 = auc(fprs_29, tprs_29)
AUC_43 = auc(fprs_43, tprs_43)

# print((AUC_5+AUC_13+AUC_29+AUC_43)/4)

# Plot the ROC curve
# plt.figure()
# plt.plot(fprs_5, tprs_5, color='darkorange', lw=2, label='AUC 5 = %0.4f' % AUC_5)
# plt.plot(mean_fpr, interp_tpr_5, color='black', lw=2, label='AUC 5 interp = %0.4f' % AUC_5_interp)
# # plt.plot(fprs_13, tprs_13, color='red', lw=2, label='AUC 13 = %0.4f' % AUC_13)
# # plt.plot(fprs_29, tprs_29, color='green', lw=2, label='AUC 29 = %0.4f' % AUC_29)
# # plt.plot(fprs_43, tprs_43, color='blue', lw=2, label='AUC 43 = %0.4f' % AUC_43)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('%s ROC Curve' % Analyte)
# plt.legend(loc="lower right")
# plt.show()