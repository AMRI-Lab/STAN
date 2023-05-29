import os
import numpy as np
import SimpleITK as sitk

num_class = 9
TruePath = None
InferPath = None
SavePath = InferPath

TrueList = os.listdir(TruePath)
TrueList.sort(reverse = False)
InferList = os.listdir(InferPath)
InferList.sort(reverse = False)

accuracy = np.zeros((len(InferList), num_class))
precision = np.zeros((len(InferList), num_class))
dice = np.zeros((len(InferList), num_class))
sensitivity = np.zeros((len(InferList), num_class))
specificity = np.zeros((len(InferList), num_class))

for NumIndex in range(len(InferList)):
    print('TrueMask:', os.path.join(TruePath, TrueList[NumIndex]))
    mask_path = os.path.join(TruePath, TrueList[NumIndex])
    mask = sitk.ReadImage(mask_path)
    TrueMask = sitk.GetArrayFromImage(mask)
    print('InferMask:', os.path.join(InferPath, InferList[NumIndex]))
    InferMask = sitk.ReadImage(os.path.join(InferPath, InferList[NumIndex]))
    InferMask = sitk.GetArrayFromImage(InferMask)

    print(TrueMask.shape, InferMask.shape)
    newtrue_mask = np.zeros(TrueMask.shape + (num_class,))
    newinfer_mask = np.zeros(InferMask.shape + (num_class,))
    for i in range(num_class):
        newtrue_mask[TrueMask == i,i] = 1
        newinfer_mask[InferMask == i, i] = 1
    TrueMask = newtrue_mask
    InferMask = newinfer_mask

    for i in range(num_class):
        TP = np.sum(TrueMask[:, :, :, i] * InferMask[:, :, :, i])
        FP = np.sum(InferMask[:, :, :, i]) - TP
        FN = np.sum(TrueMask[:, :, :, i]) - TP
        TN = InferMask.shape[0] * InferMask.shape[1] * InferMask.shape[2] - FN - FP - TP
        accuracy[NumIndex][i] = (TP + TN) / (TP + TN + FP + FN)
        precision[NumIndex][i] = TP / (TP + FP)
        sensitivity[NumIndex][i] = TP / (TP + FN)
        dice[NumIndex][i] = 2 * TP / (2 * TP + FP + FN)
        specificity[NumIndex][i] = TN / (FP + TN)

np.savetxt(SavePath + '/eva_accuracy.txt', accuracy, delimiter = '\t')
np.savetxt(SavePath + '/eva_precision.txt', precision, delimiter = '\t')
np.savetxt(SavePath + '/eva_sensitivity.txt', sensitivity, delimiter = '\t')
np.savetxt(SavePath + '/eva_dice.txt', dice, delimiter = '\t')
np.savetxt(SavePath + '/eva_specificity.txt', specificity, delimiter = '\t')
