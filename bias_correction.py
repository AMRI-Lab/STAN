import random
from sklearn.linear_model import LinearRegression
import numpy as np
import torch.nn as nn
import os,torch
import matplotlib.pyplot as plt
from config import opt
from sklearn.metrics import mean_absolute_error
from scipy import stats
from load_qsm import TrainDataset, test_aug
if opt.model == "SKC_BF_Atte_Crop":
    from SKC_BF_Atte_Crop import SKC_BF_Atte_Crop

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

random.seed(0)
files = os.listdir(opt.train_img_folder)
random.shuffle(files)
whole_data_length = len(files)
cut_point_1 = int(whole_data_length * 0.8)
cut_point_2 = int(whole_data_length * 1)
train_set = files[:cut_point_1] + files[cut_point_2:]
train_data = TrainDataset(img_root = opt.train_img_folder, mask_root = opt.train_mask_folder, excel_path = opt.excel_path, file_list = train_set, num_class = opt.num_classes, transform = test_aug)
test_set = sorted(os.listdir(opt.test_img_folder))
test_data = TrainDataset(img_root = opt.test_img_folder, mask_root = opt.test_mask_folder, excel_path = opt.excel_path_2, file_list = test_set, num_class = opt.num_classes, transform = test_aug)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if opt.model == 'SKC_BF_Atte_Crop':
        model = SKC_BF_Atte_Crop(in_channels=opt.inplace, num_filters=opt.num_filters, class_num=opt.num_classes, dropout_rate=opt.dropout_rate, 
                atte_drop_rate=opt.atte_drop_rate, nblock=opt.nblock_end, slice_num=opt.slice_num, use_gender=opt.use_gender).cuda()

model.load_state_dict(torch.load(os.path.join(opt.output_dir+opt.model_name))['state_dict'])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=1, pin_memory=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=1, pin_memory=False, drop_last=False)

train_out_list, train_targ_list, test_ID_list, test_out_list, test_targ_list = [], [], [], [], []

model.eval() 
print('======= start bias correction =============')
os.makedirs(opt.save_dir, exist_ok = True)

with torch.no_grad():
    for _, (img, name, age, gender, _) in enumerate(train_loader):
        print('name:', name[0])
        input = img.to(device)
        # mask = mask.to(device)
        age = torch.from_numpy(np.expand_dims(age,axis=1))
        age = age.type(torch.FloatTensor).to(device)
        if opt.use_gender:
            gender = torch.from_numpy(np.expand_dims(gender,axis=1))
            gender = gender.type(torch.FloatTensor).cuda(non_blocking=True)
        else:
            gender = None

        segout, ageout = model(input, gender)

        train_out_list.append(torch.flatten(ageout).cpu().numpy())
        train_targ_list.append(torch.flatten(age).cpu().numpy())
        # ID_list.append(name[0])

    for _, (img, name, age, gender, _) in enumerate(test_loader):
        print('name:', name[0])
        
        input = img.to(device)
        # mask = mask.to(device)
        age = torch.from_numpy(np.expand_dims(age,axis=1))
        age = age.type(torch.FloatTensor).to(device)
        if opt.use_gender:
            gender = torch.from_numpy(np.expand_dims(gender,axis=1))
            gender = gender.type(torch.FloatTensor).cuda(non_blocking=True)
        else:
            gender = None

        segout, ageout = model(input, gender)
        test_out_list.append(torch.flatten(ageout).cpu().numpy())
        test_targ_list.append(torch.flatten(age).cpu().numpy())
        test_ID_list.append(name[0])
        
train_targ = np.asarray(train_targ_list)
train_out = np.asarray(train_out_list)
test_targ = np.asarray(test_targ_list)
test_out = np.asarray(test_out_list)
test_ID = np.asarray(test_ID_list)


bias_correction = LinearRegression()
bias_correction.fit(train_targ, (train_out - train_targ))
print('intercept:', bias_correction.intercept_) 
print('coef:', bias_correction.coef_)       
bias_corrected = bias_correction.predict(test_targ)
bias_corrected = np.squeeze(bias_corrected, axis = 1)
test_targ = np.squeeze(test_targ, axis = 1)
test_out = np.squeeze(test_out, axis = 1)
test_corrected = test_out - bias_corrected
ori_errors = test_out - test_targ
ori_mae = np.mean(np.abs(ori_errors))
cor_errors = test_corrected - test_targ
cor_mae = np.mean(np.abs(cor_errors))
# cor_mae = np.mean(abs_errors)


out_file = open(os.path.join(opt.output_dir, 'bias_correction.txt'), 'w', encoding='utf-8')
print('===============================================================\n')

MAE_out = 'Mean Absolute Error: original MAE = ' + str(ori_mae) + ', corrected MAE = ' + str(cor_mae)
print(MAE_out)
out_file.write(MAE_out + '\n')
STD_err = 'STD_err = ' + str(np.std(cor_errors))
print(STD_err)
out_file.write(STD_err + '\n')
corrcoef = ' CC: ' + str(np.corrcoef(test_targ, test_corrected))
print(corrcoef)
out_file.write(corrcoef + '\n')
PADspearmanr = 'PAD spear man cc' + str(stats.spearmanr(cor_errors, test_targ,axis=1))
print(PADspearmanr)
out_file.write(PADspearmanr + '\n')
spearmanr = 'spear man cc' + str(stats.spearmanr(test_corrected, test_targ,axis=1))
print(spearmanr)
out_file.write(spearmanr + '\n')
meanpad = 'mean pad:' + str(np.mean(cor_errors))
print(meanpad)
out_file.write(meanpad + '\n')

title = 'ID\t\t\tpredict\t\t\tcorrected\t\ttruth\t\tori_errors\t\tcor_errors\n'
print(title)
out_file.write(title)

for i in range(len(test_ID)):
    content = str(test_ID[i][0:12]) + '\t\t' + str(test_out[i]) + '\t\t' + str(test_corrected[i]) + '\t\t' + str(test_targ[i]) + '\t\t' + str(ori_errors[i]) + '\t\t' + str(cor_errors[i])
    print(content)
    out_file.write(content + '\n')        
out_file.close()

print('\n =================================================================')

plt.figure()
lx = np.arange(np.min(test_targ),np.max(test_targ))
plt.plot(lx,lx,color='red',linestyle='--')
plt.scatter(test_targ,test_corrected)
plt.xlabel('Chronological Age')
plt.ylabel('Corrected brain age')
print("save png")
plt.savefig(os.path.join(opt.output_dir, 'corrected' + opt.plot_name))