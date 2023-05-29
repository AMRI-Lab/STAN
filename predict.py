import numpy as np
import torch.nn as nn
import os,shutil,torch
import matplotlib.pyplot as plt
from config import opt
from sklearn.metrics import mean_absolute_error
from scipy import stats
from loss import CEDice, DiceLoss, CELDice, FLDice, UncertaintyLoss, EdgeFLDice
from load_qsm import TrainDataset, dice_coeff, SaveResult, test_aug
if opt.model == "SKC_BF_Atte_Crop":
    from SKC_BF_Atte_Crop import SKC_BF_Atte_Crop

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def main():

    test_set = sorted(os.listdir(opt.test_img_folder))
    test_data = TrainDataset(img_root = opt.test_img_folder, mask_root = opt.test_mask_folder, excel_path = opt.excel_path_2, file_list = test_set, num_class = opt.num_classes, transform = test_aug)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if opt.model == 'SKC_BF_Atte_Crop':
        model = SKC_BF_Atte_Crop(in_channels=opt.inplace, num_filters=opt.num_filters, class_num=opt.num_classes, dropout_rate=opt.dropout_rate, 
                atte_drop_rate=opt.atte_drop_rate, nblock=opt.nblock_end, slice_num=opt.slice_num, use_gender=opt.use_gender).cuda()

    seg_loss_func_dict = {'CE': nn.CrossEntropyLoss().to(device),
                     'CEDice': CEDice(dice_weight=opt.dice_weight,num_classes=opt.num_classes).to(device),
                     'CELDice': CELDice(dice_weight=opt.dice_weight,num_classes=opt.num_classes).to(device),
                     'DiceLoss': DiceLoss(n_classes=opt.num_classes).to(device),
                      'FLDice': FLDice(gamma=opt.gamma, alpha=1, dice_weight=opt.dice_weight, num_classes=opt.num_classes).to(device),
                      'EdgeFLDice': EdgeFLDice(gamma=opt.gamma, alpha=1, dice_weight=opt.dice_weight, num_classes=opt.num_classes).cuda()}
    age_loss_func_dict = {'mae': nn.L1Loss().to(device), 
                        'mse': nn.MSELoss().to(device),
                        'uncertainty': UncertaintyLoss().to(device)}        
    seg_crit = seg_loss_func_dict[opt.seg_loss]
    age_crit = age_loss_func_dict[opt.age_loss]
    model.load_state_dict(torch.load(os.path.join(opt.output_dir + opt.model_name), map_location = torch.device(device))['state_dict'])

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=5, pin_memory=False, drop_last=False)

    test(valid_loader=test_loader, model=model, seg_crit=seg_crit, age_crit=age_crit, device=device, save_path=opt.save_dir)

def test(valid_loader, model, seg_crit, age_crit, device, save_path):

    '''
    [Do Test process according pretrained model]

    Args:
        valid_loader (torch.dataloader): [test set dataloader defined in 'main']
        model (torch CNN model): [pre-trained CNN model, which is used for brain age estimation]
        criterion (torch loss): [loss function defined in 'main']
        device (torch device): [GPU]
        save_npy (bool, optional): [If choose to save predicted brain age in npy format]. Defaults to False.
        npy_name (str, optional): [If choose to save predicted brain age, what is the npy filename]. Defaults to 'test_result.npz'.
        figure (bool, optional): [If choose to plot and save scatter plot of predicted brain age]. Defaults to False.
        figure_name (str, optional): [If choose to save predicted brain age scatter plot, what is the png filename]. Defaults to 'True_age_and_predicted_age.png'.

    Returns:
        [float]: MAE and pearson correlation coeficent of predicted brain age in teset set.
    '''

    Loss = AverageMeter()
    AgeLoss = AverageMeter()
    SegLoss = AverageMeter()
    Dice = AverageMeter()
    MAE = AverageMeter()
    out_list, targ_list, sigma_list, ID_list = [], [], [], []

    model.eval()
    print('======= start prediction =============')
    os.makedirs(opt.save_dir, exist_ok = True)

    with torch.no_grad():
        for _, (img, mask, name, age, gender, bbox) in enumerate(valid_loader):

            input = img.to(device)
            mask = mask.to(device)
            age = torch.from_numpy(np.expand_dims(age,axis=1))
            age = age.type(torch.FloatTensor).to(device)
            if opt.use_gender:
                gender = torch.from_numpy(np.expand_dims(gender,axis=1))
                gender = gender.type(torch.FloatTensor).to(device)
            else:
                gender = None

            segout, ageout = model(input, gender)
            age_loss = age_crit(ageout, age)
            seg_loss = seg_crit(segout, mask.long())
            loss = opt.lam2 * age_loss + opt.lam1 * seg_loss

            mae = metric(ageout.detach(), age.detach().cpu())
            dice = dice_coeff(segout.detach(), mask.long(), num_classes = segout.size(1)).cpu().data.numpy()
            Loss.update(loss, img.size(0))
            AgeLoss.update(age_loss, img.size(0))
            SegLoss.update(seg_loss, img.size(0))
            Dice.update(dice, img.size(0))
            MAE.update(mae, img.size(0))

            out_list.append(torch.flatten(ageout).cpu().numpy()[0])
            targ_list.append(torch.flatten(age).cpu().numpy()[0])
            ID_list.append(name[0])
            SaveResult(opt.test_img_folder, save_path, segout.cpu().data.numpy()[0], name[0], bbox.cpu().numpy()[0])
        
        targ = np.asarray(targ_list)
        out = np.asarray(out_list)
        sig = np.asarray(sigma_list)
        ID = np.asarray(ID_list)
        errors = out - targ
        
        out_file = open(os.path.join(opt.output_dir, 'prediction.txt'), 'w', encoding='utf-8')
        print('===============================================================\n')
        result = ('TEST: [steps {0}]\t'  'AgeLoss {AgeLoss.val:.3f} ({AgeLoss.avg:.3f})\t' 
            'SegLoss {SegLoss.val:.3f} ({SegLoss.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'
            'Dice {Dice.val:.3f} ({Dice.avg:.3f})\t'
            ).format(len(valid_loader), SegLoss=SegLoss, loss=Loss, Dice=Dice, AgeLoss=AgeLoss, MAE=MAE)
        print(result)
        out_file.write(result + '\n')
        STD_err = 'STD_err = ' + str(np.std(errors))
        print(STD_err)
        out_file.write(STD_err + '\n')
        corrcoef = ' CC: ' + str(np.corrcoef(targ,out))
        print(corrcoef)
        out_file.write(corrcoef + '\n')
        PADspearmanr = 'PAD spear man cc' + str(stats.spearmanr(errors,targ,axis=1))
        print(PADspearmanr)
        out_file.write(PADspearmanr + '\n')
        spearmanr = 'spear man cc' + str(stats.spearmanr(out,targ,axis=1))
        print(spearmanr)
        out_file.write(spearmanr + '\n')
        meanpad = 'mean pad:' + str(np.mean(errors))
        print(meanpad)
        out_file.write(meanpad + '\n')

        title = 'ID\t\t\tpredict\t\t\ttruth\t\terrors\n'
        print(title)
        out_file.write(title)
        for i in range(len(ID)):
            content = str(ID[i][0:12]) + '\t\t' + str(out[i]) + '\t\t' + str(targ[i]) + '\t\t' + str(errors[i])
            print(content)
            out_file.write(content + '\n')        
        out_file.close()

if __name__ == "__main__":
    main()
