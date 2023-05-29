# from cgi import test
import os
import random
# from tokenize import Double
import torch,json
import datetime
from torch.utils import tensorboard
# import tensorboardX as tensorboard
# import tensorboard
import numpy as np
import torch.nn as nn
from config import opt
# from predict import test
from torch.nn import DataParallel
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from loss import CEDice, DiceLoss, CELDice, FLDice, UncertaintyLoss, EdgeFLDice
from load_qsm import TrainDataset, dice_coeff, train_aug, test_aug
if opt.model == "SKC_BF_Atte_Crop":
    from SKC_BF_Atte_Crop import SKC_BF_Atte_Crop

def main(res):
    
    best_metric = 100
    json_path = os.path.join(opt.output_dir,'hyperparameter.json')
    with open(json_path,'w') as f:
        f.write(json.dumps(vars(opt), ensure_ascii=False, indent=4, separators=(',', ':')))

    files = os.listdir(opt.train_img_folder)
    random.shuffle(files)
    whole_data_length = len(files)
    cut_point_1 = int(whole_data_length * 0.8)
    cut_point_2 = int(whole_data_length * 1)
    train_set = files[:cut_point_1] + files[cut_point_2:]
    val_set = files[cut_point_1: cut_point_2]
    
    train_data = TrainDataset(img_root = opt.train_img_folder, mask_root = opt.train_mask_folder, excel_path = opt.excel_path, 
                              file_list = train_set, num_class = opt.num_classes, transform = train_aug)
    valid_data = TrainDataset(img_root = opt.train_img_folder, mask_root = opt.train_mask_folder, excel_path = opt.excel_path, 
                              file_list = val_set, num_class = opt.num_classes, transform = test_aug)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                            pin_memory=True, drop_last=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                                            pin_memory=True, drop_last=True)
 
    if opt.model == 'SKC_BF_Atte_Crop':
        model = SKC_BF_Atte_Crop(in_channels=opt.inplace, num_filters=opt.num_filters, class_num=opt.num_classes, dropout_rate=opt.dropout_rate, 
                atte_drop_rate=opt.atte_drop_rate, nblock=opt.nblock_end, slice_num=opt.slice_num, use_gender=opt.use_gender).cuda()
    model = DataParallel(model)
    model.apply(weights_init)
    if opt.pretrained_model is not None:
        model.module.load_state_dict(torch.load(opt.pretrained_model)['state_dict'], strict = False)
        print("Pretrained model loaded!")

    seg_loss_func_dict = {'CE': nn.CrossEntropyLoss().cuda(),
                     'CEDice': CEDice(dice_weight=opt.dice_weight,num_classes=opt.num_classes).cuda(),
                     'CELDice': CELDice(dice_weight=opt.dice_weight,num_classes=opt.num_classes).cuda(),
                     'DiceLoss': DiceLoss(n_classes=opt.num_classes).cuda(),
                      'FLDice': FLDice(gamma=opt.gamma, alpha=1, dice_weight=opt.dice_weight, num_classes=opt.num_classes).cuda(),
                      'EdgeFLDice': EdgeFLDice(gamma=opt.gamma, alpha=1, dice_weight=opt.dice_weight, num_classes=opt.num_classes).cuda()}
    age_loss_func_dict = {'mae': nn.L1Loss().cuda(), 
                        'mse': nn.MSELoss().cuda(),
                        'uncertainty':UncertaintyLoss().cuda()}        
    seg_crit = seg_loss_func_dict[opt.seg_loss]
    age_crit = age_loss_func_dict[opt.age_loss]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay, amsgrad=True)
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True)
    saved_metrics, saved_epos = [], []
    num_epochs = int(opt.epochs)
    sum_writer = tensorboard.SummaryWriter(opt.output_dir)
    print("Training takes {} epochs.".format(num_epochs))


    for epoch in range(opt.epochs):

        adjust_learning_rate(optimizer, epoch, opt) 
        train_loss, train_dice, train_mae = train(train_loader=train_loader, model=model, seg_crit=seg_crit, age_crit=age_crit, optimizer=optimizer, epoch=epoch)
        valid_loss, valid_dice, valid_mae = validate( valid_loader=valid_loader, model=model, seg_crit=seg_crit, age_crit=age_crit)        
        for param_group in optimizer.param_groups:
            print("\n*learning rate {:.2e}*\n".format(param_group['lr']))

        sum_writer.add_scalar('train/loss', train_loss, epoch)
        sum_writer.add_scalar('train/mae', train_mae, epoch)
        sum_writer.add_scalar('train/dice', train_dice, epoch)
        sum_writer.add_scalar('valid/loss', valid_loss, epoch)
        sum_writer.add_scalar('valid/mae', valid_mae, epoch)
        sum_writer.add_scalar('valid/dice', valid_dice, epoch)

        valid_metric = valid_mae
        is_best = False
        if valid_metric < best_metric:
            is_best = True
            best_metric = min(valid_metric, best_metric)
            saved_metrics.append(valid_metric)
            saved_epos.append(epoch)
            print('=======>   Best at epoch %d, valid mae %f\n' % (epoch, best_metric))
            
        save_checkpoint({'epoch': epoch,'arch': opt.model,'state_dict': model.module.state_dict(),'optimizer': optimizer.state_dict()}, 
                        is_best, opt.output_dir, model_name = opt.model, epoch = epoch)
        early_stopping(valid_mae)        
        if early_stopping.early_stop:
            print("======= Early stopping =======")
            break

    os.system('echo " ==== TRAIN DICE mtc:{:.5f}" >> {}'.format(train_mae, res))
    print('Epo - Mtc')
    mtc_epo = dict(zip(saved_metrics, saved_epos))
    rank_mtc = sorted(mtc_epo.keys(), reverse=False)
    try:
        for i in range(10):
            print('{:03} {:.3f}'.format(mtc_epo[rank_mtc[i]], rank_mtc[i]))
            os.system('echo "epo:{:03} mtc:{:.3f}" >> {}'.format(mtc_epo[rank_mtc[i]], rank_mtc[i], res))
    except:
        pass
    

def train(train_loader, model, seg_crit, age_crit, optimizer, epoch):
    
    Loss = AverageMeter()
    AgeLoss = AverageMeter()
    SegLoss = AverageMeter()
    Dice = AverageMeter()
    MAE = AverageMeter()

    for i, (img, mask, _, age, gender, _) in enumerate(train_loader):

        input = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        age = torch.from_numpy(np.expand_dims(age,axis=1))
        age = age.type(torch.FloatTensor).cuda(non_blocking=True)
        if opt.use_gender:
            gender = torch.from_numpy(np.expand_dims(gender,axis=1))
            gender = gender.type(torch.FloatTensor).cuda(non_blocking=True)
        else:
            gender = None

        lam = index = None
        if opt.ismixup:
            alpha = opt.alpha
            lam = np.random.beta(alpha,alpha)
            index = torch.randperm(input.size(0)).cuda()
            input = lam * input + (1 - lam) * input[index]
            age = lam * age + (1 - lam) * age[index]
            gender = lam * gender + (1 - lam) * gender[index]
            mask_a, mask_b = mask, mask[index]
#
        model.train()
        optimizer.zero_grad()
        segout, ageout = model(input, gender)
        age_loss = age_crit(ageout, age)
        if opt.ismixup:
            seg_loss = lam * seg_crit(segout, mask_a.long()) + (1 - lam) * seg_crit(segout, mask_b.long())
        else:
            seg_loss = seg_crit(segout, mask.long())
        loss = opt.lam2 * age_loss + opt.lam1 * seg_loss

        mae = metric(ageout.detach(), age.detach().cpu())
        dice = dice_coeff(segout.detach(), mask.long(), num_classes = segout.size(1)).cpu().data.numpy()
        Loss.update(loss, img.size(0))
        AgeLoss.update(age_loss, img.size(0))
        SegLoss.update(seg_loss, img.size(0))
        Dice.update(dice, img.size(0))
        MAE.update(mae, img.size(0))

        if i % opt.print_freq == 0:
            print('Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                  'AgeLoss {AgeLoss.val:.3f} ({AgeLoss.avg:.3f})\t'
                  'SegLoss {SegLoss.val:.3f} ({SegLoss.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'
                  'Dice {Dice.val:.3f} ({Dice.avg:.3f})\t'.format
                  (epoch, opt.epochs, i, len(train_loader), AgeLoss=AgeLoss, SegLoss=SegLoss, loss=Loss, Dice=Dice, MAE=MAE))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()

    return Loss.avg, Dice.avg, MAE.avg

def validate(valid_loader, model, seg_crit, age_crit):

    Loss = AverageMeter()
    AgeLoss = AverageMeter()
    SegLoss = AverageMeter()
    Dice = AverageMeter()
    MAE = AverageMeter()

    # =========== switch to evaluate mode ===========#
    model.eval()

    with torch.no_grad():
        for _, (img, mask, _, age, gender, _) in enumerate(valid_loader):
            
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            age = torch.from_numpy(np.expand_dims(age,axis=1))
            age = age.type(torch.FloatTensor).cuda(non_blocking=True)
            if opt.use_gender:
                gender = torch.from_numpy(np.expand_dims(gender,axis=1))
                gender = gender.type(torch.FloatTensor).cuda(non_blocking=True)
            else:
                gender = None

            segout, ageout = model(img, gender)
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

        print('Valid: [steps {0}]\t'
            'AgeLoss {AgeLoss.val:.3f} ({AgeLoss.avg:.3f})\t'
            'SegLoss {SegLoss.val:.3f} ({SegLoss.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'
            'Dice {Dice.val:.3f} ({Dice.avg:.3f})\t'.format
            (len(valid_loader), AgeLoss=AgeLoss, SegLoss=SegLoss, loss=Loss, Dice=Dice, MAE=MAE))

        return Loss.avg, Dice.avg, MAE.avg

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def adjust_learning_rate(optimizer, epoch, opt):
	lr = opt.base_lr * ((1 - float(epoch) / opt.epochs) ** (opt.power))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def save_checkpoint(state, is_best, out_dir, model_name, epoch):
    checkpoint_path = out_dir+model_name + str(epoch) + '_checkpoint.pth.tar'
    best_model_path = out_dir+model_name + str(epoch) + '_best_model.pth.tar'
    if is_best and epoch > 10:
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")
    if epoch % 20 == 0:
        torch.save(state, checkpoint_path)

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(w, 'weight'):
            nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(w, 'bias') and w.bias is not None:
                nn.init.constant_(w.bias, 0)
    if classname.find('Linear') != -1:
        if hasattr(w, 'weight'):
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find('BatchNorm') != -1:
        if hasattr(w, 'weight') and w.weight is not None:
            nn.init.constant_(w.weight, 1)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)

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

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    num_gpu = torch.cuda.device_count()
    print("available gpus:", num_gpu)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = True

    res = os.path.join(opt.output_dir, 'result')
    os.makedirs(opt.output_dir, exist_ok = True)
    print('Training Starts...')
    main(res)
