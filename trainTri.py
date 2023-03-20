# %%
import time
import random
from modules.dualTrans import CLstr
from modules.utils import  get_cosine_schedule_with_warmup
from tqdm.notebook import tqdm
from lightseq.training import LSCrossEntropyLayer, LSAdam
import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import AlignCollate, Batch_Balanced_Dataset, hierarchical_dataset

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='lclee',
                    help='Where to store logs and models')
parser.add_argument('--train_data', default='data_lmdb_release/training',
                    help='path to training dataset')
parser.add_argument('--valid_data', default='data_lmdb_release/validation',
                    help='path to validation dataset')
parser.add_argument('--manual_Seed', type=int, default=1111,
                    help='for random seed setting')
parser.add_argument('--batch_size', type=int,
                    default=256, help='input batch size')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
###
""" Data processing """
parser.add_argument('--select_data', type=str, default='MJ-ST',
                    help='select training data (default is MJ-ST-SY, which means MJ and ST used as training data)')
parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                    help='assign ratio for each selected data in the batch')
parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                    help='total data usage ratio, this ratio is multiplied to total number of data.')
parser.add_argument('--batch_max_length', type=int,
                    default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=64,
                    help='the height of the input image')
parser.add_argument('--imgW', type=int, default=192,
                    help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str,
                    default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true',
                    help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true',
                    help='whether to keep ratio then pad for image resize')
parser.add_argument('--data_filtering_off', action='store_true',
                    help='for data_filtering_off mode')
# %%
# 
# usrArgs = '''--total_data_usage_ratio 1 --PAD --rgb --select_data MJ-ST-SY --batch_ratio 0.25-0.25-0.5'''.split(" ")
usrArgs = '''--total_data_usage_ratio 1 --PAD --rgb'''.split(" ")

opt = parser.parse_args(args=usrArgs)


random.seed(opt.manual_Seed)
torch.manual_seed(opt.manual_Seed)
torch.cuda.manual_seed(opt.manual_Seed)

torch.backends.cudnn.benchmark = True  # It fasten training.
torch.backends.cudnn.deterministic = True
    
    
opt.select_data = opt.select_data.split('-')
opt.batch_ratio = opt.batch_ratio.split('-')

# %%
os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)
log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
AlignCollate_valid = AlignCollate(
    imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
# valid_dataset, valid_dataset_log = hierarchical_dataset(
#     root=opt.valid_data, opt=opt)
valid_dataset, valid_dataset_log = hierarchical_dataset(
    root='./data_lmdb_release/val2', opt=opt)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=opt.batch_size,
    # 'True' to check training progress with validation function.
    shuffle=False,
    num_workers=2,
    collate_fn=AlignCollate_valid, pin_memory=True)
print('-' * 80)
log.write('-' * 80 + '\n')
log.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
fp16 = True
def create_criterion():
    
    ce_config = LSCrossEntropyLayer.get_config(
        max_batch_tokens=49*256,
        padding_idx=0,
        epsilon=0.1,
        fp16=fp16,
        local_rank=0,
    )
    loss_fn = LSCrossEntropyLayer(ce_config)
    if fp16:
        loss_fn.to(dtype=torch.half, device=device)
    else:
        loss_fn.to(device=device)
    return loss_fn

modelType = "bert-base"

def create_model():
    transformer_config = CLstr.get_config(
        model=modelType,
        max_batch_tokens=49*256,
        max_seq_len=27,
        max_imgPad_len = 62,
        vocSmallPath='vocSingle.txt',
        vocLargePath='vocLarge.txt',
        n_langs=3,
        padding_idx=0,
        num_encoder_layer=6,
        num_decoder_layer=6,
        patch_size=16,
        pre_layer_norm=False,
        fp16=fp16,
        local_rank=0,
    )
    model = CLstr(transformer_config, device=device)
    if fp16:
        model.to(dtype=torch.half, device=device)
    else:
        model.to( device=device)
    return model

model = create_model()
loss_fn =create_criterion()
optimizer = LSAdam(model.parameters(), lr=3e-5)#1e-4

train_dataset = Batch_Balanced_Dataset(opt)
trainStep = opt.batchBalanceDatasetLen

# %%
epoch = 0
valAccuS,valAccuL,valAccu = 0,0,0

validStep = 7500
bestAccu = 0.87
lanList = ['small','rev','large']
runDate = "0125"
saveName = '-'.join(lanList)


n_epoch=20
# train_dataset = Batch_Balanced_Dataset(opt)
# trainStep = opt.batchBalanceDatasetLen
optimizer = LSAdam(model.parameters(), lr=3e-5)#1.2 min ratio0.45
exp_lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, trainStep*n_epoch)
lr = exp_lr_scheduler.get_last_lr()

class modelVar(object):
    def __init__(self):
        self.trg_tokens = None
        self.target = None
        self.decoderOut = None
        self.accu = -1
        self.loss = -1
allVal = {
"small": modelVar(),
"rev" : modelVar(),
"large" : modelVar(),
}
logFlag = 0
accuLOut,accuROut,accuSOut=0,0,0
each = 'large'
torch.set_num_threads(2) 
image, label = train_dataset.get_batch()
# %%

for epoch in range(n_epoch):
    pbar = tqdm(total=trainStep, desc= f"Train{epoch}/{n_epoch}", unit=" step")
    writer = SummaryWriter()

    model.train()
    for step in range(trainStep):
        logFlag = (step + 1) % 100 == 0
        
        allVal[each].trg_tokens, allVal[each].target = model.gpt2VocLarge.convertToken(label,size=each)
        allVal[each].accu,allVal[each].decoderOut  = model(image,allVal[each].trg_tokens,logFlag)

        
        loss = loss_fn(allVal[each].decoderOut, allVal[each].target)[0] 
        allVal[each].loss = loss.item()
           
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()

        if logFlag :

            each = random.choice(lanList)
            accuSOut = allVal['small'].accu if allVal['small'].accu > 0 else accuSOut 
            accuLOut = allVal['large'].accu if allVal['large'].accu > 0 else accuLOut 
            accuROut = allVal['rev'].accu if allVal['rev'].accu > 0 else accuROut 
            
            lossOut =loss.item()  # infoDict['loss']
            for eachOut in lanList:
                writer.add_scalar(f"Loss/{eachOut}", allVal[eachOut].loss, step)
            
            lr = exp_lr_scheduler.get_last_lr()
            
            if accuSOut != 0 : 
                writer.add_scalar(f"Accu/TrainS", accuSOut, step) 
            if accuLOut != 0 : 
                writer.add_scalar(f"Accu/TrainL", accuLOut, step)
            # writer.add_scalar(f"Loss/w_LR", lr[0], step)
            # pbar.set_postfix(lr=f'{lr[0]:.2e}', loss=f"{lossOut:.2f}", accu=f"{accu*100:.2f}",
            #                 valAccuL=f'{valAccuL*100:.2f}',valAccuS=f'{valAccuS*100:.2f}', step=step)
            pbar.set_postfix(lr=f'{lr[0]:.2e}', loss=f"{lossOut:.2f}", accuS=f"{accuSOut*100:.2f}",accuR=f"{accuROut*100:.2f}",accuL=f"{accuLOut*100:.2f}",
                            valAccuL=f'{valAccuL*100:.2f}',valAccuS=f'{valAccuS*100:.2f}', step=step)

        
        
        if (step + 1) % validStep == 0:
            validStep = 750 if bestAccu> 0.89 else 7500

            model.eval()
            correctNumS = 0
            correctNumL = 0
            correctNumR = 0
            correctNumSel = 0
            thre=0.4
            ratio = 1.2
            datasetLen = len(valid_loader.dataset)
            for i, (image_tensors, label) in enumerate(valid_loader):
                with torch.no_grad():

                    image = image_tensors.to(device)          
                    smallLen,largeLen = model.calcLabelLen(label)
                    
                    encoder_out, encoder_padding_mask = model.vit(image)
                    prd_small_txt,confSmall = model.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,smallLen,size='small')
                    prd_large_txt,confLarge = model.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,largeLen,size='large')
                    prd_rev_txt,confRev = model.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,smallLen,size='rev')

                                
                    for (txtS,confS,txtL,confL,txtR,confR,each_label) in zip(prd_small_txt,confSmall,prd_large_txt,confLarge,prd_rev_txt,confRev, label):
                            
                            selTxt = ""
                            if (txtS == txtR) and (confL>thre):
                                if confS/confL>ratio:
                                    selTxt = txtS
                                else:
                                    selTxt = txtL
                            else:
                                
                                confList = [confS,confL,confR ]
                                txtList = [txtS,txtL,txtR]    
                                selTxt = txtList[confList.index(max(confList))]
                            
                            if each_label == txtS:
                                correctNumS += 1
                            if each_label == txtL:
                                correctNumL += 1
                            if each_label == txtR:
                                correctNumR += 1
                            if each_label ==selTxt:
                                correctNumSel += 1

            valAccuS = correctNumS / datasetLen
            valAccuL = correctNumL / datasetLen
            valAccuR = correctNumR / datasetLen
            valAccuSel = correctNumSel / datasetLen
                
            writer.add_scalar(f"Accu/ValS", valAccuS, step)
            writer.add_scalar(f"Accu/ValL", valAccuL, step)
            writer.add_scalar(f"Accu/ValR", valAccuR, step)
            writer.add_scalar(f"Accu/ValSel", valAccuSel, step)
            valAccu =max(valAccuL,valAccuS,valAccuR,valAccuSel)
            if valAccu > bestAccu:
                bestAccu = valAccu
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    'lr_schedule': exp_lr_scheduler.state_dict(),
                    "bestAccu": bestAccu
                }
                saveDir = f"./saved_models/{runDate}"
                os.makedirs(saveDir, exist_ok=True)
                torch.save(checkpoint, f'{saveDir}/{saveName}-ckpt_best.pt')
                print(f"save model {valAccu=:4f} lr={lr[0]:.2e} {time.ctime()}")
                
            model.train()
        image, label = train_dataset.get_batch()# 
        pbar.update()

    writer.flush()
    writer.close()



# %%