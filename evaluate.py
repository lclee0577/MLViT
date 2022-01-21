# %%

from modules.dualTrans import CLstr
import torch
import argparse
from dataset import AlignCollate, hierarchical_dataset

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
torch.manual_seed(opt.manual_Seed)
torch.cuda.manual_seed(opt.manual_Seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fp16 = True
# %%

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
# %%


PATH = './saved_models/small-rev-large-ckpt_best-reEval.pt'
checkpoint = torch.load(PATH)


model.load_state_dict(checkpoint['net'],strict=False)
bestAccu = checkpoint["bestAccu"]


print(f"{bestAccu=}")
del checkpoint

# %%

eval_data_list = ['IC13_857','SVT','IIIT5k_3000', 
                      'IC15_1811', 'SVTP', 'CUTE80']
model.eval()
result = []
for datasetName in eval_data_list:

    opt.eval_data = f'data_lmdb_release/evaluation/{datasetName}'
    AlignCollate_evaluation = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)
    correctNum = 0
    lenDataset = len(evaluation_loader.dataset)

    for i,(image_tensors,labels)  in enumerate(evaluation_loader):
        image = image_tensors.to(device)
        prd_txt = model.predictWithTri(image)
            
        for each_pred, each_label in zip(prd_txt, labels):
            if each_label == each_pred:
                correctNum += 1
            else:
                # print(f'label-pred {each_pred}  {each_label} ')
                pass
    accu = correctNum/lenDataset
    result.append(accu)

print("|          |IC13|SVT |IT5K|IC15|SVTP|CUTE|")   
print("|benchmark |93.6|87.5|87.9|77.6|79.2|74.0|")
print("|ours      |",end="")
for each in result:
    print(f"{each*100:2.1f}",end="|")
# %%
    