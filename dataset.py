# %%

import os
import sys
import re
import six
import math
import numpy as np
import lmdb
import random
import cv2
from PIL import Image
import torch
from straug.geometry import Perspective, Rotate
from straug.warp import Curve, Distort
from natsort import natsorted
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        self.next_input, self.next_target = next(self.loader)
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        # if target is not None:
        #     target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, dataAug=True)
        self.datasetList = []
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0

        totalSize = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(
                root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset *
                                 float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset,
                             total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            totalSize += len(_dataset)
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'

            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            
            self.datasetList.append(_dataset)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(data_prefetcher(_data_loader))

        opt.batch_size = Total_batch_size
        opt.batchBalanceDatasetLen = totalSize//Total_batch_size
        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size} total sample = {totalSize},len = {opt.batchBalanceDatasetLen} \n'
        Total_batch_size_log += f'{dashed_line}'

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = data_prefetcher(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts
# %%


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            self.filtered_index_list=[]
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)
            
            # if self.opt.data_filtering_off:
            #     # for fast check or benchmark evaluation with no filtering
            #     self.filtered_index_list = [
            #         index + 1 for index in range(self.nSamples)]
            # else:
            #     """ Filtering part
            #     If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
            #     use --data_filtering_off and only evaluate on alphabets and digits.
            #     see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

            #     And if you want to evaluate them with the model trained with --sensitive option,
            #     use --sensitive and --data_filtering_off,
            #     see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
            #     """
            #     self.filtered_index_list = []
            #     for index in range(self.nSamples):
            #         index += 1  # lmdb starts with 1
            #         label_key = 'label-%09d'.encode() % index
            #         label = txn.get(label_key).decode('utf-8')

            #         if len(label) > self.opt.batch_max_length:
            #             # print(f'The length of the label is longer than max_length: length
            #             # {len(label)}, {label} in dataset {self.root}')
            #             continue

            #         # By default, images containing characters which are not in opt.character are filtered.
            #         # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
            #         out_of_char = f'[^{self.opt.character}]'
            #         if re.search(out_of_char, label.lower()):
            #             continue

            #         self.filtered_index_list.append(index)

                # self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def getBuf(self, txn, index):
        if index == 0:
            index = 1
            
        label_key = 'label-%09d'.encode() % index

        label = txn.get(label_key).decode('utf-8')

            
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        return buf, label

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]
        with self.env.begin(write=False) as txn:
            buf, label = self.getBuf(txn, index)
            try:
                imgForCheck = Image.open(buf)
                if min(imgForCheck.size) <= 10:
                    while(True):
                        index = random.randint(0, len(self) - 1)
                        index = self.filtered_index_list[index]
                        buf, label = self.getBuf(txn, index)
                        imgForCheck = Image.open(buf)
                        if min(imgForCheck.size) > 10:
                            break

                if self.opt.rgb:
                    img = imgForCheck.convert('RGB')  # for color image
                else:
                    img = imgForCheck.convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert(
                    'RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w -
                                    1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, dataAug=False,p=0.3):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.dataAug = dataAug
        self.p = p
        self.toTensor = transforms.ToTensor()
        if self.dataAug:
            

            geoTrans = transforms.RandomApply([transforms.RandomChoice(
                [Curve(), Distort(), Rotate(), Perspective()])], p=self.p)
            jitter = transforms.RandomApply([transforms.ColorJitter(
                brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1)], p=self.p)
            self.aug_trans = transforms.Compose([geoTrans, jitter])

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.imgW < self.imgH:
                if fix_h:
                    trg_h = self.imgH
                else:
                    trg_h = int(ratio * self.imgW)
                trg_w = self.imgW
            else:
                trg_h, trg_w = self.imgH, int(self.imgH / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.imgH - trg_h) / 2, (self.imgW - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.dataAug:
            if random.random() < 0.5:
                base, maxh, maxw = self.imgH, self.imgH, self.imgW
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h/w)
            else:
                # keep aspect ratio
                return _resize_ratio(img, img.shape[0] / img.shape[1])
        else:
            # keep aspect ratio
            return _resize_ratio(img, img.shape[0] / img.shape[1])

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper

            resized_images = []
            for image in images:
                image = np.array(image)
                resized_image = self.resize_multiscales(
                    image, cv2.cv2.BORDER_REPLICATE)
                resized_image = Image.fromarray(resized_image)
                if self.dataAug:
                    minSize = min(image.shape[:-1])
                    if minSize > 14:
                        resized_image = self.aug_trans(resized_image)

                resized_images.append(self.toTensor(resized_image))

            image_tensors = torch.cat([t.to(dtype=torch.half).unsqueeze(0)
                                      for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0)
                                      for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
