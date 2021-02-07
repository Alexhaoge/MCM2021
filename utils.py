# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
import os
from hashlib import md5
import time

classes = ['hornet']

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id):
    in_file = open('label/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('train_label/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
#         difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()

        
def get_filelist(dir, Filelist=[]):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    return Filelist


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    """

    def __init__(self, verbose, patience, no_stop):
        self.verbose = verbose
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.isToStop = False
        self.enable_stop = not no_stop

    def __call__(self, val_loss, model, optimizer, epoch, filename):
        is_best = bool(val_loss < self.best_loss)
        if is_best:
            self.best_loss = val_loss
            self.__save_checkpoint(self.best_loss, model,
                                   optimizer, epoch, filename)
            if self.verbose:
                print(filename)
            self.counter = 0
        elif self.enable_stop:
            self.counter += 1
            if self.verbose:
                print(f'=> Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.isToStop = True

    def __save_checkpoint(self, loss, model, optimizer, epoch, filename):
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'loss': loss}
        torch.save(state, filename)
        if self.verbose:
            print('=> Saving a new best')


def plot(data, columns_name, x_label, y_label, title, inline=False):
    df = pd.DataFrame(data).T
    df.columns = columns_name
    df.index += 1
    plot = df.plot(linewidth=2, figsize=(15, 8), color=['darkgreen', 'orange'], grid=True)
    train = columns_name[0]
    val = columns_name[1]
    # find position of lowest validation loss
    idx_min_loss = df[val].idxmin()
    plot.axvline(idx_min_loss, linestyle='--', color='r', label='Best epoch')
    plot.legend()
    plot.set_xlim(0, len(df.index)+1)
    plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plot.set_title(title, fontsize=16)
    if not inline:
        m = md5()
        m.update(str(time.time()))
        filename = 'output/plot/' + m.hexdigest() + '.png'
        plot.figure.savefig(filename, bbox_inches='tight')
        return filename
