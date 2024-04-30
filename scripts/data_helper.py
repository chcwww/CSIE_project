import pickle
import os
import re
import logging
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np

from scripts.buffer import Buffer
from utils.util import CAPACITY, BLOCK_SIZE, BLOCK_MIN

class SimpleListDataset(Dataset):
    def __init__(self, source):
        ## 外面換成Path了所以原本的這個檢查不出來，會看成list
        if isinstance(source, Path):
            with open(source, 'rb') as fin:
                logging.info('Loading dataset...')
                self.dataset = pickle.load(fin)
        elif isinstance(source, list):
            self.dataset = source
        if not isinstance(self.dataset, list):
            raise ValueError('The source of SimpleListDataset is not a list.')
    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

class BlkPosInterface:
    def __init__(self, dataset):
        # assert isinstance(dataset, SimpleListDataset)
        self.d = {} # KEY : blkPos, VALUE : block 
        self.dataset = dataset
        for bufs in dataset:
            for buf in bufs:
                for blk in buf:
                    assert blk.pos not in self.d
                    self.d[blk.pos] = blk
    def set_property(self, pos, key, value=None):
        blk = self.d[pos]
        if value is not None:
            setattr(blk, key, value)
        elif hasattr(blk, key):
            delattr(blk, key)
    def apply_changes_from_file(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                tmp = [
                    int(s) if s.isdigit() or s[0] == '-' and s[1:].isdigit() else s 
                    for s in line.split()
                ]
                self.set_property(*tmp)
    def apply_changes_from_dir(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('changes_'):
                self.apply_changes_from_file(filename)
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def collect_estimations_from_dir(self, tmp_dir): # 從estimation檔案把數字更新到blk身上
        ret = []
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('estimations_'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        self.d[pos].estimation = estimation
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def build_random_buffer(self, num_samples): 
        n0, n1 = [int(s) for s in num_samples.split(',')][:2]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        # max_blk_num = CAPACITY // (BLOCK_MIN + 1)
        logging.info('building buffers for introspection...')
        # 等於本來的一個buf會生出兩個人給ret
        for qbuf, dbuf in tqdm(self.dataset):
            # 1. continous 
            lb = max_blk_num - len(qbuf)
            st = random.randint(0, max(0, len(dbuf) - lb * n0))
            for i in range(n0):
                buf = Buffer()
                # 隨機找了一段continuos的blocks裝上去
                buf.blocks = qbuf.blocks + dbuf.blocks[st + i * lb:st + (i+1) * lb]
                ret.append(buf)
            # 2. pos + neg
            # p就是relv高的 n就是relv低的
            # pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.choose >= 1, need_residue=True)            
            for i in range(n1):
                # 盡量放pos 真的放完了再放nbuf
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf)))
                buf = Buffer()
                buf.blocks = qbuf.blocks + selected_pblks + selected_nblks
                ret.append(buf.sort_())
        return SimpleListDataset(ret)

    def build_promising_buffer(self, num_samples):
        n2, n3 = [int(x) for x in num_samples.split(',')][2:]
        ret = []
        # 被這個搞到了 (每次都8個)
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        # max_blk_num = CAPACITY // (BLOCK_MIN + 1) # 這東西有鬼 下面可能還要再調調
        # max_blk_num = 100
        # print(f"max_blk_num is {max_blk_num}")
        logging.info('building buffers for reasoning...')
        for qbuf, dbuf in tqdm(self.dataset):
            #1. retrieve top n2*(max-len(pos)) estimations into buf 2. cut
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            # pbuf是要的 (relevance >= 1), nbuf是剩下的
            if len(pbuf) >= max_blk_num - len(qbuf):
                pbuf = pbuf.random_sample(max_blk_num - len(qbuf) - 1) 
            lb = max_blk_num - len(qbuf) - len(pbuf)
            estimations = torch.tensor([blk.estimation for blk in nbuf], dtype=torch.long)
            keeped_indices = estimations.argsort(descending=True)[:n2 * lb]
            selected_nblks = [blk for i, blk in enumerate(nbuf) if i in keeped_indices]
            while 0 < len(selected_nblks) < n2 * lb:
                selected_nblks = selected_nblks * (n2 * lb // len(selected_nblks) + 1)
            for i in range(n2):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + selected_nblks[i * lb: (i+1) * lb]
                ret.append(buf.sort_())
            for i in range(n3):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + random.sample(nbuf.blocks, min(len(nbuf), lb))
                ret.append(buf.sort_())
        return SimpleListDataset(ret)
    
    def build_strong_buffer(self) :
        ret = []
    
        logging.info('building strong label buffers for reasoning...')
        for qbuf, dbuf in tqdm(self.dataset):
        # for qbuf, dbuf in tqdm(sw_dataset):
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.choose == 1, need_residue=True)
            local_len = 1 # CLS先佔了一個
            buf = Buffer()
            buf.blocks = qbuf.blocks
            blk_pos = []
            for b in pbuf.blocks :
                if local_len + len(b) < 512 :
                    buf.blocks = buf.blocks + [b]
                    blk_pos.append(b.pos)
                    local_len += len(b)
            for b in nbuf.blocks :
                # 已選的前後句
                if local_len + len(b) < 512 and ((b.pos + 1) in blk_pos or (b.pos - 1) in blk_pos) :
                    buf.blocks += [b]
                    local_len += len(b)
            ret.append(buf.sort_())
            
        return SimpleListDataset(ret)

def find_lastest_checkpoint(checkpoints_dir, epoch=False):
    # checkpoints_dir = 'C:\\vs_code_python\\log_dir\\introspector\\version_0\\checkpoints'
    lastest = (-1, '')
    if os.path.exists(checkpoints_dir):
        for shortname in os.listdir(checkpoints_dir):
            # m = re.match(r'_ckpt_epoch_(\d+).+', shortname)
            # if m is not None and int(m.group(1)) > lastest[0]:
            #     lastest = (shortname, shortname)
            m = re.match(r'epoch=(.*)_(.*).ckpt', shortname)
            k = re.match(r'epoch=(.*).ckpt', shortname)
            if m is None and k is not None and int(k.group(1)) > lastest[0]:
                lastest = (int(k.group(1)), shortname)            
    return os.path.join(checkpoints_dir, lastest[-1]) if not epoch else lastest[0]