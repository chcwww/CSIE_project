## other
import argparse
import logging # 我把它當成print的替代
import os
import random
import sys
from pathlib import Path # 會幫忙處理路徑格式
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from copy import deepcopy
import numpy as np
from operator import itemgetter

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset

print(torch.cuda.is_available())
print(torch.cuda.device_count())

## project
from utils.util import SAVE_DIR, TMP_DIR, LOG_DIR, USE_PATH, CAPACITY, MODEL_NAME, BIG_MODEL_NAME, AttributeDict
from utils.memreplay import mem_replay, _score_blocks
from scripts.data_helper import SimpleListDataset, BlkPosInterface, find_lastest_checkpoint
from scripts.buffer import buffer_collate
from models.model import Introspector, ALLonBert_v2, ALLonBert_v3, ALLonBert_v4
from models.hierbert import HierarchicalBert
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold

print(f"Dir : {os.getcwd()}")

dir_checkpoint = Path(SAVE_DIR)


class MySubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]
    

def _write_estimation(_file, buf, relevance_blk):
    for i, blk in enumerate(buf):
        _file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

def sep_train_weak(
        model = Introspector(MODEL_NAME), # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        m_name = 'Judge', # model_name
        device = 'cpu',
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        toy: bool = False,
):
    # 1 Create dataset
    os.makedirs(TMP_DIR, exist_ok=True)
    if toy:
        sw_dataset = SimpleListDataset(USE_PATH.weak.toy)
    else:
        sw_dataset = SimpleListDataset(USE_PATH.weak.train)

    # 2.b Create interface
    n_val = int((len(sw_dataset))*val_percent)
    n_train = len(sw_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(sw_dataset, [n_train, n_val])
    interface = BlkPosInterface(train_set)
    interface_val = BlkPosInterface(val_set)

    ## Original module part
    # 3. Create data loaders (args)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    # 4. Set up the optimizer, the loss and the learning rate scheduler
    DO_VALID = True
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    total_steps = epochs*int(len(train_set)/batch_size)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    global_step = 0
    # 5. Begin training
    # epoch = 1
    for epoch in range(1, epochs + 1):
        # Update interface
        # train_set = interface.build_random_buffer('1,1,1,1')
        train_set = interface.build_random_buffer_version2()

        train_loader = DataLoader(train_set, shuffle=True, 
                                    collate_fn = buffer_collate, # 讓dataloader可以迭代buffer類
                                    **loader_args
                                    )

        model.train()
        _file = open(Path(os.path.join(TMP_DIR, 'estimations_{}.txt'.format(device))), 'w')
        batch_steps = 0
        epoch_loss = 0
        for bufs in (pbar:=tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit=f'Paragraph({batch_size})')) : 
            batch_steps += 1
            # Make inputs for reasoner
            inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long, device=device)
            for i, buf in enumerate(bufs):
                buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 和reasoner一樣設定input
            # Train the introspector after labeling
            for i, buf in enumerate(bufs):
                buf.export_relevance(device=device, out=inputs[3, i]) # 用來設定judge的label(由relevance)
            # Label the relevance by the current reasoner  
            # breakpoint()
            loss, logits = model(*inputs[:3], labels=inputs[3])
            for i, buf in enumerate(bufs):
                _write_estimation(_file, buf, _score_blocks(buf, torch.sigmoid(logits[i]))) # 把這輪跑完的relevance更新到檔案上

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
            optimizer.step()
            # scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(**{
                'loss (batch)': loss.item()
                })
            
        logging.info(f"""
            Model {m_name} (epoch) : 
                loss  ->  {epoch_loss / batch_steps:.4f}
                    """)
            
        interface.collect_estimations_from_dir(TMP_DIR)
        
        if save_checkpoint:
            dir_ch_this = Path(dir_checkpoint / 'checkpoint' / m_name)
            Path(dir_ch_this).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_ch_this / 'checkpoint_epoch{}_{}.pth'.format(epoch, m_name)))
            logging.info(f'Model {m_name} : Checkpoint {epoch} saved!')


def sep_train(
        model = ALLonBert_v2(MODEL_NAME), # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        m_name = 'Reasoner', # model_name
        device = 'cpu',
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        toy: bool = False,
):
    # 1 Create dataset
    os.makedirs(TMP_DIR, exist_ok=True)
    if toy:
        sw_dataset = SimpleListDataset(USE_PATH.strong.toy)
    else:
        sw_dataset = SimpleListDataset(USE_PATH.strong.train)

    acc_k = []
    prec_k = []
    rec_k = []
    f1_k = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(sw_dataset), start=1):
        n_train = len(train_idx)
        n_val = len(valid_idx)
        # 2.b Create interface
        # n_val = int((len(sw_dataset))*val_percent)
        # n_train = len(sw_dataset) - n_val
        # train_set, val_set = torch.utils.data.random_split(sw_dataset, [n_train, n_val])
        train_set = MySubset(sw_dataset, train_idx)
        val_set = MySubset(sw_dataset, valid_idx)
        
        interface = BlkPosInterface(train_set)
        interface_val = BlkPosInterface(val_set)

        ## Original module part
        # 3. Create data loaders (args)
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

        # train_loader = DataLoader(train_set, shuffle=True, **loader_args)

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Fold:            {fold}
        ''')

        # 4. Set up the optimizer, the loss and the learning rate scheduler
        DO_VALID = True
        optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

        global_step = 0
        # 5. Begin training
        # epoch = 1
        for epoch in range(1, epochs + 1):
            # Update interface
            train_set = interface.build_strong_buffer()
                
            train_loader = DataLoader(train_set, shuffle=True, 
                                        collate_fn = buffer_collate, # 讓dataloader可以迭代buffer類
                                        **loader_args
                                        )

            model.train()
            epoch_loss = 0
            batch_steps = 0
            epoch_sum = 0
            epoch_len = 0
            epoch_tp = 0
            epoch_fn = 0
            epoch_fp = 0
            for bufs in (pbar:=tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit=f'Paragraph({batch_size})')) : 
                batch_steps += 1
                # Make inputs for reasoner
                inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long, device=device)  # [ 3, BATCH, 512 ]
                # 因為tokenizer.convert_ids_to_tokens(0) = '[PAD]' 所以等於是天生PAD然後再填Buffer每個Block的ids進去 (buf.export那裡的操作)
                blk_pos = []
                for i, buf in enumerate(bufs):
                    # export -> ids, att_masks, types, blk_position
                    _, _, _, b_p = buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 其實搞不太懂怎麼用export把buf的資訊給inputs的 python還能搞指標的?
                    blk_pos.append(b_p) # b_p : [ 1, NUM_OF_BLOCK ] 選到的(從interface那邊)每個block在相應buffer中的位置(之後損失函數那邊才可跟label比)
                # Extract the labels for reasoner, e.g. start and end position for QA reasoner
                # crucials (list) : BATCH * [1, NUM_OF_BLOCK_blk_type==0]
                labels, crucials = model.export_labels(bufs, device) # TODO A
                result = model(*inputs, labels=labels, pos = blk_pos, device = device, debug_buf=bufs)
                
                losses = result[0] if isinstance(result, tuple) else result
                # loss = sum(losses)/len(losses) # Mean or Sum ?
                loss = sum(losses)
                
                logits = result[2]
                local_labels = result[1]
                softmax_preds = [F.softmax(logit, dim = 1) for logit in logits]
                preds = [torch.max(s_pred, dim=1).indices for s_pred in softmax_preds]
                # sum_correct = sum(sum(ans) for ans in map(lambda x, y : x==y, preds, local_labels))
                # len_label = sum(len(lab) for lab in local_labels)
                
                preds_list = np.array(sum([pred.tolist() for pred in preds], []))
                labels_list = np.array(sum([lab.tolist() for lab in local_labels], []))
                # sum_correct = sum(map(lambda x, y : x==y, preds_list, labels_list))
                sum_correct = (preds_list == labels_list).sum()
                len_label = len(labels_list)
                
                acc_batch = (sum_correct / len_label).item()
                epoch_sum += sum_correct
                epoch_len += len_label
                
                batch_tp = np.logical_and(labels_list == 1, preds_list == 1).sum(axis=0)
                batch_fn = np.logical_and(labels_list == 1, preds_list == 0).sum(axis=0)
                batch_fp = np.logical_and(labels_list == 0, preds_list == 1).sum(axis=0)            
                epoch_tp += batch_tp
                epoch_fn += batch_fn
                epoch_fp += batch_fp
                # logging.info(f'Model {m_name} : batch acc -> {acc_batch:.3f}')

                # _intervention(_file, bufs, labels, crucials, result, model)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
                optimizer.step()
                # scheduler.step(acc_batch)

                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{
                    'acc (batch)': acc_batch,
                    'loss (batch)': loss.item()
                    })
                # 'f1 (batch)': 2*batch_tp / (2*batch_tp + batch_fn + batch_fp),
                # 'recall (batch)': batch_tp / (batch_tp + batch_fn),
                # 'precision (batch)': batch_tp / (batch_tp + batch_fp),
                
            logging.info(f"""
                Model {m_name} (epoch) Fold {fold} : 
                    loss      ->  {epoch_loss / batch_steps:.4f} 
                    accuracy  ->  {(epoch_sum/epoch_len).item():.4f}
                    precision ->  {epoch_tp / (epoch_tp + epoch_fp):.4f}
                    recall    ->  {epoch_tp / (epoch_tp + epoch_fn):.4f}
                    f1-score  ->  {2*epoch_tp / (2*epoch_tp + epoch_fn + epoch_fp):.4f}
                        """)
            
            if DO_VALID and not toy :
                val_set = interface_val.build_strong_buffer()
                val_loader = DataLoader(val_set, collate_fn=buffer_collate, **loader_args)
                model.eval()
                vepoch_loss = 0
                vbatch_steps = 0
                vepoch_sum = 0
                vepoch_len = 0
                vepoch_tp = 0
                vepoch_fn = 0
                vepoch_fp = 0
                with torch.no_grad() :
                    for bufs in (pbar:=tqdm(val_loader, desc='Validation', unit=f'Paragraph({batch_size})')) : 
                        vbatch_steps += 1
                        # Make inputs for reasoner
                        inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long, device=device)  # [ 3, BATCH, 512 ]
                        # 因為tokenizer.convert_ids_to_tokens(0) = '[PAD]' 所以等於是天生PAD然後再填Buffer每個Block的ids進去 (buf.export那裡的操作)
                        blk_pos = []
                        for i, buf in enumerate(bufs):
                            # export -> ids, att_masks, types, blk_position
                            _, _, _, b_p = buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 其實搞不太懂怎麼用export把buf的資訊給inputs的 python還能搞指標的?
                            blk_pos.append(b_p) # b_p : [ 1, NUM_OF_BLOCK ] 選到的(從interface那邊)每個block在相應buffer中的位置(之後損失函數那邊才可跟label比)
                        # Extract the labels for reasoner, e.g. start and end position for QA reasoner
                        # crucials (list) : BATCH * [1, NUM_OF_BLOCK_blk_type==0]
                        labels, crucials = model.export_labels(bufs, device) # TODO A
                        result = model(*inputs, labels=labels, pos = blk_pos, device = device)
                        
                        losses = result[0] if isinstance(result, tuple) else result
                        loss = sum(losses) # Mean or Sum ?
                        
                        logits = result[2]
                        local_labels = result[1]
                        softmax_preds = [F.softmax(logit, dim = 1) for logit in logits]
                        preds = [torch.max(s_pred, dim=1).indices for s_pred in softmax_preds]
                        # sum_correct = sum(sum(ans) for ans in map(lambda x, y : x==y, preds, local_labels))
                        # len_label = sum(len(lab) for lab in local_labels)
                        
                        preds_list = np.array(sum([pred.tolist() for pred in preds], []))
                        labels_list = np.array(sum([lab.tolist() for lab in local_labels], []))
                        # sum_correct = sum(map(lambda x, y : x==y, preds_list, labels_list))
                        sum_correct = (preds_list == labels_list).sum()
                        len_label = len(labels_list)
                        
                        acc_batch = (sum_correct / len_label).item()
                        vepoch_sum += sum_correct
                        vepoch_len += len_label
                        
                        vbatch_tp = np.logical_and(labels_list == 1, preds_list == 1).sum(axis=0)
                        vbatch_fn = np.logical_and(labels_list == 1, preds_list == 0).sum(axis=0)
                        vbatch_fp = np.logical_and(labels_list == 0, preds_list == 1).sum(axis=0)            
                        vepoch_tp += vbatch_tp
                        vepoch_fn += vbatch_fn
                        vepoch_fp += vbatch_fp
                        
                        vepoch_loss += loss.item()
                        global_step += 1
                logging.info(f"""
                Model {m_name} (validation) Fold {fold} : 
                    loss      ->  {vepoch_loss / vbatch_steps:.4f} 
                    accuracy  ->  {(vepoch_sum/vepoch_len).item():.4f}
                    precision ->  {vepoch_tp / (vepoch_tp + vepoch_fp):.4f}
                    recall    ->  {vepoch_tp / (vepoch_tp + vepoch_fn):.4f}
                    f1-score  ->  {2*vepoch_tp / (2*vepoch_tp + vepoch_fn + vepoch_fp):.4f}
                        """)
                acc_k.append((vepoch_sum/vepoch_len).item())
                prec_k.append(vepoch_tp / (vepoch_tp + vepoch_fp))
                rec_k.append(vepoch_tp / (vepoch_tp + vepoch_fn))
                f1_k.append(2*vepoch_tp / (2*vepoch_tp + vepoch_fn + vepoch_fp))
    logging.info(f"""
    Model {m_name} (validation) Cross Validation (3 Fold) : 
        accuracy  ->  {sum(acc_k)/len(acc_k):.4f}
        precision ->  {sum(prec_k)/len(prec_k):.4f}
        recall    ->  {sum(rec_k)/len(rec_k):.4f}
        f1-score  ->  {sum(f1_k)/len(f1_k):.4f}
            """)        
        # if save_checkpoint:
        #     dir_ch_this = Path(dir_checkpoint / 'checkpoint' / m_name)
        #     Path(dir_ch_this).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     torch.save(state_dict, str(dir_ch_this / 'checkpoint_epoch{}_{}.pth'.format(epoch, m_name)))
        #     logging.info(f'Model {m_name} : Checkpoint {epoch} saved!')





def valid_train(
        models = [ALLonBert_v2(MODEL_NAME), None], # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        m_name = 'Reasoner', # model_name
        device = 'cpu',
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        toy: bool = False,
):
    model = models[0]
    intro_model = models[1]
    # 1 Create dataset
    os.makedirs(TMP_DIR, exist_ok=True)
    if toy:
        sw_dataset = SimpleListDataset(USE_PATH.strong.toy)
    else:
        sw_dataset = SimpleListDataset(USE_PATH.strong.train)

    acc_k = []
    prec_k = []
    rec_k = []
    f1_k = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(sw_dataset), start=1):
        n_train = len(train_idx)
        n_val = len(valid_idx)
        # 2.b Create interface
        # n_val = int((len(sw_dataset))*val_percent)
        # n_train = len(sw_dataset) - n_val
        # train_set, val_set = torch.utils.data.random_split(sw_dataset, [n_train, n_val])
        train_set = MySubset(sw_dataset, train_idx)
        val_set = MySubset(sw_dataset, valid_idx)
        
        interface = BlkPosInterface(train_set)
        interface_val = BlkPosInterface(val_set)

        ## Original module part
        # 3. Create data loaders (args)
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

        # train_loader = DataLoader(train_set, shuffle=True, **loader_args)

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Fold:            {fold}
        ''')

        # 4. Set up the optimizer, the loss and the learning rate scheduler
        DO_VALID = True
        optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

        global_step = 0
        # 5. Begin training
        # epoch = 1
        for epoch in range(1, epochs + 1):
            # Update interface
            train_set = interface.build_strong_buffer()
                
            train_loader = DataLoader(train_set, shuffle=True, 
                                        collate_fn = buffer_collate, # 讓dataloader可以迭代buffer類
                                        **loader_args
                                        )

            model.train()
            epoch_loss = 0
            batch_steps = 0
            epoch_sum = 0
            epoch_len = 0
            epoch_tp = 0
            epoch_fn = 0
            epoch_fp = 0
            for bufs in (pbar:=tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit=f'Paragraph({batch_size})')) : 
                batch_steps += 1
                # Make inputs for reasoner
                inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long, device=device)  # [ 3, BATCH, 512 ]
                # 因為tokenizer.convert_ids_to_tokens(0) = '[PAD]' 所以等於是天生PAD然後再填Buffer每個Block的ids進去 (buf.export那裡的操作)
                blk_pos = []
                for i, buf in enumerate(bufs):
                    # export -> ids, att_masks, types, blk_position
                    _, _, _, b_p = buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 其實搞不太懂怎麼用export把buf的資訊給inputs的 python還能搞指標的?
                    blk_pos.append(b_p) # b_p : [ 1, NUM_OF_BLOCK ] 選到的(從interface那邊)每個block在相應buffer中的位置(之後損失函數那邊才可跟label比)
                # Extract the labels for reasoner, e.g. start and end position for QA reasoner
                # crucials (list) : BATCH * [1, NUM_OF_BLOCK_blk_type==0]
                labels, crucials = model.export_labels(bufs, device) # TODO A
                result = model(*inputs, labels=labels, pos = blk_pos, device = device, debug_buf=bufs)
                
                losses = result[0] if isinstance(result, tuple) else result
                # loss = sum(losses)/len(losses) # Mean or Sum ?
                loss = sum(losses)
                
                logits = result[2]
                local_labels = result[1]
                softmax_preds = [F.softmax(logit, dim = 1) for logit in logits]
                preds = [torch.max(s_pred, dim=1).indices for s_pred in softmax_preds]
                # sum_correct = sum(sum(ans) for ans in map(lambda x, y : x==y, preds, local_labels))
                # len_label = sum(len(lab) for lab in local_labels)
                
                preds_list = np.array(sum([pred.tolist() for pred in preds], []))
                labels_list = np.array(sum([lab.tolist() for lab in local_labels], []))
                # sum_correct = sum(map(lambda x, y : x==y, preds_list, labels_list))
                sum_correct = (preds_list == labels_list).sum()
                len_label = len(labels_list)
                
                acc_batch = (sum_correct / len_label).item()
                epoch_sum += sum_correct
                epoch_len += len_label
                
                batch_tp = np.logical_and(labels_list == 1, preds_list == 1).sum(axis=0)
                batch_fn = np.logical_and(labels_list == 1, preds_list == 0).sum(axis=0)
                batch_fp = np.logical_and(labels_list == 0, preds_list == 1).sum(axis=0)            
                epoch_tp += batch_tp
                epoch_fn += batch_fn
                epoch_fp += batch_fp
                # logging.info(f'Model {m_name} : batch acc -> {acc_batch:.3f}')

                # _intervention(_file, bufs, labels, crucials, result, model)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
                optimizer.step()
                # scheduler.step(acc_batch)

                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{
                    'acc (batch)': acc_batch,
                    'loss (batch)': loss.item()
                    })
                # 'f1 (batch)': 2*batch_tp / (2*batch_tp + batch_fn + batch_fp),
                # 'recall (batch)': batch_tp / (batch_tp + batch_fn),
                # 'precision (batch)': batch_tp / (batch_tp + batch_fp),
                
            logging.info(f"""
                Model {m_name} (epoch) Fold {fold} : 
                    loss      ->  {epoch_loss / batch_steps:.4f} 
                    accuracy  ->  {(epoch_sum/epoch_len).item():.4f}
                    precision ->  {epoch_tp / (epoch_tp + epoch_fp):.4f}
                    recall    ->  {epoch_tp / (epoch_tp + epoch_fn):.4f}
                    f1-score  ->  {2*epoch_tp / (2*epoch_tp + epoch_fn + epoch_fp):.4f}
                        """)
            
            if DO_VALID and not toy :
                # val_set = interface_val.build_strong_buffer()
                # val_loader = DataLoader(val_set, collate_fn=buffer_collate, **loader_args)
                # model.eval()
                total_tp = 0
                total_fn = 0
                total_fp = 0
                total_tn = 0
                total_sum = 0
                total_len = 0
                model.eval()
                intro_model.eval()
                with torch.no_grad() :
                    # for bufs in (pbar:=tqdm(val_loader, desc='Validation', unit=f'Paragraph({batch_size})')) :
                    for qbuf, dbuf in tqdm(val_set, desc='valid..'): 
                        dbuf_label = [blk.choose for blk in dbuf.blocks]
                        # pdb.set_trace()
                        # 推論實在是太慢了
                        buf, relevance_score = mem_replay(intro_model, qbuf, dbuf, times='3,5', device=device) # TODO times hyperparam
                        # Model預設想吃多BATCH 故要unsqueeze讓他多一維
                        info = [t for t in buf.export(device=device)]
                        inputs = [t.unsqueeze(0) for t in info if not isinstance(t, list)]
                        # *[input_ids, attn_mask, token_type_ids]
                        output = model(*inputs)

                        # 這裡只會有一維(batch_size=1)而已
                        softmax_preds = F.softmax(output[0][0], dim = 1)
                        preds = torch.max(softmax_preds, dim=1).indices

                        preds_list = [i+1 for i, p in enumerate(preds) if p==1]
                        # selected_blk = itemgetter(*preds_list)(buf.blocks)
                        selected_blk = [buf.blocks[i] for i in preds_list]
                        # trans to paragraph string
                        selected_point = [tokenizer.decode(blk.ids[:-1]).replace(' ', '') for blk in selected_blk]
                        selected_blk_list = [blk.place for blk in buf.blocks]
                        selected_blk_list.pop(0)
                        preds_reason = np.zeros(len(dbuf_label))
                        preds_reason[np.array(selected_blk_list)-1] = 1
                        # yield qbuf, dbuf, buf, relevance_score, inputs[0][0], output

                        globe_label = np.array(dbuf_label)
                        round_tp = np.logical_and(globe_label == 1, preds_reason == 1).sum(axis=0)
                        round_fn = np.logical_and(globe_label == 1, preds_reason == 0).sum(axis=0)
                        round_fp = np.logical_and(globe_label == 0, preds_reason == 1).sum(axis=0)
                        round_tn = np.logical_and(globe_label == 0, preds_reason == 0).sum(axis=0)

                        total_tp += round_tp
                        total_fn += round_fn
                        total_fp += round_fp
                        total_tn += round_tn
                        total_sum += (globe_label==preds_reason).sum()
                        total_len += len(preds_reason)

                        # print(f'Original Para len: {len(total_para)-total_blk_num} | Summary Para len: {sum_para_num}')
                        # print(f"""
                        # Test data :
                        #     accuracy  ->  {((globe_label==preds_reason).sum()/len(preds_reason)).item():.4f}
                        #     precision ->  {round_tp / (round_tp + round_fp):.4f}
                        #     recall    ->  {round_tp / (round_tp + round_fn):.4f}
                        #     f1-score  ->  {2*round_tp / (2*round_tp + round_fn + round_fp):.4f}
                        #         """)
                print(f"""
                    Test data final result :
                        accuracy  ->  {total_sum/total_len:.4f}
                        precision ->  {total_tp / (total_tp + total_fp):.4f}
                        recall    ->  {total_tp / (total_tp + total_fn):.4f}
                        f1-score  ->  {2*total_tp / (2*total_tp + total_fn + total_fp):.4f}
                    """)
                acc_k.append(total_sum/total_len)
                prec_k.append(total_tp / (total_tp + total_fp))
                rec_k.append(total_tp / (total_tp + total_fn))
                f1_k.append(2*total_tp / (2*total_tp + total_fn + total_fp))
    logging.info(f"""
    Model {m_name} (validation) Cross Validation (3 Fold) : 
        accuracy  ->  {sum(acc_k)/len(acc_k):.4f}
        precision ->  {sum(prec_k)/len(prec_k):.4f}
        recall    ->  {sum(rec_k)/len(rec_k):.4f}
        f1-score  ->  {sum(f1_k)/len(f1_k):.4f}
            """)        
        # We don't need to save checkpoint here in the validation mode, unless it is ready to release.
        # if save_checkpoint:
        #     dir_ch_this = Path(dir_checkpoint / 'checkpoint' / m_name)
        #     Path(dir_ch_this).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     torch.save(state_dict, str(dir_ch_this / 'checkpoint_epoch{}_{}.pth'.format(epoch, m_name)))
        #     logging.info(f'Model {m_name} : Checkpoint {epoch} saved!')





            
def get_args():
    parser = argparse.ArgumentParser(description='Train the ALLonBERT on social work data')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--load2', '-f2', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--log-level', '-g', type=bool, default=False, help='')
    parser.add_argument('--model-name', '-m', type=str, default=None,
                        choices = ['default', 'large'], help='Specifiy the name of BERT pre-trained model')
    parser.add_argument('--baseline', action='store_true',
                        help='Train Reasoner model')
    parser.add_argument('--judge', action='store_true',
                        help='Train Judge model')
    parser.add_argument('--hier', action='store_true',
                        help='Train Hier model')
    parser.add_argument('--toy', action='store_true',
                        help='Use toy dataset')
    parser.add_argument('--valid', action='store_true',
                        help='Do 3 fold validation?')
    parser.add_argument('--data', type=str, default='sw_data', help='Data path prefix')
    parser.add_argument('--model-num', '-mn', type=int, default=4, help='ALLonBert_v<>', choices=[2, 3, 4])
    return parser.parse_args()

    
if __name__ == '__main__':
    args = get_args()
    log_level = logging.INFO
    if args.log_level :
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    DATA_NAME = args.data
        
    # Can chain call
    USE_PATH = AttributeDict({
        'strong' : AttributeDict({
            'train' : Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_strong_train.pkl')),
            'test' : Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_strong_test.pkl')),
            'toy' : Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_strong_toy_train.pkl'))
        }),
        'weak' : AttributeDict({
            'train' : Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_weak_train.pkl')),
            'toy' : Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_weak_toy_train.pkl'))       
        })
    })


    # Set Model and Tokenizer
    if args.model_name is not None and args.model_name == 'large' :
        MODEL_NAME = BIG_MODEL_NAME # 有指定的話就給個新的
    
    MODEL_CLASS = eval(f'ALLonBert_v{str(args.model_num)}')
   
    reasoner = MODEL_CLASS(MODEL_NAME).to(device)
    judger = Introspector(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # logging.debug(f'Network:\n\t## Judge:\n\t\t{models[0]}\n\n\t## Reasoner:\n\t\t{models[1]}')

    if args.load: # 之後再改 反正先False
        state_dict = torch.load(args.load, map_location=device)
        reasoner.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
        
    if args.load2: # 之後再改 反正先False
        state_dict = torch.load(args.load2, map_location=device)
        judger.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load2}')

    # if args.baseline==
    
    # try:
    if args.judge:
        sep_train_weak(
            model = judger, # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
            m_name = 'Judge', # model_name
            device = device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,  
            toy=args.toy,     
        )
    if args.baseline:
        sep_train(
            model = reasoner, # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
            m_name = 'Reasoner', # model_name
            device = device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            toy=args.toy,
        )
    if args.valid:
        valid_train(
            models = [reasoner, judger], # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
            m_name = 'Valid-Mix', # model_name
            device = device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            toy=args.toy,
        )
    # if args.hier:
    #     encoder = AutoModel.from_pretrained("bert-base-chinese")
    #     hier_model = HierarchicalBert(encoder).to(device)
        
    #     hier_train(
    #         model = hier_model, # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
    #         m_name = 'hier', # model_name
    #         device = device,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         toy=args.toy,
    #     )