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

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split

## project
from utils.util import SAVE_DIR, TMP_DIR, LOG_DIR, USE_PATH, CAPACITY, MODEL_NAME, BIG_MODEL_NAME
from utils.memreplay import mem_replay, _score_blocks
from scripts.data_helper import SimpleListDataset, BlkPosInterface, find_lastest_checkpoint
from scripts.buffer import buffer_collate
from models.model import Introspector, ALLonBert_v2

print(f"Dir : {os.getcwd()}")

dir_checkpoint = Path(SAVE_DIR)

def _write_estimation(_file, buf, relevance_blk):
    for i, blk in enumerate(buf):
        _file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

def _write_changes(_file, blk, key, value): # intervention才會用到的東西
    _file.write('{} {} {}\n'.format(blk.pos, key, value))

def _intervention(_file, bufs, labels, crucials, loss_reasoner, model) :
    loss_reasoner = torch.FloatTensor(loss_reasoner).detach() # 這輪本來的Loss

    # Paper default parameter :
    batch_size_reason_per_gpu = 4
    levelup_threshold = 0.2
    leveldown_threshold = -0.05
    
    with torch.no_grad():
        max_bs = batch_size_reason_per_gpu * 4
        max_blk_num = max([len(buf) for buf in bufs]) # max blk_num in this batch
        for i in range(len(bufs)): # num of batch
            ids, attn_masks, type_ids, blk_pos = bufs[i].export(device=device)
            bs = len(bufs[i]) - len(crucials[i]) # 需要處理的句數 (crucials在qa是問題那句(blk_type==0) 當然重要)
            # Make inputs by expand with different attention masks
            ids = ids.view(1, -1).expand(bs, -1) # 把ids複製bs份 (這樣每句可以有各自的att_mask來看情況)
            type_ids = type_ids.view(1, -1).expand(bs, -1) # 就先當沒用吧 好像都是0 以後再看要不要當成embbeding來用
            attn_masks = attn_masks.view(1, -1).repeat(bs, 1) # 跟expand意思應該一樣 可能在炫技
            label = torch.FloatTensor(labels[i]).view(1, -1).expand(bs, -1) # 一樣把label expand
            # label = [labels[i] for _ in range(bs)] # 不是阿那寫上面那個幹嘛阿 哦不是這是我寫的
            blk_pos = torch.IntTensor(blk_pos).view(1, -1).expand(bs, -1)
            blk_start, t = 0, 0
            for blk in bufs[i]:
                blk_end = blk_start + len(blk)
                if blk not in crucials[i]:
                    # 把要測的那句的att_mask設成0
                    attn_masks[t, blk_start: blk_end].zero_() 
                    t += 1
                blk_start = blk_end
            assert t == bs
            # ForkedPdb().set_trace()
            # if bs > max_bs, we cannot feed the inputs directly.
            losses = []
            logging.debug(f"Hello {(bs - 1) // max_bs + 1} and {bs} and {max_bs}")

            for j in range((bs - 1) // max_bs + 1): 
                l, r = max_bs * j, min(bs, max_bs * (j + 1)) # 丟一個batch的概念 這裡反正只有7個就一次丟了
                result = model(ids[l:r], attn_masks[l:r], type_ids[l:r], labels=label[l:r], pos = blk_pos[l:r])
                # result = result[0] if isinstance(result, tuple) else result
                losses.append(result)
            try :   
                losses_delta = torch.cat(losses, dim=0) - loss_reasoner[i]
            except Exception as e :
                logging.debug(e)
                losses_delta = [lk.detach() - loss_reasoner[i] for lk in losses[0]] # 和reasoner的差
            # Label relevance
            t = 0

            ##################################################################
            ##### 終於找到你啦
            ##################################################################

            for blk in bufs[i]:
                if blk in crucials[i]:
                    pass
                    # self._write_changes(blk, 'relevance', 3)
                else:
                    # 移掉這一塊有負面影響 (loss上升了)
                    if losses_delta[t] >= levelup_threshold and blk.relevance < 2: # TODO topk
                        _write_changes(_file, blk, 'relevance', blk.relevance + 1) # 直接更新那個塊的relevance
                    # 移掉這一塊沒啥差 (或甚至loss還降了)
                    elif losses_delta[t] <= leveldown_threshold and blk.relevance > -1:
                        _write_changes(_file, blk, 'relevance', blk.relevance - 1)
                    t += 1
            assert t == bs


def train_model(
        models, # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        # img_scale: float = 0.5,
        # amp: bool = False,
        # weight_decay: float = 1e-8,
        # momentum: float = 0.999,
        # gradient_clipping: float = 1.0,
):
    # 1 Create dataset
    os.makedirs(TMP_DIR, exist_ok=True)
    sw_dataset = SimpleListDataset(TRAIN_SRC)

    # 2.a Split into train / validation partitions
    # n_val = int(len(sw_dataset) * val_percent)
    # n_train = len(sw_dataset) - n_val
    # train_set, val_set = random_split(sw_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2.b Create interface
    train_set = sw_dataset
    n_train = len(train_set)
    n_val = 0
    interface = BlkPosInterface(train_set)
    # interface_valid = BlkPosInterface(val_set)

    ## Original module part
    # 3. Create data loaders (args)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    # , drop_last=True

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    optimizer = []
    scheduler = []

    optimizer.append(optim.AdamW(models[0].parameters(), lr = learning_rate))
    scheduler.append(optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], 'min', patience=5))  # goal: maximize Dice score
    
    optimizer.append(optim.AdamW(models[1].parameters(), lr = learning_rate))
    scheduler.append(optim.lr_scheduler.ReduceLROnPlateau(optimizer[1], 'min', patience=5))

    

    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() 
    # if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    temp_loss = None
    for epoch in range(1, epochs + 1):
        # Update interface
        for m_idx, model in enumerate(models) :
            m_name = "Judge" if not m_idx else "Reasoner"
            # print(f"Training {m_name}...")
            logging.info(f"\n\n{'=' * 10} {m_name:^10} {'=' * 10}\n\n")
            if temp_loss is not None :
                logging.debug(f"\n\n{'=' * 10} {temp_loss:^10} {'=' * 10}\n\n")
            # 建一個新的記錄檔
            if m_idx == 0 :
                _file = open(Path(os.path.join(TMP_DIR, 'estimations_{}.txt'.format(device))), 'w')
            else :
                _file = open(os.path.join(TMP_DIR, 'changes_{}.txt'.format(device)), 'w')
                
            # 每輪都要更新一次資料 (不知道會部會跟hw3一樣又因為num_workers卡住)
            if m_idx == 0 : # Judge
                train_set = interface.build_random_buffer(num_samples = '1,1,1,1') # 我其實也還不知道這怎麼搞
            else : # Reasoner
                if temp_loss is not None :
                    temp_loss = None
                interface.collect_estimations_from_dir(TMP_DIR) # 上一輪judge跑出來的mean( sigmoid(logits) )
                train_set = interface.build_promising_buffer(num_samples = '1,1,1,1')
                
            train_loader = DataLoader(train_set, shuffle=True, 
                                      collate_fn = buffer_collate, # 讓dataloader可以迭代buffer類
                                      **loader_args
                                      )

            model.train()
            epoch_loss = [0, 0]
            batch_steps = 0
            with tqdm(total=n_train*2, desc=f'Epoch {epoch}/{epochs}', unit=' item') as pbar:

                for bufs in train_loader : # batch 也許可以考慮一個batch是 J -> R -> J
                    batch_steps += 1
                    if m_idx == 0 :
                        # images, true_masks = batch['image'], batch['mask']
                        inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long, device=device)
                        for i, buf in enumerate(bufs):
                            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 和reasoner一樣設定input
                        # Train the introspector after labeling
                        for i, buf in enumerate(bufs):
                            buf.export_relevance(device=device, out=inputs[3, i]) # 用來設定judge的label(由relevance)
                        # Label the relevance by the current reasoner  
                        loss, logits = model(*inputs[:3], labels=inputs[3])
                        for i, buf in enumerate(bufs):
                            # _score_blocks把原本每個token各一個的分數轉為每個block(句子)一個
                            _write_estimation(_file, buf, _score_blocks(buf, torch.sigmoid(logits[i]))) # 把這輪跑完的relevance更新到檔案上
                    else :
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
                        result = model(*inputs, labels=labels, pos = blk_pos)
                        result = result[0] if isinstance(result, tuple) else result # loss of reasoner
                        loss = sum(result).mean()

                        ### 感覺可以考慮做個local loss監控一下reasoner的訓練情況
                        ### 另外有必要去檢查一下interface 總覺得常常看到整個batch都是一樣內容的情況
                        ### 還是我錯怪interface了阿 也可能是跑intervention的時候跳的 嗎

                        _intervention(_file, bufs, labels, crucials, result, model)

                    optimizer[m_idx].zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
                    optimizer[m_idx].step()

                    pbar.update(len(train_loader)) # 吧?
                    logging.debug(f"Loader len : {len(train_loader)}")

                    global_step += 1
                    epoch_loss[m_idx] += loss.item()
                    # experiment.log({
                    #     'train loss': loss.item(),
                    #     'step': global_step,
                    #     'epoch': epoch
                    # })
                    pbar.set_postfix(**{'loss (batch)': loss.item()}) # 這東西顯示怪怪的
                    

                    # Evaluation round
                    # division_step = (n_train // (5 * batch_size))
                    # if division_step > 0:
                    #     if global_step % division_step == 0:
                    #         histograms = {}
                    #         for tag, value in model.named_parameters():
                    #             tag = tag.replace('/', '.')
                    #             if not (torch.isinf(value) | torch.isnan(value)).any():
                    #                 # histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    #             if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    #                 # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    #         val_score = evaluate(model, val_loader, device, amp)
                    #         scheduler.step(val_score)

                    #         logging.info('Validation Dice score: {}'.format(val_score))
                            # try:
                            #     experiment.log({
                            #         'learning rate': optimizer.param_groups[0]['lr'],
                            #         'validation Dice': val_score,
                            #         'images': wandb.Image(images[0].cpu()),
                            #         'masks': {
                            #             'true': wandb.Image(true_masks[0].float().cpu()),
                            #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            #         },
                            #         'step': global_step,
                            #         'epoch': epoch,
                            #         **histograms
                            #     })
                            # except:
                            #     pass
                scheduler[m_idx].step(loss)
            
            _file.close()
            if temp_loss is not None :
                temp_loss /= batch_steps
            
            if save_checkpoint:
                dir_ch_this = Path(dir_checkpoint / 'checkpoint' / m_name)
                Path(dir_ch_this).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(dir_ch_this / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Model {m_name} : Checkpoint {epoch} saved!')

            logging.info(f'Model {m_name} : epoch loss -> {epoch_loss[m_idx] / batch_steps}')
        if epoch > 1 :
            interface.apply_changes_from_dir(TMP_DIR)

# from importlib import reload
# import models.model as mod
# # from utils.util import SAVE_DIR, TMP_DIR, LOG_DIR, USE_PATH, CAPACITY, MODEL_NAME, BIG_MODEL_NAME
# import utils.util as utl
# # from scripts.data_helper import SimpleListDataset, BlkPosInterface
# import scripts.data_helper as hep
# reload(mod) # 終於找到了
# reload(utl) 
# reload(hep)
# BlkPosInterface = hep.BlkPosInterface
# USE_PATH = utl.USE_PATH
# model = mod.Introspector(MODEL_NAME) # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
# m_name = 'Judge' # model_name
# device = 'cpu'
# epochs: int = 5
# batch_size: int = 3
# learning_rate: float = 1e-5
# val_percent: float = 0.1
# save_checkpoint: bool = True


def sep_train_weak(
        model = Introspector(MODEL_NAME), # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        m_name = 'Judge', # model_name
        device = 'cpu',
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
):
    # 1 Create dataset
    os.makedirs(TMP_DIR, exist_ok=True)
    # sw_dataset = SimpleListDataset(USE_PATH.weak.toy)
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)
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
            scheduler.step()

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
):
    # 1 Create dataset
    os.makedirs(TMP_DIR, exist_ok=True)
    # sw_dataset = SimpleListDataset(USE_PATH.strong.toy)
    sw_dataset = SimpleListDataset(USE_PATH.strong.train)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

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
            result = model(*inputs, labels=labels, pos = blk_pos, device = device)
            
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
            scheduler.step(acc_batch)

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
            Model {m_name} (epoch) : 
                loss      ->  {epoch_loss / batch_steps:.4f} 
                accuracy  ->  {(epoch_sum/epoch_len).item():.4f}
                precision ->  {epoch_tp / (epoch_tp + epoch_fp):.4f}
                recall    ->  {epoch_tp / (epoch_tp + epoch_fn):.4f}
                f1-score  ->  {2*epoch_tp / (2*epoch_tp + epoch_fn + epoch_fp):.4f}
                    """)
        
        if DO_VALID :
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
                    loss = sum(losses)/len(losses) # Mean or Sum ?
                    
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
            Model {m_name} (validation) : 
                loss      ->  {vepoch_loss / vbatch_steps:.4f} 
                accuracy  ->  {(vepoch_sum/vepoch_len).item():.4f}
                precision ->  {vepoch_tp / (vepoch_tp + vepoch_fp):.4f}
                recall    ->  {vepoch_tp / (vepoch_tp + vepoch_fn):.4f}
                f1-score  ->  {2*vepoch_tp / (2*vepoch_tp + vepoch_fn + vepoch_fp):.4f}
                    """)

        if save_checkpoint:
            dir_ch_this = Path(dir_checkpoint / 'checkpoint' / m_name)
            Path(dir_ch_this).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_ch_this / 'checkpoint_epoch{}_{}.pth'.format(epoch, m_name)))
            logging.info(f'Model {m_name} : Checkpoint {epoch} saved!')
            


def get_args():
    parser = argparse.ArgumentParser(description='Train the ALLonBERT on social work data')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--log-level', '-g', type=bool, default=False, help='')
    parser.add_argument('--model-name', '-m', type=str, default=None,
                        choices = ['default', 'large'], help='Specifiy the name of BERT pre-trained model')

    return parser.parse_args()

    
if __name__ == '__main__':
    args = get_args()
    log_level = logging.INFO
    if args.log_level :
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Set Model and Tokenizer
    if args.model_name is not None and args.model_name == 'large' :
        MODEL_NAME = BIG_MODEL_NAME # 有指定的話就給個新的
        
    reasoner = ALLonBert_v2(MODEL_NAME).to(device)
    judger = Introspector(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # logging.debug(f'Network:\n\t## Judge:\n\t\t{models[0]}\n\n\t## Reasoner:\n\t\t{models[1]}')

    if args.load: # 之後再改 反正先False
        state_dict = torch.load(args.load, map_location=device)
        reasoner.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    # try:
    sep_train_weak(
        model = judger, # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        m_name = 'Judge', # model_name
        device = device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,       
    )
    sep_train(
        model = reasoner, # 之後可以考慮加入判斷是不是list的來一次練兩個 (已經改了)
        m_name = 'Reasoner', # model_name
        device = device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   '完蛋啦 爆炸了')
    #     torch.cuda.empty_cache()
    #     # models[0].use_checkpointing() # 也是他原Model有寫的東西 用來把每一層載回來
    #     train_model(
    #         models=reasoner,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #     )