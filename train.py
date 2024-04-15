## other
import argparse
import logging # 我把它當成print的替代
import os
import random
import sys
from pathlib import Path # 會幫忙處理路徑格式
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from copy import deepcopy

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split

## project
from utils.util import SAVE_DIR, TMP_DIR, LOG_DIR, TRAIN_SRC, TEST_SRC, CAPACITY, MODEL_NAME
from utils.memreplay import mem_replay, _score_blocks
from scripts.data_helper import SimpleListDataset, BlkPosInterface, find_lastest_checkpoint
from scripts.buffer import buffer_collate
from models.model import Introspector, ALLonBert

print(f"Dir now : {os.getcwd()}")

dir_checkpoint = Path(SAVE_DIR)

def _write_estimation(_file, buf, relevance_blk):
    for i, blk in enumerate(buf):
        _file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

def _write_changes(_file, blk, key, value): # intervention才會用到的東西
    _file.write('{} {} {}\n'.format(blk.pos, key, value))


def train_model(
        models, # 之後可以考慮加入判斷是不適list的來一次練兩個 (已經改了)
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
                interface.collect_estimations_from_dir(TMP_DIR)
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
                        if temp_loss is not None :
                            logging.debug(f"\n#########\nAdd Reasoner Loss\n#########\n")
                            t_loss = temp_loss
                            loss += t_loss # connect the Reasoner loss below
                        for i, buf in enumerate(bufs):
                            _write_estimation(_file, buf, _score_blocks(buf, torch.sigmoid(logits[i]))) # 把這輪跑完的relevance更新到檔案上
                    else :
                        # Make inputs for reasoner
                        inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long, device=device)
                        # 因為tokenizer.convert_ids_to_tokens(0) = '[PAD]' 所以等於是天生PAD然後再填Buffer每個Block的ids進去 (buf.export那裡的操作)
                        blk_pos = []
                        for i, buf in enumerate(bufs):
                            _, _, _, b_p = buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 其實搞不太懂怎麼用export把buf的資訊給inputs的 python還能搞指標的?
                            blk_pos.append(b_p) # 選到的(從interface那邊)每個block在相應buffer中的位置(之後損失函數那邊才可跟label比)
                        # Extract the labels for reasoner, e.g. start and end position for QA reasoner
                        labels, crucials = model.export_labels(bufs, device) # TODO A
                        result = model(*inputs, labels=labels, pos = blk_pos)
                        result = result[0] if isinstance(result, tuple) else result
                        loss = result.mean()
                        if temp_loss is not None :
                            temp_loss += loss.item() # connect the Judge loss above
                        else :
                            temp_loss = loss.item()

                    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp): # 混和精度

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

            logging.info(f'Model {m_name} : epoch loss -> {epoch_loss[m_idx]}')

            


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

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.log_level else logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Set Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    models = []
    models.append(Introspector.from_pretrained(MODEL_NAME))
    models.append(ALLonBert.from_pretrained(MODEL_NAME))

    logging.debug(f'Network:\n\t## Judge:\n\t\t{models[0]}\n\n\t## Reasoner:\n\t\t{models[1]}')

    if args.load: # 之後再改 反正先False
        state_dict = torch.load(args.load, map_location=device)
        models[0].load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    models = [model.to(device=device) for model in models]
    try:
        train_model(
            models=models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      '完蛋啦 爆炸了')
        torch.cuda.empty_cache()
        # models[0].use_checkpointing() # 也是他原Model有寫的東西 用來把每一層載回來
        train_model(
            model=models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )