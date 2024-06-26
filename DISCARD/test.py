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
    parser.add_argument('--model-name', '-m', type=str, default=None, help='Specifiy the name of BERT pre-trained model')

    return parser.parse_args()


print(f"Dir now : {os.getcwd()}")

dir_checkpoint = Path(SAVE_DIR)

def _write_estimation(_file, buf, relevance_blk):
    for i, blk in enumerate(buf):
        _file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

def _write_changes(_file, blk, key, value): # intervention才會用到的東西
    _file.write('{} {} {}\n'.format(blk.pos, key, value))


args = get_args()

logging.basicConfig(level=logging.DEBUG if args.log_level else logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Set Model and Tokenizer
if args.model_name is not None :
    MODEL_NAME = args.model_name
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

models=models
epochs=args.epochs
batch_size=args.batch_size
learning_rate=args.lr
device=device
save_checkpoint=False

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
# for epoch in range(1, epochs + 1):

# =========
epoch = 1

m_idx = 1
model = models[m_idx]
# =========

# Update interface
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

# =========
BUF = iter(train_loader)
bufs = next(BUF)
# =========

# for bufs in train_loader : # batch 也許可以考慮一個batch是 J -> R -> J
batch_steps += 1
# if m_idx == 0 :
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
# else :
# Make inputs for reasoner
inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long, device=device) # [ 3, BATCH, 512 ]
# 因為tokenizer.convert_ids_to_tokens(0) = '[PAD]' 所以等於是天生PAD然後再填Buffer每個Block的ids進去 (buf.export那裡的操作)
blk_pos = []
for i, buf in enumerate(bufs):
    # export -> ids, att_masks, types, blk_position
    _, _, _, b_p = buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i])) # 其實搞不太懂怎麼用export把buf的資訊給inputs的 python還能搞指標的?
    blk_pos.append(b_p) # b_p : [ 1, NUM_OF_BLOCK ]  選到的(從interface那邊)每個block在相應buffer中的位置(之後損失函數那邊才可跟label比)
# Extract the labels for reasoner, e.g. start and end position for QA reasoner
labels, crucials = model.export_labels(bufs, device) # TODO A
result = model(*inputs, labels=labels, pos = blk_pos)
result = result[0] if isinstance(result, tuple) else result
loss_reasoner = result
loss = sum(result).mean()
if temp_loss is not None :
    temp_loss += loss.item() # connect the Judge loss above
else :
    temp_loss = loss.item()





# ==================== CHANGE ==================== #

loss_reasoner = torch.FloatTensor(loss_reasoner).detach() # 這輪本來的Loss

# Paper default
batch_size_reason_per_gpu = 4

max_bs = batch_size_reason_per_gpu * 4
max_blk_num = max([len(buf) for buf in bufs]) # max blk_num in this batch
# for i in range(len(bufs)): # num of batch

# =========
i = 2
# =========

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
logging.info(f"Hello {(bs - 1) // max_bs + 1} and {bs} and {max_bs}")

for j in range((bs - 1) // max_bs + 1): 
    l, r = max_bs * j, min(bs, max_bs * (j + 1)) # 丟一個batch的概念 這裡反正只有7個就一次丟了
    result = model(ids[l:r], attn_masks[l:r], type_ids[l:r], labels=label[l:r], pos = blk_pos[l:r])
    # result = result[0] if isinstance(result, tuple) else result
    losses.append(result)
try :   
    losses_delta = torch.cat(losses, dim=0) - loss_reasoner[i]
except RuntimeError :
    losses_delta = [lk.detach() - loss_reasoner[i] for lk in losses[0]] # 和reasoner的差
# Label relevance
t = 0

##################################################################
##### 終於找到你啦
##################################################################

# Paper default
levelup_threshold = 0.2
leveldown_threshold = -0.05

# =========
blk = bufs[i][1]
# =========

for blk in bufs[i]:
    if blk in crucials[i]:
        pass
        # self._write_changes(blk, 'relevance', 3)
    else:
        # 移掉這一塊有負面影響 (loss上升了)
        if losses_delta[t] >= levelup_threshold and blk.relevance < 2: # TODO topk
            _write_changes(blk, 'relevance', blk.relevance + 1) # 直接更新那個塊的relevance
        # 移掉這一塊沒啥差 (或甚至loss還降了)
        elif losses_delta[t] <= leveldown_threshold and blk.relevance > -1:
            _write_changes(blk, 'relevance', blk.relevance - 1)
        t += 1
assert t == bs






