from utils.memreplay import mem_replay
from scripts.buffer import Buffer
from models.model import ALLonBert_v2
from utils.util import SAVE_DIR, TMP_DIR, LOG_DIR, USE_PATH, CAPACITY, MODEL_NAME, BIG_MODEL_NAME
from scripts.data_helper import SimpleListDataset

from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter 
from transformers import AutoTokenizer

CHK_DIR = Path(SAVE_DIR / 'checkpoint')
R_DIR = Path(CHK_DIR / 'Reasoner')

# / model_name (Reasoner|Judge) / checkpoint_epoch<>_<model_name>.pth

# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

state_dict_reason = torch.load(Path(R_DIR / '{}_checkpoint_epoch.pth'.format('Reasoner')), map_location=device)
reason_model = ALLonBert_v2(MODEL_NAME).to(device)
reason_model.load_state_dict(state_dict_reason)

sw_dataset = SimpleListDataset(USE_PATH.strong.test)

# We using sliding windows here to evalute the test data (as baseline)
# We define stride with # of document here for convenience

total_tp = 0
total_fn = 0
total_fp = 0
total_sum = 0
total_len = 0

for i, para in tqdm(enumerate(sw_dataset)):
    qbuf = para[0]
    dbuf = para[1]
    global_label = []
    global_preds = []
    for blk in dbuf.blocks:
        buf = Buffer()
        buf.blocks = [blk]
        info = [t for t in buf.export(device=device)]
        
        # Add [CLS]
        info[0] = torch.cat([torch.tensor(101, device=device).view(-1), info[0]])
        info[1] = torch.cat([torch.tensor(1, device=device).view(-1), info[1]])
        info[2] = torch.cat([torch.tensor(1, device=device).view(-1), info[2]])
        
        inputs = [t.unsqueeze(0) for t in info if not isinstance(t, list)]
        output = reason_model(*inputs)
        softmax_preds = F.softmax(output[0][0], dim = 1)
        pred = torch.max(softmax_preds, dim=1).indices
        whatif = pred.item() == blk.choose
        global_label.append(blk.choose)
        global_preds.append(pred.item())
        
    round_label = np.array(global_label)
    round_preds = np.array(global_preds)
    round_tp = np.logical_and(round_label == 1, round_preds == 1).sum(axis=0)
    round_fn = np.logical_and(round_label == 1, round_preds == 0).sum(axis=0)
    round_fp = np.logical_and(round_label == 0, round_preds == 1).sum(axis=0)
    total_tp += round_tp
    total_fn += round_fn
    total_fp += round_fp
    total_sum += (round_label==round_preds).sum()
    total_len += len(round_preds)
    print(f"""
        Round {i+1} :
            accuracy  ->  {((round_label==round_preds).sum()/len(round_preds)).item():.4f}
            precision ->  {round_tp / (round_tp + round_fp):.4f}
            recall    ->  {round_tp / (round_tp + round_fn):.4f}
            f1-score  ->  {2*round_tp / (2*round_tp + round_fn + round_fp):.4f}
        """)
    
print(f"""
    Test data final result :
        accuracy  ->  {total_sum/total_len:.4f}
        precision ->  {total_tp / (total_tp + total_fp):.4f}
        recall    ->  {total_tp / (total_tp + total_fn):.4f}
        f1-score  ->  {2*total_tp / (2*total_tp + total_fn + total_fp):.4f}
    """)