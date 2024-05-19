import re
import os
import shutil
from pathlib import Path
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# dirName = '/content/drive/MyDrive/資工專題/checkpoint/'

SAVE_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'saved_dir'))
CHK_DIR = Path(SAVE_DIR / 'checkpoint')
J_DIR = Path(CHK_DIR / 'Judge')
R_DIR = Path(CHK_DIR / 'Reasoner')

def find_latest_checkpoint(check_dir, epoch=False):
  latest = (-1, '')
  if os.path.exists(check_dir):
    for file_name in os.listdir(check_dir):
      re_obj = re.match('checkpoint_epoch(.*)_(.*)', file_name)
      try:
        epoch_record = int(re_obj.group(1))
        model_name = re_obj.group(2)
        if epoch_record > latest[0]:
          latest = (epoch_record, file_name)
      except:
        pass

  return Path(check_dir / latest[-1]) if not epoch else latest[0]

# shutil.copyfile(find_latest_checkpoint(J_DIR), dirName + 'Judge_checkpoint.pth')
# shutil.copyfile(find_latest_checkpoint(R_DIR), dirName + 'Reasoner_checkpoint.pth')
try:
  shutil.copyfile(find_latest_checkpoint(J_DIR), Path(J_DIR / 'Judge_checkpoint_epoch.pth'))
except:
  print('Baseline evaluating...')
shutil.copyfile(find_latest_checkpoint(R_DIR), Path(R_DIR / 'Reasoner_checkpoint_epoch.pth'))





from utils.memreplay import mem_replay
from scripts.buffer import Buffer
from models.model import Introspector, ALLonBert_v2
from utils.util import SAVE_DIR, TMP_DIR, LOG_DIR, USE_PATH, CAPACITY, MODEL_NAME, BIG_MODEL_NAME
from scripts.data_helper import SimpleListDataset

SET = False # Whether to use large model
if SET:
  MODEL_NAME = BIG_MODEL_NAME

from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
from transformers import AutoTokenizer
import numpy as np

CHK_DIR = Path(SAVE_DIR / 'checkpoint')
J_DIR = Path(CHK_DIR / 'Judge')
R_DIR = Path(CHK_DIR / 'Reasoner')

# / model_name (Reasoner|Judge) / checkpoint_epoch<>_<model_name>.pth

# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

times = '3,5'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

state_dict_intro = torch.load(Path(J_DIR / '{}_checkpoint_epoch.pth'.format('Judge')), map_location=device)
intro_model = Introspector(MODEL_NAME).to(device)
intro_model.load_state_dict(state_dict_intro)

state_dict_reason = torch.load(Path(R_DIR / '{}_checkpoint_epoch.pth'.format('Reasoner')), map_location=device)
reason_model = ALLonBert_v2(MODEL_NAME).to(device)
reason_model.load_state_dict(state_dict_reason)

sw_dataset = SimpleListDataset(USE_PATH.strong.test)

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
dirName = '/content/drive/MyDrive/資工專題/result/'
path = dirName + 'result.txt'

total_tp = 0
total_fn = 0
total_fp = 0
total_tn = 0
total_sum = 0
total_len = 0

def tabulate_metrics(tp_fn_fp_tn, main=''):
    msg = f'====== Confusion Matrix: {main} =======\n'
    header = '|'.join([f'{tp_fn_fp_tn[0]:^14}(TP)', f'{tp_fn_fp_tn[1]:^14}(FN)'])
    values = '|'.join([f'{tp_fn_fp_tn[2]:^14}(FP)', f'{tp_fn_fp_tn[3]:^14}(TN)'])
    msg += f"{' '*5}|{header}|\nTrue |{'-----------------:|' * 2}\n{' '*5}|{values}|\n{' '*5}{'Predict':^39}\n"
    return msg

with torch.no_grad():
    with open(path, 'w') as f:
        f.write(f'FORMAT :\n<LABEL>TAB<BLOCK_i_in_PARAGRAPH>\n\n')
        for qbuf, dbuf in tqdm(sw_dataset):
            dbuf_label = [blk.choose for blk in dbuf.blocks]
            # pdb.set_trace()
            # 推論實在是太慢了
            buf, relevance_score = mem_replay(intro_model, qbuf, dbuf, times=times, device=device) # TODO times hyperparam
            # Model預設想吃多BATCH 故要unsqueeze讓他多一維
            info = [t for t in buf.export(device=device)]
            inputs = [t.unsqueeze(0) for t in info if not isinstance(t, list)]
            # *[input_ids, attn_mask, token_type_ids]
            output = reason_model(*inputs)

            # 這裡只會有一維(batch_size=1)而已
            softmax_preds = F.softmax(output[0][0], dim = 1)
            preds = torch.max(softmax_preds, dim=1).indices

            preds_list = [i+1 for i, p in enumerate(preds) if p==1]
            selected_blk = itemgetter(*preds_list)(buf.blocks)
            # trans to paragraph string
            selected_point = [tokenizer.decode(blk.ids[:-1]).replace(' ', '') for blk in selected_blk]
            selected_blk_list = [blk.place for blk in buf.blocks]
            selected_blk_list.pop(0)
            preds_reason = np.zeros(len(dbuf_label))
            preds_reason[np.array(selected_blk_list)-1] = 1
            # yield qbuf, dbuf, buf, relevance_score, inputs[0][0], output

            total_para = ''
            total_blk_num = 0
            for i, blk in enumerate(dbuf.blocks):
                total_para += f'{dbuf_label[i]}\t'
                total_para += tokenizer.decode(blk.ids[:-1]).replace(' ', '')
                if blk.place not in selected_blk_list:
                  total_para += ' --drop' # 沒被選是哪些block
                  total_blk_num += 7
                total_para += '\n'
                total_blk_num += 1
            print({total_para}, end = '\n\n')
            f.write(total_para)
            f.write('\n\n')

            sum_para_num = 0
            for i, para in enumerate(selected_point) :
                f.write(f'{i}):\n\t{para}\n\n')
                print(f'{i}):\n\t{para}', end = '\n\n')
                sum_para_num += len(para)

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

            print(f'Original Para len: {len(total_para)-total_blk_num} | Summary Para len: {sum_para_num}')
            print(f"""
            Test data :
                accuracy  ->  {((globe_label==preds_reason).sum()/len(preds_reason)).item():.4f}
                precision ->  {round_tp / (round_tp + round_fp):.4f}
                recall    ->  {round_tp / (round_tp + round_fn):.4f}
                f1-score  ->  {2*round_tp / (2*round_tp + round_fn + round_fp):.4f}
                    """)
            f.write(f'Original Para len: {len(total_para)-total_blk_num} | Summary Para len: {sum_para_num}\n\n')
            f.write(f"""
            Test data :
                accuracy  ->  {((globe_label==preds_reason).sum()/len(preds_reason)).item():.4f}
                precision ->  {round_tp / (round_tp + round_fp):.4f}
                recall    ->  {round_tp / (round_tp + round_fn):.4f}
                f1-score  ->  {2*round_tp / (2*round_tp + round_fn + round_fp):.4f}
                    \n\n""")

        print(tabulate_metrics([total_tp, total_fn, total_fp, total_tn], 'CogLTX struct evaluation'))

        print(f"""
            Test data final result :
                accuracy  ->  {total_sum/total_len:.4f}
                precision ->  {total_tp / (total_tp + total_fp):.4f}
                recall    ->  {total_tp / (total_tp + total_fn):.4f}
                f1-score  ->  {2*total_tp / (2*total_tp + total_fn + total_fp):.4f}
            """)

        f.write(tabulate_metrics([total_tp, total_fn, total_fp, total_tn], 'CogLTX struct evaluation'))
        f.write(f"""
            Test data final result :
                accuracy  ->  {total_sum/total_len:.4f}
                precision ->  {total_tp / (total_tp + total_fp):.4f}
                recall    ->  {total_tp / (total_tp + total_fn):.4f}
                f1-score  ->  {2*total_tp / (2*total_tp + total_fn + total_fp):.4f}
            """)