# %%
import sys
sys.path.append(r'C:\\vs_code_python\\cogLTX')
import re
import json
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from itertools import chain
import os
import sys
import pickle
import logging
import pdb
from bisect import bisect_left
import string

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)
from buffer import Buffer, Block
from utils import DEFAULT_MODEL_NAME
# from hotpotqa.cogqa_utils import find_start_end_after_tokenized

# from sklearn.datasets import fetch_20newsgroups


# another change

###==========
## Original 20news dataset
###==========

# data_train = fetch_20newsgroups(subset='train', random_state=21)
# data_test = fetch_20newsgroups(subset='test', random_state=21)

# data_train.data[0]
# data_train.target[0]

def textIoU(pred, answer) :
  newList = pred + answer
  length = max(newList) - min(newList) + 1
  start = max(pred[0], answer[0])
  end = min(pred[1], answer[1]) + 1
  IoU = (end - start) / length

  if(IoU < 0) :
    IoU = 0
  if(answer[0] == 511 and answer[1] == 510 and pred[0] == 511 and pred[1] == 510) :
    IoU = 1
  return IoU

def pdftext_to_para(text_data) :
  paragraph = []
  question = []

  para2 = []
  ques2 = []

  txt_len = []
  paraQues = []
  select = []
  temp_text = []
  count = 0
  count_imp = 0
  catch = 0
  catch2 = 0
  p = []

  for sel, text in zip(text_data.choose, text_data.paragraph) :
    if(len(text) > 5 and text[0:5] == "-----") :
      if(len(txt_len) != 0) :
        ### 如果太長了就隨機刪掉幾句不是重點的 Good design demands good compromise.
      #   while(sum(txt_len) > 512) :
      #     delete = random.randrange(0, count)
      #     if(select[delete] != 1 or sum(select) == len(select)) :
      #       select.pop(delete)
      #       txt_len.pop(delete)
      #       paraQues.pop(delete)
      #       p.pop(delete)
      #       count -= 1
        my_str = ""
        for t in p :
          my_str += t
        if(catch == 10 and catch2 < 15) : # 切訓練集測試集 吧
          para2.append(my_str)
          ques2.append([catch2, count, select, txt_len, paraQues, count_imp])
          catch2 += 1
        else :
          paragraph.append(my_str)
          question.append([catch, count, select, txt_len, paraQues, count_imp])
          catch += 1
        txt_len = []
        paraQues = []
        select = []
        temp_text = []
        count = 0
        count_imp = 0
        p = []
    else :
      # p.append("[CLS]" + text + "[SEP]")
      p.append(text)
      temp_text.append(p)
      txt_len.append(len(text))

      if(sel == '1') :
        select.append(1)
        paraQues.append(text)
        count_imp += 1
      else :
        select.append(0)
        paraQues.append(-1)
      count += 1

  ans_position = []
  ans_pos2 = []

  for check_target in question :
     ans_position.append([[sum(check_target[3][:k]), sum(check_target[3][:k])+check_target[3][k]-1] for k in range(len(check_target[2])) if check_target[2][k] == 1])

  for check_target in ques2 :
     ans_pos2.append([[sum(check_target[3][:k]), sum(check_target[3][:k])+check_target[3][k]-1] for k in range(len(check_target[2])) if check_target[2][k] == 1])

  anspos_all = [q[2] for q in question]
  anspos_all2 = [q[2] for q in ques2]

  return [[paragraph, [ans_position, anspos_all]], [para2, [ans_pos2, anspos_all2]]] 
  # [ [ 訓練內文, [訓練標籤相關, 原文本位置] ], [ 測試內文, [測試標籤相關, 原文本位置] ] ]

import pandas as pd
text_data = pd.read_csv(r"C:\Users\chcww\Downloads\社會期刊訓練資料集.csv")

# import random

train_data, test_data = pdftext_to_para(text_data)
# # question = [paraID, 分句句數, 哪些分句是答案, 每句長度, 重點本文, 重點數]
# target_info = train_data[1]

# #======================
# # 透過index看重點 (已完成)
# #======================
# t = 4 # 哪篇
# m = 2 # 哪個重點
# check_data = train_data[0][t]
# check_target = target_info[t]
# one_of_answer_text = check_data[check_target[m][0]:check_target[m][1]+1]
# one_of_answer_text

# # 只顯示答案本文
# ans_list = [check_target[4][k] for k in range(len(check_target[2])) if check_target[2][k] == 1]
# # check_target[3] = [29, 97, 25, 45, 38, 125, 44, 44, 44, 39, 98, 39, 161, 107, 59, 90, 68, 115, 96, 66, 117, 115]
# # 只顯示答案本文的位置
# ans_position = [[sum(check_target[3][:k]), sum(check_target[3][:k])+check_target[3][k]-1] for k in range(len(check_target[2])) if check_target[2][k] == 1]
# # 實際取出來看看對不對
# check_data[ans_position[0][0]:ans_position[0][1]+1]



def clean(data): # 清掉如報導資訊(email或電話啥的)跟換行符號那些 暫時用不到
    tmp_doc = []
    for words in data.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c) 
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc
# %%
def process(para, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    cnt, batches = 0, []
    target_names = ['not_taken', 'taken']
    # for i in tqdm(range(len(para[0]))):
    for i in trange(10):
        # print(i)
        # d, l = clean(para[i]), random.randrange(0, 2) # random是亂打的，就是亂給他一個label
        d, l = para[0][i], [para[1][0][i], para[1][1][i] ] # [ 內文, 標籤相關 ([ 答案token位置, 答案block位置 ]) ] 
        # print()
        # print("#" * 50)
        # print(para[0][i])
        # print("#" * 50)

        label_name = target_names[1]
        # 把這輪輸進來的資料轉成Buffer (cnt是裡面有幾塊)
        # qbuf, cnt, _ = Buffer.split_document_into_blocks([tokenizer.cls_token], tokenizer, cnt=cnt, hard=False, properties=[('label_name', label_name), ('label', l[1]), ('_id', str(i)), ('blk_type', 0)])
        dbuf, cnt, qbuf = Buffer.split_document_into_blocks(tokenizer.tokenize(d), tokenizer, cnt, hard=False, ans_pos = l[0], properties=[('label_name', label_name), ('label', l[1]), ('_id', str(i)), ('blk_type', 0)])
        # qbuf是標籤內容 dbuf是正文
        batches.append((qbuf, dbuf))
    with open(os.path.join(root_dir, 'data', f'20news_{dataset_name}.pkl'), 'wb') as fout: 
        pickle.dump(batches, fout) # 將batches的內容保存到fout中
    return batches
# %%
process(train_data, 'train')
process(test_data, 'test')


# train_data[1][0][9]
# para = train_data

# CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME = 63, 10, 'bert-base-chinese'

# i = 5
# d, l = train_data[0][i], [train_data[1][0][i], train_data[1][1][i]]
# tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
# tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
# d, tokenizer, cnt, hard, properties, ans_pos = tokenizer.tokenize(d), tokenizer, 0, False, None, l # tokenizer會提示文本太長，那個警告是他發的

# # d is only a list of tokens, not split. 
# # properties are also a list of tuples.
# end_tokens = {'\n':0, '.':1, '?':1, '!':1, ',':2, '。':1, '？':1, '！':1, '，':2}
# # for k, v in list(end_tokens.items()):
# #     end_tokens['Ġ' + k] = v
# sen_cost, break_cost = 4, 8
# # 在第幾個token出現了標點，以及標點的種類
# poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
# poses.insert(0, (-1, 0))
# if poses[-1][0] < len(d) - 1:
#     poses.append((len(d) - 1, 0))
# x = 0
# # Futher adjust (確認都能小於block_size)
# while x < len(poses) - 1:
#     if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
#         poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
#     x += 1
# # simple dynamic programming
# best = [(0, 0)]
# for i, (p, cost) in enumerate(poses):
#     if i == 0:
#         continue    
#     best.append((-1, 100000))
#     for j in range(i-1, -1, -1):
#         if p - poses[j][0] > BLOCK_SIZE:
#             break
#         value = best[j][1] + cost + sen_cost
#         if value < best[i][1]:
#             best[i] = (j, value)
#     assert best[i][0] >= 0
# intervals, x = [], len(poses) - 1
# while x > 0:
#     l = poses[best[x][0]][0 ]
#     intervals.append((l + 1, poses[x][0] + 1))
#     x = best[x][0]
# if properties is None:
#     properties = []

# for st, en in reversed(intervals):
#     # copy from hard version
#     blk_chose = 0
#     if ans_pos is not None :
#       for p_a in ans_pos [0]:
#         iou_tmp = textIoU([st, en-1], p_a)
#         if(iou_tmp > 0.7) :
#            blk_chose = 1
#            print("got_answer_block")
#     cnt += 1
#     tmp = d[st: en] + [tokenizer.sep_token]
#     # inject properties into blks
#     tmp_kwargs = {}
#     for p in properties:
#         if len(p) == 2:
#             # p[0] : ('name', label_name), p[1] : ('label', label)
#             tmp_kwargs[p[0]] = p[1]
#         elif len(p) == 3:
#             if st <= p[1] < en:
#                 tmp_kwargs[p[0]] = (p[1] - st, p[2])
#         else:
#             raise ValueError('Invalid property {}'.format(p))
        
#     Block(tokenizer.convert_tokens_to_ids(tmp), cnt, choose = blk_chose, **tmp_kwargs)
#     # def __init__(self, ids, pos, blk_type=1, choose = 0, **kwargs):
#     # ret.insert()