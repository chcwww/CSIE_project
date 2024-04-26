import os
import sys
import pickle
import pandas as pd
from tqdm import trange
from transformers import AutoTokenizer

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)
print(f"Dir now : {os.getcwd()}")

from scripts.buffer import Buffer
from utils.util import DEFAULT_MODEL_NAME, DATA_NAME

def pdftext_to_para_v2(text_data) :
  paragraph = []
  strong_label = []
  weak_label = []

  test_para = []
  test_label = []

  p = []
  s_q = []
  w_q = []
  count = 0
  
  iter_data = zip(text_data.weak_choose, text_data.strong_choose, text_data.paragraph)

  for weak, strong, text in iter_data :
    if(len(text) > 5 and text[0:5] == "-----") : # 遇到分隔
      count += 1
      if count % 10 == 0 :
        test_para.append(p)
        test_label.append(s_q) # 測試集只給strong_label
      else :
        paragraph.append(p)
        strong_label.append(s_q)
        weak_label.append(w_q)
      p = []
      s_q = []
      w_q = []
    else :
      p.append([text, int(strong)]) # [block, choose]
      s_q.append(int(strong))
      w_q.append(int(weak))
  assert len(test_para) == len(test_label)
  assert len(paragraph) == len(strong_label) and len(paragraph) == len(weak_label)
  # [ [train_p, strong, weak], [test_p, test_strong] ]
  return [{'para' : paragraph, 'strong' : strong_label, 'weak' : weak_label}, 
          {'para' : test_para, 'strong' : test_label, 'weak' : None}]

def process(para, dataset_name, label_type = 'strong'):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    cnt, batches = 0, []
    
    for i in trange(len(para['para'])):
        # [ 內文([block, choose]), 標籤 ]
        d, l = para['para'][i], para[label_type][i]  

        # 把這輪輸進來的資料轉成Buffer (cnt是總共有幾塊)
        dbuf, cnt, qbuf = Buffer.split_version2(d, tokenizer, cnt, label = l)
        # qbuf是標籤內容 dbuf是正文
        batches.append((qbuf, dbuf))
    with open(os.path.join(root_dir, 'data', f'{DATA_NAME}_{label_type}_{dataset_name}.pkl'), 'wb') as fout: 
        pickle.dump(batches, fout) # 將batches的內容保存到fout中
    return batches

if __name__ == "__main__" :
  text_data = pd.read_csv(r"C:\Users\chcww\Downloads\社會期刊訓練資料集_threshold0.6.csv")
  train_data, test_data = pdftext_to_para_v2(text_data)

  process(train_data, 'train', 'strong')
  process(train_data, 'train', 'weak')
  process(test_data, 'test')