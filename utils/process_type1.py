import os
import sys
import pickle
import pandas as pd
from tqdm import trange
from transformers import AutoTokenizer

# root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(root_dir)
print(f"Dir now : {os.getcwd()}")

from scripts.buffer import Buffer
from utils.util import MODEL_NAME, DATA_NAME

def pdftext_to_para_v2(text_data) :
  paragraph = []
  strong_label = []

  test_para = []
  test_label = []

  p = []
  s_q = []
  count = 0
  total_token = 0
  
  iter_data = zip(text_data.is_summary, text_data.content)

  for strong, text in iter_data :
    if len(str(text)) > 5 and text[0:5] == "-----": # 遇到分隔
      if len(p) != 0:
        count += 1
        if count % 10 == 0 :
          test_para.append(p)
          test_label.append(s_q) # 測試集只給strong_label
        else :
          paragraph.append(p)
          strong_label.append(s_q)
      p = []
      s_q = []
    else:
      total_token += len(text)
      p.append([text, int(strong)]) # [block, choose]
      s_q.append(int(strong))
  assert len(test_para) == len(test_label)
  assert len(paragraph) == len(strong_label)
  # [ [train_p, strong, weak], [test_p, test_strong] ]
  print(f'Para: {count}, Token: {total_token}, W bar: {total_token/count:.2f}')
  return [{'para' : paragraph, 'strong' : strong_label, 'weak' : None}, 
          {'para' : test_para, 'strong' : test_label, 'weak' : None}]

def process(para, dataset_name, label_type = 'strong', toy_sample = 0):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cnt, batches = 0, []
    
    trans_len = len(para['para']) if not toy_sample else 5
    output_type = label_type if not toy_sample else label_type + '_toy'
    
    # Cool Walrus Operator
    for i in (pbar := trange(trans_len, desc = dataset_name+'-'+label_type)):
      pbar.set_postfix(**{'type' : output_type})
      # [ 內文([block, choose]), 標籤 ]
      d, l = para['para'][i], para[label_type][i]  

      # 把這輪輸進來的資料轉成Buffer (cnt是總共有幾塊)
      dbuf, cnt, qbuf = Buffer.split_version2(d, tokenizer, cnt, label = l)
      # qbuf是標籤內容 dbuf是正文
      batches.append((qbuf, dbuf))
    
    with open(os.path.join('./', 'data', f'type1_data_{output_type}_{dataset_name}.pkl'), 'wb') as fout: 
        pickle.dump(batches, fout) # 將batches的內容保存到fout中
    return batches

dirName = r"C:\Users\chcww\Downloads\type1_excel"

text_data = pd.DataFrame()
temp_sep = pd.DataFrame(["-----", "-----"])
real_sep = temp_sep.T
real_sep.columns = ['content', 'is_summary']

for i, file in enumerate(os.listdir(dirName), 1):
  temp_data = pd.read_excel(os.path.join(dirName, file))
  temp_sep = real_sep.copy()
  temp_sep.iloc[0, 0] += str(i)
  temp_sep.iloc[0, 1] += str(i)
  text_data = pd.concat([text_data, temp_data, temp_sep], axis=0)


# text_data = pd.read_csv(r"C:\Users\chcww\Downloads\data1.csv")
# text_data_add = pd.read_csv(r"C:\Users\chcww\Downloads\data2.csv")

# text_data = pd.concat([text_data, text_data_add], axis=0)
train_data, test_data = pdftext_to_para_v2(text_data)

process(train_data, 'train', 'strong')
process(train_data, 'train', 'weak')
process(train_data, 'train', 'strong', toy_sample = 1)
process(train_data, 'train', 'weak', toy_sample = 1)
process(test_data, 'test')