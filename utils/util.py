CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
MODEL_NAME = "bert-base-chinese"
DATA_NAME = "sw_data"

BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode

import os
from pathlib import Path
SAVE_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'saved_dir'))
TMP_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'tmp_dir'))
LOG_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'log_dir'))

STRONG_SRC = Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_strong_train.pkl'))
WEAK_SRC = Path(os.path.join(os.getcwd(), 'data', 'weak',f'{DATA_NAME}_weak_train.pkl'))
TEST_SRC = Path(os.path.join(os.getcwd(), 'data', f'{DATA_NAME}_strong_test.pkl'))

# check change
def convert_caps(s): # 得到小寫
    # ret = []
    # for word in s.split():
    #     if word[0].isupper():
    #         ret.append('<pad>')
    #     ret.append(word)
    # return ' '.join(ret).lower()    
    return s.lower()

import pdb
import sys
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin