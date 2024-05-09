CAPACITY = 512 # Working Memory
BIG_MODEL_NAME = "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
MODEL_NAME = "bert-base-chinese"
# BIG_MODEL_NAME = "bert-base-chinese"
DATA_NAME = "sw_data"

BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode

import os
from pathlib import Path
SAVE_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'saved_dir'))
TMP_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'tmp_dir'))
LOG_DIR = Path(os.path.join(os.getcwd(), 'load_dir', 'log_dir'))

# Ref : LibMultiLabel "https://github.com/ASUS-AICS/LibMultiLabel.git"
class AttributeDict(dict):
    """AttributeDict is an extended dict that can access
    stored items as attributes.

    >>> ad = AttributeDict({'ans': 42})
    >>> ad.ans
    >>> 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_used", set())

    def __getattr__(self, key: str) -> any:
        try:
            value = self[key]
            self._used.add(key)
            return value
        except KeyError:
            raise AttributeError(f'Missing attribute "{key}"')

    def __setattr__(self, key: str, value: any) -> None:
        self[key] = value
        self._used.discard(key)

    def used_items(self) -> dict:
        """Returns the items that have been used at least once after being set.

        Returns:
            dict: the used items.
        """
        return {k: self[k] for k in self._used}

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