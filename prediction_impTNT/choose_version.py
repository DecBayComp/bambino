from constants import var_t15
from constants import var_t5
from constants import var_t2
from constants import var_t0
from constants import label_weak_strong
from constants import label_large
from constants import label_norm


def choose(ac):
    if ac == 't2':
        return var_t2
    if ac == 't5':
        return var_t5
    if ac == 't15':
        return var_t15
    if ac == 't0':
        return var_t0

def chooselabel(state):
    if state == 'weak_strong':
        return label_weak_strong
    if state == 'large':
        return label_large
    if state == 'norm':
        return label_norm
