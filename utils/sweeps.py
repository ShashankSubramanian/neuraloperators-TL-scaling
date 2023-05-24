from datetime import datetime
from decimal import Decimal

def format_lr(lr):
    sci = '%.0E'%lr
    return sci.replace('E-0', 'em') # assumes 1e-10 < lr < 1e0

def sweep_name_suffix(params, sweep_id):
    '''
    Return a unique name for each sweep trial in a given sweep
    Allows custom naming of sweep trials, e.g. according to trial hyperparams
    Naming scheme is chosen based on the given sweep id and trial parameters
    '''
    if sweep_id in ['jponn9sj']:
        return 'lr%s_s%d'%(format_lr(params.lr), params.subsample)
    else:
        dt = datetime.now()
        return dt.strftime("%Y%m%d-%H-%M-%S")
