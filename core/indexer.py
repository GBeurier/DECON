import core.datacache as datacache
import core.pipeliner as pipeliner
import core.hasher as hasher

def get(dataset, configs, pre_indexation_step):
    preindexation_uid = hasher.hash_dict(pre_indexation_step)