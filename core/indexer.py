import core.datacache as datacache
import core.pipeliner as pipeliner

def get(dataset, configs, pre_indexation_step):
    preindexation_uid = hasher.hash_dict(pre_indexation_step)