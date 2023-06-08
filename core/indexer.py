from abc import ABC, abstractmethod
import logging
import core.datacache as datacache


def index(configs, pre_indexation_step, dataset_uid, dataset_name):
    indexes_list = []
    for config in configs:
        if config is None:
            continue

        name, indexer_class, config, params = config
        logging.info("Indexing %s with %s", dataset_name, name)
        indexer = indexer_class(name, config)

        preprocessed_data = None
        if indexer.is_data_dependent():
            preprocessed_data = datacache.get_data(dataset_name, dataset_uid, [], pre_indexation_step)

        indexes_list.append(indexer.indexes(preprocessed_data))

    return indexes_list


class AbstractIndexer(ABC):
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def indexes(self, data=None):
        if "test_size" in self.config:
            return self.split(data)
        elif "folds" in self.config:
            return self.cv(data)
        else:
            raise ValueError("Invalid index configuration")

    @abstractmethod
    def split(self, data=None):
        pass

    @abstractmethod
    def cv(self, data=None):
        pass

    @abstractmethod
    def is_data_dependent(self) -> bool:
        pass


class RandomSampling(AbstractIndexer):
    def __init__(self, name, config):
        super().__init__(name, config)

    def split(self, data=None):
        pass

    def cv(self, data=None):
        pass

    def is_data_dependent(self) -> bool:
        return False


class SXPY(AbstractIndexer):
    def __init__(self, name, config):
        super().__init__(name, config)

    def split(self, data=None):
        pass

    def cv(self, data=None):
        pass

    def is_data_dependent(self) -> bool:
        return True
