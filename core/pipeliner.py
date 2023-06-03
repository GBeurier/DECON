import sklearn.pipeline as pipeline


def get_pipeline(config):
    return pipeline.Pipeline([])


def apply_pipeline(config, data):  # TODO: data is a dict so parse and apply for differents elements. Take into account the pipeline config and type
    return get_pipeline(config).fit_transform(data)
