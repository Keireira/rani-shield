"""The integrated inference path uses explicit constructors, never dynamic imports."""


def instantiate_from_config(config):
    raise ValueError("Dynamic DiffSwap training configurations are not supported.")
