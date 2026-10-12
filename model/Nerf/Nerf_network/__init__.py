from .Lineformer import Lineformer


def get_network(type):
    if type == "Lineformer":
        return Lineformer
    else:
        raise NotImplementedError("Unknown network type!")

