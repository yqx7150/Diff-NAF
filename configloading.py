import yaml

def load_config(path, default_path=None):
    with open(path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):  
            update_recursive(dict1[k], v) 
        else:
            dict1[k] = v     