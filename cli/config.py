import glob
import yaml

from Tools.cli import flags

FLAGS = flags.FLAGS
notValid = "NotApply"
default_arguments = {
    "config_file": notValid,
    "method": notValid,
    "seed_bias": 0,
    "gpu": notValid,
    "gpu_number": notValid,
    "key": notValid,
    "dataset": notValid,
    "results_folder": notValid,
    "subfolder": notValid,
    "old_model": notValid,
}


def flatten_dicts(tdict):
    new_dict = dict({})
    for k in tdict:
        v = tdict[k]
        if isinstance(v, dict) is True:
            cdict = flatten_dicts(v)
            for ck in cdict:
                new_dict[k + "." + ck] = cdict[ck]
        else:
            new_dict[k] = v
    return new_dict


def parse_dict_cfg(cfg):
    shared_dict = dict({})
    for k in cfg:
        if k.startswith("include"):
            config_path = cfg[k]
            with open(config_path, "r") as f:
                cfg_special = yaml.load(f, Loader=yaml.FullLoader)
                cfg_special = flatten_dicts(cfg_special)
            print("config_file", config_path, cfg_special)
            for tk in cfg_special:
                assert tk not in default_arguments
                assert tk not in shared_dict
                shared_dict[tk] = cfg_special[tk]

    for k in shared_dict:
        if k not in cfg:
            cfg[k] = shared_dict[k]
        else:
            print("ignore include files", k, "use cfg value", cfg[k])
    return cfg


def load_config(config_path):
    """ Loads config file.
    Args:
        config_path (str): path to config file
    """
    # Load configuration from file itself
    with open(config_path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)
        cfg_special = flatten_dicts(cfg_special)
        cfg_special = parse_dict_cfg(cfg_special)

    ignore_arguments = default_arguments.keys()
    input_arguments = []
    all_keys = FLAGS.get_dict()
    # Include main configuration
    for k in all_keys:
        if k in ignore_arguments:
            continue
        input_arguments.append(k)
        print("Note!: input args: {} with value {}".format(k, FLAGS.__getattr__(k)))

    all_keys = parse_dict_cfg(all_keys)
    for k in cfg_special:
        if k in all_keys and all_keys[k] != notValid:
            print("Ignore {}".format(k))
        else:
            FLAGS.__setattr__(k, cfg_special[k])

    FLAGS.toNameSpace()
    return input_arguments


def init_cli():
    existed_args = list(default_arguments.keys())
    flags.DEFINE_argument(
        "config_file", type=str, help="Path to config file.",
    )
    for k in default_arguments:
        if k != "config_file":
            flags.DEFINE_argument("-" + k, "--" + k, type=type(k), default=default_arguments[k])

    others_yml = glob.glob("./configs/*.yml") + glob.glob("./configs/*.yaml")
    others_yml_inc = glob.glob("./configs/*/*.yml") + glob.glob("./configs/*/*.yaml")
    others_yml += others_yml_inc
    for yml in others_yml:
        with open(yml, "r") as f:
            newdict = yaml.load(f, Loader=yaml.FullLoader)
            newdict = flatten_dicts(newdict)
        for k in newdict:
            if k in existed_args:
                continue
            existed_args.append(k)
            flags.define_argument(k, newdict[k])


if __name__ == "__main__":
    with open("./configs/debug.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    print(flatten_dicts(cfg))
    load_config("./configs/debug.yaml")
    print(FLAGS.GenDis.activation.normal)
