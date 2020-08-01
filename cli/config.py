import glob

import yaml

from Tools.cli import flags

FLAGS = flags.FLAGS
notValid = "NotApply"
default_arguments = {
    "config_file": notValid,
    "gpu": notValid,
    "key": notValid,
    "dataset": notValid,
    "results_folder": notValid,
    "subfolder": notValid,
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


def load_config(config_path):
    """ Loads config file.
    Args:
        config_path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(config_path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)
        cfg_special = flatten_dicts(cfg_special)

    ignore_arguments = default_arguments.keys()
    input_arguments = []
    all_keys = FLAGS.get_dict()
    # Include main configuration
    for k in all_keys:
        if k in ignore_arguments or all_keys[k] == notValid:
            continue
        input_arguments.append(k)
        print("Note!: input args: {} with value {}".format(k, FLAGS.__getattr__(k)))
    for k in cfg_special:
        if k in all_keys and all_keys[k] != notValid:
            print("Ignore {}".format(k))
        else:
            FLAGS.__setattr__(k, cfg_special[k])

    all_keys = FLAGS.get_dict()
    tobedone = dict({})
    for k in all_keys:
        if k.startswith("include"):
            print(k, k.startswith("include"))
            config_path = all_keys[k]
            with open(config_path, "r") as f:
                cfg_special = yaml.load(f, Loader=yaml.FullLoader)
                cfg_special = flatten_dicts(cfg_special)
            print(config_path, cfg_special)
            for tk in cfg_special:
                if FLAGS.__hasattr__(tk) is False or getattr(FLAGS, tk) == notValid:
                    tobedone[tk] = cfg_special[tk]
    for k in tobedone:
        FLAGS.__setattr__(k, tobedone[k])
    FLAGS.toNameSpace()
    return input_arguments


def init_cli():
    existed_args = list(default_arguments.keys())
    flags.DEFINE_argument(
        "config_file", type=str, help="Path to config file.",
    )
    for k in default_arguments:
        if k != "config_file":
            flags.DEFINE_argument(
                "-" + k, "--" + k, type=str, default=default_arguments[k]
            )

    others_yml = glob.glob("./configs/*.yml") + glob.glob("./configs/*.yaml")
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
