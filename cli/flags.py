import argparse as _argparse
import collections


def dict2Namespace(dic, root=False):
    normal = dict({})
    sub = collections.defaultdict(dict)
    for k in dic:
        if "." in k:
            namesplit = k.split(".")
            sub[namesplit[0]][".".join(namesplit[1:])] = dic[k]
        else:
            normal[k] = dic[k]
    for k in sub:
        tnamespace = dict2Namespace(sub[k])
        normal[k] = tnamespace
    if root is False:
        return _argparse.Namespace(**normal)
    else:
        return normal


def namespace2Dict(namespace):
    new_dict = dict({})
    if isinstance(namespace, _argparse.Namespace):
        namespace = vars(namespace)
    for k in namespace:
        v = namespace[k]
        if isinstance(v, _argparse.Namespace) is True:
            cdict = namespace2Dict(v)
            for ck in cdict:
                new_dict[k + "." + ck] = cdict[ck]
        else:
            new_dict[k] = v
    return new_dict


class _FlagValues(object):
    """Global container and accessor for flags and their values."""

    def __init__(self):
        self.__dict__["__flags"] = {}
        self.__dict__["__actions"] = {}
        self.__dict__["__parsed"] = False
        self.__dict__["__namespace"] = False

    def _parse_flags(self, args=None):
        result = _global_parser.parse_args(args=args)
        for flag_name, val in vars(result).items():
            self.__dict__["__flags"][flag_name] = val
        self.__dict__["__parsed"] = True
        # return unparsed

    def get_dict(self):
        if not self.__dict__["__parsed"]:
            self._parse_flags()

        if self.__dict__["__namespace"] is False:
            return self.__dict__["__flags"]
        else:
            return namespace2Dict(self.__dict__["__flags"])

    def set_dict(self, newdict, overwrite=False):
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        ignore = ["gpu", "results_folder", "subfolder", "old_model"]
        mydict = namespace2Dict(self.__dict__["__flags"])
        for k in newdict:
            if k in ignore:
                continue
            mydict[k] = newdict[k]
        self.__dict__["__flags"] = mydict
        self.__dict__["__namespace"] = False
        self.toNameSpace()

    def toNameSpace(self):
        if self.__dict__["__namespace"] is True:
            return
        self.__dict__["__flags"] = dict2Namespace(self.__dict__["__flags"], root=True)
        self.__dict__["__namespace"] = True

    def __hasattr__(self, name):
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        if name not in self.__dict__["__flags"]:
            return False
        else:
            return True

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the flag --name."""
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        if name not in self.__dict__["__flags"]:
            raise AttributeError(name)
        return self.__dict__["__flags"][name]

    def __setattr__(self, name, value):
        """Sets the 'value' attribute of the flag --name."""
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        self.__dict__["__flags"][name] = value


SUPPRESS = _argparse.SUPPRESS
MUST_INPUT = None
_global_parser = _argparse.ArgumentParser()
FLAGS = _FlagValues()


def define_argument(k, defv):
    if type(defv) == bool:
        DEFINE_boolean("-" + k, "--" + k, default=_argparse.SUPPRESS)
    else:
        DEFINE_argument("-" + k, "--" + k, default=_argparse.SUPPRESS, type=type(defv))


def DEFINE_argument(*args, default=MUST_INPUT, rep=False, **kwargs):

    if rep is True:
        kwargs["nargs"] = "+"

    _global_parser.add_argument(*args, default=default, **kwargs)


def DEFINE_boolean(*args, default=MUST_INPUT, docstring=None, **kwargs):
    """Defines a flag of type 'boolean'.

    Args:
      flag_name: The name of the flag as a string.
      default_value: The default value the flag should take as a boolean.
      docstring: A helpful message explaining the use of the flag.
    """

    # Register a custom function for 'bool' so --flag=True works.
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    docstring = "" if docstring is None else docstring
    _global_parser.add_argument(*args, help=docstring, default=default, type=str2bool, **kwargs)
