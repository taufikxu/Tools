import os
import torch


class CheckpointIO(object):
    def __init__(self, checkpoint_dir="./chkpts", **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load_file(self, filename):
        print("Loading! {}".format(filename))
        if os.path.exists(filename):
            state_dict = torch.load(filename)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            raise FileNotFoundError

    def parse_state_dict(self, state_dict):
        """Parse state_dict of model and return scalars.
        Args:
            state_dict (dict): State dict of model
        """

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print("Warning: Could not find %s in checkpoint!" % k)
        scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict}
        return scalars
