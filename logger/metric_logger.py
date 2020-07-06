import os
import torch
import torchvision
import pickle
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir="./logs"):
        self.stats = dict()
        self.log_dir = log_dir
        self.tb_writter = SummaryWriter(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}
        if k not in self.stats[category]:
            self.stats[category][k] = []
        self.stats[category][k].append((it, v))
        self.tb_writter.add_scalar("{}/{}".format(category, k), v, it)

    def add_imgs(self, imgs, name=None, class_name=None, vrange=None, nrow=10):
        if class_name is None:
            outdir = self.log_dir
            class_name = "ImageVisualization"
        else:
            outdir = os.path.join(self.log_dir, class_name)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if isinstance(name, str):
            outfile = os.path.join(outdir, "{}.png".format(name))
        else:
            outfile = os.path.join(outdir, "%08d.png" % name)

        if vrange is None:
            maxv, minv = float(torch.max(imgs)), float(torch.min(imgs))
        else:
            maxv, minv = max(vrange), min(vrange)
        imgs = (imgs - minv) / (maxv - minv + 1e-8)
        torchvision.utils.save_image(imgs, outfile, nrow=nrow)
        self.tb_writter.add_image(class_name, imgs, dataformats="CHW")

    def log_info(self, prefix, log_func, cats=None):
        if cats is None:
            cats = self.stats.keys()
        prefix += "\n"
        for cat in cats:
            prefix += "|{}: ".format(cat)
            for k in self.stats[cat]:
                prefix += "{}:{:.5f} ".format(k, self.stats[cat][k][-1][1])
            prefix += "\n"
        log_func(prefix)

    def save_stats(self, filename="stat.pkl"):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, "wb") as f:
            pickle.dump(self.stats, f)
