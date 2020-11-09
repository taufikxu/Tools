import os
import torch
import torchvision
import pickle
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from matplotlib import pyplot as plt

from Tools import FLAGS

matplotlib.use("agg")


class Logger(object):
    def __init__(self, log_dir="./logs", save_interval=10000):
        self.stats = dict()
        self.log_dir = log_dir
        self.save_interval = save_interval
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
        if it % self.save_interval == 0:
            self.save()

    def addvs(self, category, keyvalue, it):
        for k in keyvalue:
            v = keyvalue[k]
            self.add(category, k, v, it)
        if it % self.save_interval == 0:
            self.save()

    def __getfilename(self, name, class_name=None):
        if class_name is None:
            class_name = "ImageVisualization"
        outdir = os.path.join(self.log_dir, class_name)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if isinstance(name, str):
            outfile = os.path.join(outdir, "{}.png".format(name))
        else:
            outfile = os.path.join(outdir, "%08d.png" % name)
        return outfile

    def add_imgs(self, imgs, name, class_name=None, vrange=None, nrow=10):
        outfile = self.__getfilename(name, class_name)
        if vrange is None:
            maxv, minv = float(torch.max(imgs)), float(torch.min(imgs))
        else:
            maxv, minv = max(vrange), min(vrange)
        imgs = (imgs - minv) / (maxv - minv + 1e-8)
        torchvision.utils.save_image(imgs, outfile, nrow=nrow)
        # imgs = torchvision.utils.make_grid(imgs, nrow=nrow)
        # self.tb_writter.add_image(class_name, imgs, dataformats="CHW")

    def add_hist(self, vectors, name=None, **kwargs):
        outfile = self.__getfilename(name)
        figure = plt.figure()
        plt.hist(vectors)
        plt.savefig(outfile)

    def add_scatter(self, vectors, name=None, xyrange=None, **kwargs):
        outfile = self.__getfilename(name)
        figure = plt.figure()
        plt.scatter(vectors[:, 0], vectors[:, 1], **kwargs)
        if xyrange is not None:
            plt.xlim(-xyrange, xyrange)
            plt.ylim(-xyrange, xyrange)
        plt.savefig(outfile)

    def add_scatter_condition(self, vectors_dict, name=None, xyrange=None, **kwargs):
        outfile = self.__getfilename(name)
        figure = plt.figure()
        legend_names = []
        for k in vectors_dict:
            legend_names.append(k)
            vectors = vectors_dict[k]
            plt.scatter(vectors[:, 0], vectors[:, 1])
        if xyrange is not None:
            plt.xlim(-xyrange, xyrange)
            plt.ylim(-xyrange, xyrange)
        plt.legend(legend_names)
        plt.savefig(outfile)

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

    def save(self, filename="ModelStats.pkl"):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, "wb") as f:
            pickle.dump(self.stats, f)
