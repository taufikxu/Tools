import logging
import operator
import os
import pickle
import shutil
import socket
import time
import glob

import coloredlogs

from Tools.cli import flags
from Tools.cli.config import notValid

FLAGS = flags.FLAGS


def get_logger(logger_name=None):
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.propagate = 0
    else:
        logger = logging.getLogger("taufikxu")
    return logger


def build_logger(file_names=None, logger_name=None):
    FORMAT = "%(asctime)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT)
    logger = get_logger(logger_name)

    if isinstance(file_names, str):
        file_names = [file_names]

    if file_names is not None:
        for filename in file_names:
            fh = logging.FileHandler(filename=filename)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s;%(levelname)s|%(message)s", "%H:%M:%S"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(
        level=logging.INFO, fmt=FORMAT, datefmt=DATEF, level_styles=LEVEL_STYLES
    )

    def get_list_name(obj):
        if type(obj) is list:
            for i in range(len(obj)):
                if callable(obj[i]):
                    obj[i] = obj[i].__name__
        elif callable(obj):
            obj = obj.__name__
        return obj

    sorted_list = sorted(FLAGS.get_dict().items(), key=operator.itemgetter(0))
    host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
    logger.info("#" * 120)
    logger.info("----------Configurable Parameters In this Model----------")
    logger.info(host_info)
    for name, val in sorted_list:
        logger.info("# " + ("%30s" % name) + ":\t" + str(get_list_name(val)))
    logger.info("#" * 120)
    return logger


def save_context(filename, keys):
    filename = os.path.splitext(filename)[0]
    project = os.path.basename(os.getcwd())
    experiment_name = ""
    logfiles = []
    FILES_TO_BE_SAVED = ["./configs", "./library"]
    KEY_ARGUMENTS = keys

    if FLAGS.gpu.lower() not in ["-1", "none", notValid.lower()]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configs_dict = FLAGS.get_dict()

    default_key = ""
    for item in KEY_ARGUMENTS:
        default_key += (
            "(" + item + "_" + str(configs_dict[item]).replace("/", "_")[:20] + ")"
        )

    if FLAGS.results_folder == notValid:
        FLAGS.results_folder = "./Aresults/"
    if FLAGS.subfolder != notValid:
        FLAGS.results_folder = os.path.join(FLAGS.results_folder, FLAGS.subfolder)

    experiment_name = "({file})_({data})_({time})_({default_key})_({user_key})".format(
        file=filename.replace("/", "_"),
        data=FLAGS.dataset,
        time=time.strftime("%Y-%m-%d-%H-%M-%S"),
        default_key=default_key,
        user_key=FLAGS.key,
    )
    FLAGS.results_folder = os.path.join(FLAGS.results_folder, experiment_name)
    logfiles.append(os.path.join(FLAGS.results_folder, "Log.txt"))

    if os.path.isabs(FLAGS.results_folder):
        experiment_name = "({})".format(project) + experiment_name
        FLAGS.results_folder = "({})".format(project) + FLAGS.results_folder
        logfiles.append("./Aresults/{}_log.txt".format(experiment_name))

    if os.path.exists(FLAGS.results_folder):
        raise FileExistsError(
            "{} exits. Run it after a second.".format(FLAGS.results_folder)
        )

    MODELS_FOLDER = FLAGS.results_folder + "/models/"
    SUMMARIES_FOLDER = FLAGS.results_folder + "/summary/"
    SOURCE_FOLDER = FLAGS.results_folder + "/source/"

    # creating result directories
    os.makedirs(FLAGS.results_folder)
    os.makedirs(MODELS_FOLDER)
    os.makedirs(SUMMARIES_FOLDER)
    os.makedirs(SOURCE_FOLDER)
    logger = build_logger(logfiles)

    destination = SOURCE_FOLDER
    for f in glob.glob("./*.py"):
        shutil.copy(f, os.path.join(destination, f))
    for f in FILES_TO_BE_SAVED:
        shutil.copytree(f, os.path.join(destination, f))

    configs_dict = FLAGS.get_dict()
    with open(os.path.join(SOURCE_FOLDER, "configs_dict.pkl"), "wb") as f:
        pickle.dump(configs_dict, f)
    return logger, MODELS_FOLDER, SUMMARIES_FOLDER
