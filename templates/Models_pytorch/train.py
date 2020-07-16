import torch
import numpy as np

from Tools.logger import save_context, Logger, CheckpointIO
from Tools import FLAGS, load_config

# from library import loss_gan
from library import inputs, trainer_ce
from library import utils_dataset

KEY_ARGUMENTS = ["model_name"] + KEY_ARGUMENTS
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

itr = inputs.get_dataloader(FLAGS.training.batch_size)
netC = inputs.get_model()
netC = netC.to(device)
optim, scheduler = inputs.get_optim(netC)

checkpoint_io = CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netC=netC, optim=optim)
logger = Logger(log_dir=SUMMARIES_FOLDER)

trainer_dict = {"ce": trainer_ce}
trainer_used = trainer_dict[FLAGS.loss_func]
trainner = trainer_used.Trainer(netC, optim, itr)


def prefix(it):
    total = FLAGS.training.n_iter
    percent = (it / total) * 100
    return "Itera {}/{} ({:.02f}%)".format(it, total, percent)


for it in range(FLAGS.training.n_iter):
    returnv = trainner.step()
    scheduler.step()
    logger.addvs("Training", returnv, it)

    if (it + 1) % 50 == 0:
        logger.log_info(prefix(it + 1), text_logger.info, ["Training"])

    if (it + 1) % 500 == 0:
        test_iter = inputs.get_dataloader(100, train=False, infinity=False)
        returnv, predvec = trainner.test(test_iter)
        logger.addvs("Test_real", returnv, it)
        logger.add_hist(predvec, "Hist_{:08d}_real".format(it))

    if (it + 1) % 10000 == 0:
        checkpoint_io.save("Model{:08d}.pth".format(it + 1))
        logger.save()
