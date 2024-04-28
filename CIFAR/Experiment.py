import os.path
import json

import spikingjelly
import torch.nn as nn
import torch
import numpy as np
import random
import tensorboard.summary
from prefetch_generator import BackgroundGenerator
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import VGG
import transforms
from Trainer import Trainer


class BackgroundLoader(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Argument:
    T = 10
    device = "cuda:0"
    b = 32
    lr = 0.0001
    criterion = "MSELoss"
    datasetdir = "/home/HaoYuDeng/桌面/Machine Learning/Dataset/CIFAR10DVS"
    opt = "Adam"
    amp = True
    momentum = 0.9
    lr_scheduler = "StepLR"
    resume = False
    step_size = 20
    gamma = 0.1
    T_max = 32
    j = 16
    epoch = 400
    save_step = 20
    kernel_size = 3
    rank = 64
    tet = False

    def __init__(self):
        self.parameter_json = None

    def getParameter_json(self):
        if self.parameter_json is None:
            dic = {}
            dic["T"] = self.T
            dic["device"] = self.device
            dic["b"] = self.b
            dic["lr"] = self.lr
            dic["datasetdir"] = self.datasetdir
            dic["opt"] = self.opt
            dic["amp"] = self.amp
            dic["momentum"] = self.momentum
            dic["lr_scheduler"] = self.lr_scheduler
            dic["step_size"] = self.step_size
            dic["gamma"] = self.gamma
            dic["T_max"] = self.T_max
            dic["j"] = self.j
            dic["epoch"] = self.epoch
            dic["save_step"] = self.save_step
            dic["kernel_size"] = self.kernel_size
            dic["rank"] = self.rank
            dic["tet"] = self.tet
            return json.dumps(dic)
        else:
            return self.parameter_json

    def praseJSON(self, path: str):
        self.parameter_json = open(path, 'r').read()
        arg = json.loads(self.parameter_json)
        if arg.has_key("T"):
            self.T = int(arg['T'])

        if arg.has_key("device"):
            self.device = str(arg["device"])

        if arg.has_key("b"):
            self.b = int(arg["b"])

        if arg.has_key("lr"):
            self.lr = float(arg["lr"])

        if arg.has_key("datasetdir"):
            self.datasetdir = str(arg["datasetdir"])

        if arg.has_key("opt"):
            self.opt = str(arg["opt"])

        if arg.has_key("amp"):
            self.amp = bool(arg["b"])

        if arg.has_key("momentum"):
            self.momentum = float(arg["momentum"])

        if arg.has_key("lr_scheduler"):
            self.lr_scheduler = str(arg["lr_scheduler"])

        if arg.has_key("step_size"):
            self.step_size = int(arg["step_size"])

        if arg.has_key("gamma"):
            self.gamma = float(arg["gamma"])

        if arg.has_key("T_max"):
            self.T_max = int(arg["T_max"])

        if arg.has_key("j"):
            self.j = int(arg["j"])

        if arg.has_key('epoch'):
            self.epoch = int(arg['epoch'])

        if arg.has_key('save_step'):
            self.save_step = int(arg['save_step'])

        if arg.has_key('kernel_size'):
            self.kernel_size = int(arg['kernel_size'])

        if arg.has_key('rank'):
            self.rank = int(arg['rank'])

        if arg.has_key('tet'):
            self.tet = bool(arg['tet'])


# 实验类的基类
class Experiment:
    def __init__(self, name: str, args: Argument):
        self.name = name
        self.args = args
        if not os.path.exists(os.path.join("./", name)):
            os.makedirs(os.path.join("./", name))
        if not os.path.exists(os.path.join("./", name, "code")):
            os.makedirs(os.path.join("./", name, "code"))

        self.path = os.path.join("./", name)
        arg_file = open(os.path.join(self.path, "arguments.json"), 'w')
        arg_file.write(args.getParameter_json())
        arg_file.close()

        files = os.listdir("./")
        for file in files:
            if file[-3:] == '.py':
                with open(os.path.join("./", file), 'r') as f:
                    content = f.read()
                    with open(os.path.join("./", name, "code", file), 'w') as tosave:
                        tosave.write(content)

        Experiment.setup_Environment()

    @staticmethod
    def setup_Environment(seed=2022):
        np.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def report_intermediate_value(self, metric):
        raise NotImplemented()

    def report_final_value(self, metric):
        raise NotImplemented()


class DVSBasedExperiment(Experiment):
    def __init__(self, name: str, args: Argument):
        super(DVSBasedExperiment, self).__init__(name, args)

    def report_intermediate_value(self, metric):  # NNI 再来用
        pass

    def report_final_value(self, metric):
        pass

    def log(self, x):
        file = open(os.path.join(self.path, self.name + '_output.log'), 'a')
        file.write(x + '\n')
        print(x)
        file.close()

    def train(self, net, dataset, categories, train_set, validate_set, resume_path: str = ""):
        if train_set is None or validate_set is None:
            train_set, validate_set = \
                spikingjelly.datasets.split_to_train_test_set(0.9, dataset(self.args.datasetdir, data_type='frame',
                                                                           frames_number=self.args.T,
                                                                           split_by='number',
                                                                           transform=transforms.RandomCompose(
                                                                               select_num=1,
                                                                               const_transforms=[transforms.Flip()],
                                                                               random_transforms=(
                                                                                   transforms.Rotation(15),
                                                                                   transforms.XShear(),
                                                                                   transforms.Cutout(),
                                                                                   transforms.Rolling()), )),
                                                              categories)
        train_loader = BackgroundLoader(
            dataset=train_set,
            batch_size=self.args.b,
            shuffle=True,
            num_workers=self.args.j,
            drop_last=True,
            pin_memory=True
        )
        validate_loader = BackgroundLoader(
            dataset=validate_set,
            batch_size=self.args.b,
            shuffle=True,
            num_workers=self.args.j,
            drop_last=True,
            pin_memory=True
        )

        self.log(self.name + ' on ' + self.args.device)
        trainer = Trainer(name=self.name, net=net, train_data=train_loader,
                          validate_data=validate_loader, test_data=None, criterion=self.args.criterion,
                          device=self.args.device, optimizer=self.args.opt, lr=self.args.lr, opt_args=(),
                          tensorboard_path=os.path.join(self.path, "tensoboard"), enable_amp=self.args.amp,
                          lr_scheduler_name=self.args.lr_scheduler)
        if self.args.resume:
            trainer.load(resume_path)
        trainer.trian(self.args.epoch, categories, self.log, os.path.join(self.path, "saves"), self.args.save_step,
                      lr_scheuler_arg={'step_size': self.args.step_size},
                      intermediate_reporter=self.report_intermediate_value, final_reporter=self.report_final_value)
        self.log("training finish")
    def test_batch(self, net, categories,validate_set):
        validate_loader = BackgroundLoader(
            dataset=validate_set,
            batch_size=self.args.b,
            shuffle=True,
            num_workers=self.args.j,
            drop_last=True,
            pin_memory=True
        )
        trainer = Trainer(name=self.name, net=net, train_data=None,
                          validate_data=validate_loader, test_data=None, criterion=self.args.criterion,
                          device=self.args.device, optimizer=self.args.opt, lr=self.args.lr, opt_args=(),
                          tensorboard_path=os.path.join(self.path, "tensoboard"), enable_amp=self.args.amp)

        return trainer.test_batch(categories)