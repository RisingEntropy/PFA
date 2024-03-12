import datetime
import os.path
from torch.cuda import amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim import SGD

import VGG
from Exception import TrianerNotLoaded
from spikingjelly.clock_driven import functional


class Trainer:
    def __init__(self, name: str, net, train_data, validate_data, test_data, criterion: str = "MSELoss", TET=False,
                 device='cpu', optimizer: str = "Adam", lr=0.001, opt_args: tuple = (),
                 tensorboard_path: str = "./", enable_amp: bool = False, lr_scheduler_name=None):
        self.name = name
        self.net = net
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.start_epoch = 0
        self.lr = lr
        self.tensorboard_path = tensorboard_path
        self.device = device
        self.criterion_name = criterion
        self.lr_scheduler_name = lr_scheduler_name
        self.best_accuracy = 0
        self.enable_amp = enable_amp
        if self.criterion_name == "MSELoss":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.criterion_name == "TETLoss":
            self.criterion = functional.temporal_efficient_training_cross_entropy
        else:
            raise NotImplemented("unsupported loss:" + self.criterion_name)

        if optimizer == "Adam":
            self.optimizer = Adam(self.net.parameters(), lr=self.lr)
            self.optimizer_name = "Adam"
        if optimizer == 'SGD':
            self.optimizer = SGD(self.net.parameters(), lr=self.lr, *opt_args)
            self.optimizer_name = "SGD"
        self.loaded = True

    def __init_(self, name: str, net, train_data, validate_data,
                test_data,
                lr_scheduler_name=None):  # this means the trainer is configured by file rather than by parameters
        self.loaded = False
        self.net = net
        self.train_data = train_data
        self.test_data = test_data
        self.validate_data = validate_data
        self.name = name
        self.lr_scheduler_name = lr_scheduler_name
        self.best_accuracy = 0

    def save(self, file_path):  # save the trainer configuration
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.check_point_dic(), os.path.join(file_path, "net_{}_e{}_best{}_cri_{}.pth".format(self.name,
                                                                                                         self.start_epoch,
                                                                                                         self.best_accuracy,
                                                                                                         self.criterion_name)))

    def check_point_dic(self):
        check_point = {
            'name': self.name,
            'net_parameters': self.net.state_dict(),
            'lr': self.lr,
            'optimizer_name': self.optimizer_name,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.start_epoch,
            'tensorboard_path': self.tensorboard_path,
            'criterion_name': self.criterion_name,
            'device': self.device,
            'best_accuracy': self.best_accuracy,
            'enable_amp': self.enable_amp
        }

        return check_point

    def load(self, file_path: str):  # load configuration from file
        check_point = torch.load(file_path, map_location='cpu')
        self.name = check_point['name']
        self.lr = check_point['lr']
        self.start_epoch = check_point['epoch']
        self.net.load_state_dict(check_point['net_parameters'])
        self.tensorboard_path = self.tensorboard_path
        self.device = check_point['device']
        self.best_accuracy = check_point['best_accuracy']
        self.enable_amp = check_point['enable_amp']
        if check_point['optimizer_name'] == 'Adam':
            self.optimizer = Adam(self.net.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(check_point['optimizer'])
        elif check_point['optimizer_name'] == 'SGD':
            self.optimizer = SGD(self.net.parameters, lr=self.lr)
            self.optimizer.load_state_dict(check_point['optimizer'])

        if self.criterion_name == "MSELoss":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.criterion_name == "TETLoss":
            self.criterion = functional.temporal_efficient_training_cross_entropy
        else:
            raise NotImplemented("unsupported loss:" + self.criterion_name)

        self.loaded = True

    def trian(self, total_epoch: int, categories, log_report, save_path: str = '', save_step: int = 100,
              lr_scheuler_arg={}, intermediate_reporter=None, final_reporter=None):

        if not self.loaded:
            raise TrianerNotLoaded()

        scaler = None
        if self.enable_amp:
            scaler = amp.GradScaler()

        if self.lr_scheduler_name == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=30)
        elif self.lr_scheduler_name == 'CosALR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, *lr_scheuler_arg)
        else:
            lr_scheduler = None

        max_test_acc = 0
        writer = SummaryWriter(self.tensorboard_path, purge_step=self.start_epoch)
        log_report("start to train...")

        for epoch in range(self.start_epoch + 1, total_epoch + 1):
            self.start_epoch = epoch
            log_report("epoch{} starts at:{}".format(epoch, datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))

            train_loss = 0
            train_acc = 0
            train_samples = 0
            if epoch % save_step == 0:
                self.save(os.path.join(save_path, "routine_save"))
            self.net.train()
            for frame, label in self.train_data:
                self.optimizer.zero_grad()
                frame = frame.float().to(self.device)
                label = label.to(self.device)
                label_onehot = F.one_hot(label, categories).float()
                # start training
                if self.enable_amp:
                    with amp.autocast():
                        out = self.net(frame)
                        if self.criterion_name != 'CrossEntropyLoss':
                            loss = self.criterion(out, label_onehot)
                        else:
                            loss = self.criterion(out, label)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    functional.reset_net(self.net)
                else:
                    out = self.net(frame)
                    if self.criterion_name != 'CrossEntropyLoss':
                        loss = self.criterion(out, label_onehot)
                    else:
                        loss = self.criterion(out, label)
                    loss.backward()
                    self.optimizer.step()
                    functional.reset_net(self.net)

                # review training
                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()
                # snn has memory, we have to reset it here after every training

            if lr_scheduler is not None:
                lr_scheduler.step()
            train_loss /= train_samples
            train_acc /= train_samples
            log_report("train_loss:{}   train_acc{}".format(train_loss, train_acc))
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)

            # start to validate
            self.net.eval()

            test_loss = 0
            test_acc = 0
            test_samples = 0
            with torch.no_grad():  # do not use grad
                for frame, label in self.validate_data:
                    frame = frame.float().to(self.device)
                    label = label.to(self.device)
                    label_onehot = F.one_hot(label, categories).float()
                    out = self.net(frame)
                    if self.criterion_name != 'CrossEntropyLoss':
                        loss = self.criterion(out, label_onehot)
                    else:
                        loss = self.criterion(out, label)

                    test_samples += label.numel()
                    test_acc += (out.argmax(1) == label).float().sum().item()
                    test_loss += loss.item() * label.numel()
                    functional.reset_net(self.net)

            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            log_report("test_loss:{} test_acc:{}".format(test_loss, test_acc))
            intermediate_reporter(
                {'default': test_acc, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
                 'test_acc': test_acc})
            if test_acc > max_test_acc and save_path != '':
                max_test_acc = test_acc
                self.best_accuracy = max_test_acc
                self.save(os.path.join(save_path, "bests"))

            log_report("epoch end at{}".format(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))
        final_reporter({'default': self.best_accuracy})
        return max_test_acc

    def test_single(self, data):
        with torch.no_grad():
            out = self.net(data)
            return out.argmx(1)

    def test_batch(self, categories):
        self.net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():  # do not use grad
            for frame, label in self.validate_data:
                frame = frame.float().to(self.device)
                label = label.to(self.device)
                label_onehot = F.one_hot(label, categories).float()
                out = self.net(frame)
                loss = self.criterion(out, label_onehot)
                test_samples += label.numel()
                test_acc += (out.argmax(1) == label).float().sum().item()
                test_loss += loss.item() * label.numel()
                functional.reset_net(self.net)
        test_loss /= test_samples
        test_acc /= test_samples
        print("test_loss:{} test_acc:{}".format(test_loss, test_acc))
        return (test_loss, test_acc)
