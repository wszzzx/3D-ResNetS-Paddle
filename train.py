import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from model import resnet_3d
from reader import Ucf101
from config import parse_config, merge_configs, print_configs
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='resnet_3d',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ucf101.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        print_configs(train_config, 'Train')
        # train_model = TSN1.TSNResNet('TSN',train_config['MODEL']['num_layers'],
        #                             train_config['MODEL']['num_classes'],
        #                             train_config['MODEL']['seg_num'],0.00002)
        train_model = resnet_3d.generate_model(train_config['MODEL']['num_layers'])
        # 根据自己定义的网络，声明train_model
        # opt = fluid.optimizer.Momentum(learning_rate=train_config['MODEL']['learning_rate'],momentum = 0.9, parameter_list=train_model.parameters())
        # opt = fluid.optimizer.Momentum(0.001, 0.9, parameter_list=train_model.parameters())
        # opt=fluid.optimizer.SGDOptimizer(learning_rate=train_config['MODEL']['learning_rate'], parameter_list=train_model.parameters())
        opt = fluid.optimizer.AdamOptimizer(learning_rate=train_config['MODEL']['learning_rate'],
                                            parameter_list=train_model.parameters())
        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            train_model = resnet_3d.generate_model(train_config['MODEL']['num_layers'], n_classes=1039)
            # model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/tsn_model')
            model, _ = fluid.dygraph.load_dygraph('data/data51645/paddle_dy')

            train_model.load_dict(model)
            train_model.fc = fluid.dygraph.Linear(512 * 4, 101, act='softmax')
            print('pretrain is ok')

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        # train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()
        train_reader = Ucf101(args.model_name.upper(), 'train', train_config).create_reader()

        epochs = args.epoch or train_model.epoch_num()

        # test
        test_config = merge_configs(config, 'test', vars(args))
        label_dic = np.load('label_dir.npy', allow_pickle=True).item()
        label_dic = {v: k for k, v in label_dic.items()}

        # get infer reader
        # test_reader = Ucf101(args.model_name.upper(), 'test', test_config).create_reader()
        t_acc = []
        v_acc = []
        t_loss = []
        for i in range(epochs):
            train_acc_list = []
            train_loss_list = []
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                dy_x_data = np.transpose(dy_x_data, (0, 2, 1, 3, 4))
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                # if batch_id ==0:
                #     print(dy_x_data.shape)
                #     print(y_data.shape)

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                # out, acc = train_model.forward(img, label)
                out, acc = train_model(img, label)
                train_acc_list.append(acc.numpy()[0])
                # print('shape',out.shape,label.shape)
                # print(out)
                # print(label)

                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                train_loss_list.append(avg_loss.numpy())

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()

                if batch_id % 10 == 0:
                    logger.info(
                        "Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
            t_loss.append(np.mean(train_loss_list))
            t_acc.append(np.mean(train_acc_list))
            # val_acc_list = []
            # for batch_id, data in enumerate(test_reader()):
            #     dy_x_data = np.array([x[0] for x in data]).astype('float32')
            #     dy_x_data = np.transpose(dy_x_data,(0,2,1,3,4))
            #     y_data = np.array([[x[1]] for x in data]).astype('int64')

            #     img = fluid.dygraph.to_variable(dy_x_data)
            #     label = fluid.dygraph.to_variable(y_data)
            #     label.stop_gradient = True
            #     out, acc = train_model.forward(img, label)
            #     val_acc_list.append(acc.numpy()[0])
            # v_acc.append(np.mean(val_acc_list))
            # print("测试集准确率为:{}".format(np.mean(val_acc_list)))
            fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/res3d_model_' + str(i + 1))

        print('t_acc', t_acc)
        print('t_loss', t_loss)
        # print('v_acc',v_acc)
        # get infer reader
        # val_reader = KineticsReader(args.model_name.upper(), 'valid', val_config).create_reader()
        # logger.info("Final loss: {}".format(avg_loss.numpy()))
        # print("Final loss: {}".format(avg_loss.numpy()))
        result_list = []
        result_list.append(t_acc)
        result_list.append(t_loss)
        np_list = np.array(result_list).T
        name = ['train_acc', 'train_loss']
        test = pd.DataFrame(columns=name, data=np_list)
        now = int(time.time())
        timeArray = time.localtime(now)
        today_time = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
        test.to_csv('train_result_' + today_time + '_.csv')


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
