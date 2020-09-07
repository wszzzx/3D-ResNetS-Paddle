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
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
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
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir', type=str, default='./output', help='directory to store results')
    parser.add_argument(
        '--epoch',
        type=int,
        default=50,
        help='epoch number, 0 for read from config file')
    args = parser.parse_args()
    return args


def eval(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, "test")
    # test_config = merge_configs(config, 'train', vars(args))
    # print_configs(test_config, "train")
    with fluid.dygraph.guard():
        test_model = resnet_3d.generate_model(test_config['MODEL']['num_layers'])

        # label_dic = np.load('label_dir.npy', allow_pickle=True).item()
        # label_dic = {v: k for k, v in label_dic.items()}

        # get infer reader
        test_reader = Ucf101(args.model_name.upper(), 'test', test_config).create_reader()
        # test_reader = Ucf101(args.model_name.upper(), 'train', test_config).create_reader()
        test_acc = []
        for num in range(20, args.epoch + 1):
            # for num in range(1,3):
            weights = 'checkpoints_models/res3d_model_' + str(num)
            print("weights", weights)
            para_state_dict, _ = fluid.load_dygraph(weights)
            test_model.load_dict(para_state_dict)
            test_model.eval()

            acc_list = []
            for batch_id, data in enumerate(test_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                dy_x_data = np.transpose(dy_x_data, (0, 2, 1, 3, 4))
                y_data = np.array([[x[1]] for x in data]).astype('int64')

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = test_model(img, label)
                acc_list.append(acc.numpy()[0])
                if batch_id % 10 == 0:
                    logger.info("step {}: {}, acc: {}".format(num, batch_id, acc.numpy()))
                    print("step {}: {}, acc: {}".format(num, batch_id, acc.numpy()))
            print("测试准确率为:{}".format(np.mean(acc_list)))
            test_acc.append(np.mean(acc_list))
        print('test_acc', test_acc)
        # result_list = []
        # result_list.append(test_acc)
        name = ['test_acc']
        np_list = np.array(test_acc).T
        test = pd.DataFrame(columns=name, data=np_list)
        now = int(time.time())
        timeArray = time.localtime(now)
        today_time = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
        test.to_csv('test_result_' + today_time + '_.csv')


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    eval(args)
    print("***************************************************")
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
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
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
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir', type=str, default='./output', help='directory to store results')
    parser.add_argument(
        '--epoch',
        type=int,
        default=50,
        help='epoch number, 0 for read from config file')
    args = parser.parse_args()
    return args


def eval(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, "test")
    # test_config = merge_configs(config, 'train', vars(args))
    # print_configs(test_config, "train")
    with fluid.dygraph.guard():
        test_model = resnet_3d.generate_model(test_config['MODEL']['num_layers'])

        # label_dic = np.load('label_dir.npy', allow_pickle=True).item()
        # label_dic = {v: k for k, v in label_dic.items()}

        # get infer reader
        test_reader = Ucf101(args.model_name.upper(), 'test', test_config).create_reader()
        # test_reader = Ucf101(args.model_name.upper(), 'train', test_config).create_reader()
        test_acc = []
        for num in range(20, args.epoch + 1):
            # for num in range(1,3):
            weights = 'checkpoints_models/res3d_model_' + str(num)
            print("weights", weights)
            para_state_dict, _ = fluid.load_dygraph(weights)
            test_model.load_dict(para_state_dict)
            test_model.eval()

            acc_list = []
            for batch_id, data in enumerate(test_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                dy_x_data = np.transpose(dy_x_data, (0, 2, 1, 3, 4))
                y_data = np.array([[x[1]] for x in data]).astype('int64')

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = test_model(img, label)
                acc_list.append(acc.numpy()[0])
                if batch_id % 10 == 0:
                    logger.info("step {}: {}, acc: {}".format(num, batch_id, acc.numpy()))
                    print("step {}: {}, acc: {}".format(num, batch_id, acc.numpy()))
            print("测试准确率为:{}".format(np.mean(acc_list)))
            test_acc.append(np.mean(acc_list))
        print('test_acc', test_acc)
        # result_list = []
        # result_list.append(test_acc)
        name = ['test_acc']
        np_list = np.array(test_acc).T
        test = pd.DataFrame(columns=name, data=np_list)
        now = int(time.time())
        timeArray = time.localtime(now)
        today_time = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
        test.to_csv('test_result_' + today_time + '_.csv')


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    eval(args)
    print("***************************************************")
