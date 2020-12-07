# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import os
import time

import pandas as pd
import numpy as np
# import sys
# sys.path.append('../')

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import ramps
from DatasetDcase2019Task4_weak2strong_NN_ex2 import DatasetDcase2019Task4_weak2strong_NN_ex2
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel_weak2strong_NN_ex2 import test_model
from evaluation_measures_weak2strong_NN_ex2 import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config_weak2strong_NN_ex2_ww2s as cfg
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, get_transforms_AANPT, get_transforms_nopad, AverageMeterSet
from utils.Logger import LOG, TIME
import shutil
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def adjust_learning_rate(optimizer, rampup_value, rampdown_value):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Custom_BCE_Loss(object):
    def __init__(self, batch_input, class_criterion):
        self.batch_input = batch_input
        self.new_weak = None
        self.length_list = []
        self.class_criterion = class_criterion
        self.new_batch_st_loss = None

    def initialize(self, batch_st_out, batch_sof):
        self.batch_st_out = batch_st_out
        self.batch_sof = batch_sof

        for ID, (sample_input, sample_st_out, sample_sof) in enumerate(zip(self.batch_input, self.batch_st_out, self.batch_sof)):
            # print(self.batch_input.shape)
            # print(self.batch_st_out.shape)
            # print(self.batch_sof.shape)

            if len(sample_input.shape) >= 3:
                sample_input = sample_input.squeeze(0)
                # print(sample_input.shape, type(sample_input))
 
            strong_length = len(sample_input)
            # print(strong_length)
            # exit()


            index = torch.where(sample_input.clone().sum(axis=1) == 0)
            if len(index[0]) != 0:
                strong_length = index[0][0]
                # print(strong_length)

            # for i in reversed(range(sample_input.shape[0])):
            #     #################################################
            #     # if (sample_input[i] ** 2).sum() != 0:
            #     if sample_input[i].sum() != 0:
            #     # if sample_input[i][0] != 0:
            #     # if len(sample_input[i][sample_input[i] != 0]) != 0:
            #         print(i+1)
            #         strong_length = i+1
            #         break
            #     #################################################

            if strong_length < len(sample_input):
                # print(strong_length)
                output_length = strong_length // cfg.pooling_time_ratio
                sample_weak = (sample_st_out * sample_sof)[:output_length].sum(axis=0) / sample_sof[:output_length].sum(axis=0)
                # print("output_length:", output_length)
                # print((sample_st_out * sample_sof)[:output_length].shape)
            else:
                output_length = strong_length // cfg.pooling_time_ratio
                sample_weak = (sample_st_out * sample_sof).sum(axis=0) / sample_sof.sum(axis=0)

            # print(strong_length)
            if self.new_weak is None:
                self.new_weak = sample_weak.unsqueeze(0)
                # print(self.new_weak.shape)

            else:
                self.new_weak = torch.cat([self.new_weak, sample_weak.unsqueeze(0)], dim=0)
                # print(self.new_weak.shape)

            self.length_list.append(output_length)
            # print(self.length_list)

        # print(type(self.new_weak[0]), self.new_weak[0])
        # print("new_weak:{}".format(self.new_weak.shape))
        # print("length_list:{}".format(len(self.length_list)))

        # print("new_weak_type:{}".format(type(self.new_weak)))
        # print("new_weak_shape:{}".format(self.new_weak.shape))


    def weak(self, target_weak, weak_mask):
        # print(type(self.class_criterion(self.new_weak[weak_mask], target_weak[weak_mask]).mean()))
        return self.class_criterion(self.new_weak[weak_mask], target_weak[weak_mask]).mean()

    def strong(self, target_strong, strong_mask):
        batch_st_loss = self.class_criterion(self.batch_st_out[strong_mask], target_strong[strong_mask])
        batch_length_list = self.length_list[strong_mask]

        # print(self.length_list)
        for sample_st_loss, sample_length in zip(batch_st_loss, batch_length_list):
            real_st_loss = sample_st_loss[:sample_length].mean()
            if self.new_batch_st_loss is None:
                self.new_batch_st_loss = real_st_loss
                counter = 1
            else:
                self.new_batch_st_loss += real_st_loss
                counter += 1

        # print("new_batch_st_loss:{}".format(self.new_batch_st_loss), counter)

        strong_loss = self.new_batch_st_loss / counter
        # print(strong_loss)
        # exit()

        return strong_loss

class Custom_BCE_Loss_difficulty(Custom_BCE_Loss):
    def __init__(self, batch_input, class_criterion, paramater=1):
        super().__init__(batch_input, class_criterion)
        self.paramater = paramater

    def weak(self, target_weak, weak_mask):
        # self.weak_loss = self.class_criterion(self.new_weak[weak_mask], target_weak[weak_mask])
        weak_loss = self.class_criterion(self.new_weak[weak_mask], target_weak[weak_mask])

        # print("weak_loss:\n", weak_loss)
        # print(self.weak_loss.shape)
        matrix = torch.where(target_weak[weak_mask] == 0, self.new_weak[weak_mask] ** self.paramater, 1-self.new_weak[weak_mask] ** self.paramater)
        # print("target:\n", target_weak[weak_mask])
        # print("weak:\n", self.new_weak[weak_mask])
        # print("matrix:\n", matrix)
        new_weak_loss = weak_loss.clone() * matrix
        # exit()

        # print("weak_loss:\n", self.weak_loss)
        # print(self.weak_loss.shape)
        # exit()

        return new_weak_loss.mean()

    def strong(self, target_strong, strong_mask):
        batch_st_loss = self.class_criterion(self.batch_st_out[strong_mask], target_strong[strong_mask])
        # print("strong_loss:\n", batch_st_loss)
        # print(self.batch_st_loss.shape)
        batch_length_list = self.length_list[strong_mask] 

        matrix = torch.where(target_strong[strong_mask] == 0, self.batch_st_out[strong_mask] ** self.paramater, 1-self.batch_st_out[strong_mask] ** self.paramater)
        # print("target:\n", target_strong[strong_mask])
        # print("weak:\n", self.batch_st_out[strong_mask])
        # print("matrix:\n", matrix)
        new_batch_st_loss = batch_st_loss.clone() * matrix
        # print("new_strong:\n", new_batch_st_loss)
        # exit()

        for sample_st_loss, sample_length in zip(new_batch_st_loss, batch_length_list):
            real_st_loss = sample_st_loss[:sample_length].mean()
            if self.new_batch_st_loss is None:
                self.new_batch_st_loss = real_st_loss
                counter = 1
            else:
                self.new_batch_st_loss += real_st_loss
                counter += 1

        strong_loss = self.new_batch_st_loss / counter
        return strong_loss


    # def cal_diff_loss(self, loss, target, mask_start, *args):
    #     for sample_id, (sample_loss, sample_target) in enumerate(zip(loss, target)):
    #         if len(loss.shape) == 1:
    #             ID = (*args, sample_id)
    #             if mask_start == None:
    #                 mask_start = 0
    #             start = mask_start + ID[0]
    #             # print(mask_start)
    #             # print(sample_id, sample_target)
    #             # print(ID)
    #             if sample_target == 0: 
    #                 if len(ID) == 2:
    #                     self.weak_loss[start][ID[1]] = self.new_weak[start][ID[1]] * sample_loss
    #                     # print("0:\n", self.new_weak[start][ID[1]], sample_loss)

    #                 if len(ID) == 3:
    #                     self.batch_st_loss[ID[0]][ID[1]][ID[2]] = self.batch_st_out[start][ID[1]][ID[2]] * sample_loss
    #                     # print("0:\n", self.batch_st_out[start][ID[1]][ID[2]], sample_loss)


    #             if sample_target == 1:
    #                 if len(ID) == 2:
    #                     self.weak_loss[start][ID[1]] = (1 - self.new_weak[start][ID[1]]) * sample_loss
    #                     # print("1:\n", self.new_weak[start][ID[1]], sample_loss)

    #                 if len(ID) == 3:
    #                     self.batch_st_loss[ID[0]][ID[1]][ID[2]] = (1 - self.batch_st_out[start][ID[1]][ID[2]]) * sample_loss
    #                     # print("1:\n", self.batch_st_out[start][ID[1]][ID[2]], sample_loss)

    #         else:
    #             self.cal_diff_loss(sample_loss, sample_target, mask_start, *args, sample_id)


          



def train(train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    class_criterion = nn.BCELoss()

    ##################################################
    class_criterion1 = nn.BCELoss(reduction='none')
    ##################################################

    consistency_criterion = nn.MSELoss()

    # [class_criterion, consistency_criterion] = to_cuda_if_available(
    #     [class_criterion, consistency_criterion])
    [class_criterion, class_criterion1, consistency_criterion] = to_cuda_if_available(
        [class_criterion, class_criterion1, consistency_criterion])

    meters = AverageMeterSet()

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(train_loader) * cfg.n_epoch // 2

    print("Train\n")
    # LOG.info("Weak[k] -> Weak[k]")
    # LOG.info("Weak[k] -> strong[k]")

    # print(weak_mask.start)
    # print(strong_mask.start)
    # exit()
    count = 0
    check_cus_weak = 0
    difficulty_loss = 0
    loss_w = 1
    LOG.info("loss paramater：{}".format(loss_w))
    for i, (batch_input, ema_batch_input, target) in enumerate(train_loader):
        # print(batch_input.shape)
        # print(ema_batch_input.shape)
        # exit()
        global_step = epoch * len(train_loader) + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        LOG.debug("batch_input:{}".format(batch_input.mean()))

        # print(batch_input)
        # exit()

        # Outputs
        ##################################################
        # strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema, weak_pred_ema, sof_ema = ema_model(ema_batch_input)
        sof_ema = sof_ema.detach()
        ##################################################

        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        ##################################################
        # strong_pred, weak_pred = model(batch_input)
        strong_pred, weak_pred, sof = model(batch_input)
        ##################################################

        ##################################################
        # custom_ema_loss = Custom_BCE_Loss(ema_batch_input, class_criterion1)

        if difficulty_loss == 0:
            LOG.info("############### Deffine Difficulty Loss ###############")
            difficulty_loss = 1
        custom_ema_loss = Custom_BCE_Loss_difficulty(ema_batch_input, class_criterion1, paramater=loss_w)
        custom_ema_loss.initialize(strong_pred_ema, sof_ema)

        # custom_loss = Custom_BCE_Loss(batch_input, class_criterion1)
        custom_loss = Custom_BCE_Loss_difficulty(batch_input, class_criterion1, paramater=loss_w)
        custom_loss.initialize(strong_pred, sof)
        ##################################################

        # print(strong_pred.shape)
        # print(strong_pred)
        # print(weak_pred.shape)
        # print(weak_pred)
        # exit()

        loss = None
        # Weak BCE Loss
        # Take the max in the time axis
        # torch.set_printoptions(threshold=10000)
        # print(target[-10])
        # # print(target.max(-2))
        # # print(target.max(-2)[0])
        # print(target.max(-1)[0][-10])
        # exit()

        target_weak = target.max(-2)[0]
        if weak_mask is not None:
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])

            print("noraml_weak:",class_criterion(weak_pred[weak_mask], target_weak[weak_mask]))

            ##################################################
            custom_weak_class_loss = custom_loss.weak(target_weak, weak_mask)
            custom_ema_class_loss = custom_ema_loss.weak(target_weak, weak_mask)
            print("custom_weak:",custom_weak_class_loss)
            ##################################################

            count += 1
            check_cus_weak += custom_weak_class_loss
            # print(custom_weak_class_loss.item())
            
            if i == 0:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
                LOG.debug(custom_weak_class_loss) ###
                LOG.debug("rampup_value: {}".format(rampup_value))
            meters.update('weak_class_loss', custom_weak_class_loss.item()) ###
            meters.update('Weak EMA loss', custom_ema_class_loss.item()) ###

            # loss = weak_class_loss
            loss = custom_weak_class_loss


            ####################################################################################
            # weak_class_loss = class_criterion(strong_pred[weak_mask], target[weak_mask])
            # ema_class_loss = class_criterion(strong_pred_ema[weak_mask], target[weak_mask])
            # # if i == 0:
            # #     LOG.debug("target: {}".format(target.mean(-2)))
            # #     LOG.debug("Target_weak: {}".format(target))
            # #     LOG.debug("Target_weak mask: {}".format(target[weak_mask]))
            # #     LOG.debug(weak_class_loss)
            # #     LOG.debug("rampup_value: {}".format(rampup_value))
            # meters.update('weak_class_loss', weak_class_loss.item())
            # meters.update('Weak EMA loss', ema_class_loss.item())

            # loss = weak_class_loss
            ####################################################################################


        # Strong BCE loss
        if strong_mask is not None:
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            # meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
            # meters.update('Strong EMA loss', strong_ema_class_loss.item())
    
            print("normal_strong:",class_criterion(strong_pred[strong_mask], target[strong_mask]))

            ##################################################
            custom_strong_class_loss = custom_loss.strong(target, strong_mask)
            meters.update('Strong loss', custom_strong_class_loss.item())

            custom_strong_ema_class_loss = custom_ema_loss.strong(target, strong_mask)
            meters.update('Strong EMA loss', custom_strong_ema_class_loss.item())
            print("custom_strong:", custom_strong_class_loss)
            ##################################################

            if loss is not None:
                # loss += strong_class_loss
                loss += custom_strong_class_loss
            else:
                # loss = strong_class_loss
                loss = custom_strong_class_loss


        # print("check_weak:", class_criterion1(weak_pred[weak_mask], target_weak[weak_mask]).mean())
        # print("check_strong:", class_criterion1(strong_pred[strong_mask], target[strong_mask]).mean())
        # print("\n")

        # exit()

        # Teacher-student consistency cost
        if ema_model is not None:

            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred,
                                                                               strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))

    print("\ncheck_cus_weak:\n", check_cus_weak / count)

    
if __name__ == '__main__':
    LOG.info("MEAN TEACHER")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("Using synthetic data = {}".format(not no_synthetic))

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic"

    store_dir = os.path.join("stored_data", "{}".format(TIME), "MeanTeacher" + add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)
    shutil.copy2(__file__, os.path.join(store_dir, "code.py"))


    pooling_time_ratio = cfg.pooling_time_ratio  # --> Be careful, it depends of the model time axis pooling
    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task4_weak2strong_NN_ex2(cfg.workspace,
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)

    weak_df = dataset.initialize_and_get_df(cfg.weak, reduced_number_of_data)

    # print(weak_df)
    # exit()
    # unlabel_df = dataset.initialize_and_get_df(cfg.unlabel, reduced_number_of_data)

    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = dataset.initialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    #############################################################################
    weak2strong_df = dataset.initialize_and_get_df(cfg.weak2strong, reduced_number_of_data, download=False)
    #############################################################################
    validation_df = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data)

    LOG.info("Select Label : {}".format(synthetic_df.loc[(synthetic_df['filename'].str.contains('.000.wav', regex=False))]['event_label'].unique()))
    # exit()
    classes = cfg.classes

    #############################################################################
    # many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)
    many_hot_encoder = ManyHotEncoder(classes, cfg.sample_rate, cfg.hop_length, cfg.pooling_time_ratio, n_frames=cfg.max_frames // pooling_time_ratio)
    #############################################################################

    #############################################################################
    # transforms = get_transforms(cfg.max_frames)
    # # Normalize時に無音部分を考慮しない
    LOG.info("Normalize時に無音部分を考慮しない")
    transforms = get_transforms_nopad()
    #############################################################################

    # Divide weak in train and valid
    train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    LOG.debug(valid_weak_df.event_labels.value_counts())

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    #############################################################################
    train_w2s_df = weak2strong_df.sample(frac=0.8, random_state=26)
    valid_w2s_df = weak2strong_df.drop(train_w2s_df.index).reset_index(drop=True)
    train_w2s_df = train_w2s_df.reset_index(drop=True)
    
    valid_w_w2s_df = pd.concat([valid_weak_df, valid_w2s_df]).reset_index(drop=True)
    #############################################################################

    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio

    # print(train_synth_df)
    # exit()
    LOG.debug(valid_synth_df.event_label.value_counts())
    LOG.debug(valid_w2s_df.event_labels.value_counts())

    # LOG.info(train_synth_df)
    # exit()

    train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)

    # unlabel_data = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
    #                           transform=transforms)

    train_synth_data = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    #############################################################################
    train_w2s_data = DataLoadDf(train_w2s_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)
    #############################################################################

    # if not no_synthetic:
    #     list_dataset = [train_weak_data, unlabel_data, train_synth_data]
    #     batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
    #     strong_mask = slice(cfg.batch_size//4 + cfg.batch_size//2, cfg.batch_size)

    # if not no_synthetic:
    #     list_dataset = [train_weak_data, train_synth_data, train_w2s_data]

    #     batch_sizes = [6*cfg.batch_size//15, 8*cfg.batch_size//15, cfg.batch_size//15]
    #     strong_mask = slice(6*cfg.batch_size//15, cfg.batch_size)

    #     # batch_sizes = [cfg.batch_size//3, 2*cfg.batch_size//3]
    #     # strong_mask = slice(cfg.batch_size//3, cfg.batch_size)

    # else:
    #     list_dataset = [train_weak_data, unlabel_data]
    #     batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
    #     strong_mask = None
    # # Assume weak data is always the first one
    # weak_mask = slice(batch_sizes[0])

    #############################################################################
    if not no_synthetic:
        list_dataset = [train_weak_data, train_w2s_data, train_synth_data]

        batch_sizes = [6*cfg.batch_size//15, 2*cfg.batch_size//15, 7*cfg.batch_size//15]
        strong_mask = slice(6*cfg.batch_size//15 + 2*cfg.batch_size//15, cfg.batch_size)

        # batch_sizes = [cfg.batch_size//3, 2*cfg.batch_size//3]
        # strong_mask = slice(cfg.batch_size//3, cfg.batch_size)

    weak_mask = slice(batch_sizes[0] + batch_sizes[1])
    #############################################################################


    scaler = Scaler()
    scaler.calculate_scaler(ConcatDataset(list_dataset))

    LOG.debug(scaler.mean_)
    # print(train_weak_data.filenames)
    # exit()

    #############################################################################
    # transforms = get_transforms(cfg.max_frames, scaler, augment_type="noise")
    LOG.info("Change Normalize(Zero-padding)")
    transforms = get_transforms_AANPT(cfg.max_frames, scaler, augment_type="noise")
    #############################################################################

    for i in range(len(list_dataset)):
        list_dataset[i].set_transform(transforms)

    # torch.set_printoptions(profile="full")
    # print(list_dataset[0].filenames)
    # for i in range(len(list_dataset[0].filenames)):
    #     if list_dataset[0].filenames.iloc[i].split('.')[-2] != "000":
    #         print(list_dataset[0].filenames.iloc[i])
    #         print(list_dataset[0][i][0][0], len(list_dataset[0][i][2]))
    #         exit()
    # print(list_dataset[0][0]),len(concat_dataset[100][1]), len(concat_dataset[100][2])

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes)

    # len(sampler)
    # exit()

    training_data = DataLoader(concat_dataset, batch_sampler=sampler)
    # print(len(concat_dataset[100][0]),len(concat_dataset[100][1]), len(concat_dataset[100][2]))
    # exit()

    #############################################################################
    # transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    transforms_valid = get_transforms_AANPT(cfg.max_frames, scaler=scaler)
    #############################################################################

    valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)
    valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                 transform=transforms_valid)
    #############################################################################
    valid_w_w2s_data = DataLoadDf(valid_w_w2s_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms_valid)
    # print(valid_w_w2s_df)
    # exit()
    #############################################################################

    # Eval 2018
    eval_2018_df = dataset.initialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)

    # ##############
    # Model
    # ##############
    crnn_kwargs = cfg.crnn_kwargs
    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)
    LOG.info(crnn)

    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    state = {
        'model': {"name": crnn.__class__.__name__,
                  'args': '',
                  "kwargs": crnn_kwargs,
                  'state_dict': crnn.state_dict()},
        'model_ema': {"name": crnn_ema.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn_ema.state_dict()},
        'optimizer': {"name": optimizer.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": pooling_time_ratio,
        "scaler": scaler.state_dict(),
        "many_hot_encoder": many_hot_encoder.state_dict()
    }

    save_best_cb = SaveBest("sup")


    writer = SummaryWriter(log_dir="./loss_logs/{}".format(os.path.join("{}".format(TIME), "MeanTeacher" + add_dir_model_name)))


    # ##############
    # Train
    # ##############
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        [crnn, crnn_ema] = to_cuda_if_available([crnn, crnn_ema])

        train(training_data, crnn, optimizer, epoch, ema_model=crnn_ema, weak_mask=weak_mask, strong_mask=strong_mask)

        crnn = crnn.eval()
        LOG.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      save_predictions=None)
        valid_events_metric = compute_strong_metrics(predictions, valid_synth_df)

        #############################################################################
        LOG.info("\n ### Valid weak metric ### \n")
        weak_metric = get_f_measure_by_class(crnn, len(classes),
                                             DataLoader(valid_w_w2s_data, batch_size=cfg.batch_size))
        #############################################################################

        LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if not no_synthetic:
                global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

                ########################################################################
                writer.add_scalar("macro_F1score_strong", global_valid * 100, epoch)
                writer.add_scalar("macro_F1score_weak", np.mean(weak_metric) * 100, epoch)
                ########################################################################

                global_valid = global_valid + np.mean(weak_metric)
            else:
                global_valid = np.mean(weak_metric)
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))
    else:
        LOG.info("testing model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")
    test_model(state, cfg.validation, reduced_number_of_data, predicitons_fname)

    # ##############
    # Evaluation
    # ##############
    predicitons_eval2019_fname = os.path.join(saved_pred_dir, "baseline_eval2019.tsv")
    test_model(state, cfg.eval_desed, reduced_number_of_data, predicitons_eval2019_fname)
