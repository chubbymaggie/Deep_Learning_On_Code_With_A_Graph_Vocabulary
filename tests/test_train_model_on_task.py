# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import shutil
import unittest

from data import FITBTask, VarNamingTask
from experiments.utils import get_time
from models import FITBFixedVocabGGNN, FITBFixedVocabDTNN, FITBFixedVocabRGCN, FITBFixedVocabGAT, VarNamingFixedVocabGGNN, \
    VarNamingCharCNNGGNN, VarNamingNameGraphVocabGGNN
from models.FITB.CharCNN import FITBCharCNN
from models.FITB.FixedVocab import FITBFixedVocab
from models.FITB.NameGraphVocab import FITBNameGraphVocab
from models.VarNaming.FixedVocab import VarNamingFixedVocab
from preprocess_task_for_model import preprocess_task_for_model
from tests import test_s3shared_path
from train_model_on_task import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestTrainModelOnFITBTask(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = FITBTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'FITBTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_with_FITBFixedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBFixedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='FITBFixedVocabGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(FITBFixedVocab.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=256,
              debug=True)

    def test_train_model_on_task_with_FITBCharCNNGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBCharCNNGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='FITBCharCNNGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(FITBCharCNN.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=256,
              debug=True)

    def test_train_model_on_task_with_FITBNameGraphVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBNameGraphVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='FITBNameGraphVocabGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(FITBNameGraphVocab.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=256,
              debug=True)


class TestTrainModelOnVarNamingTask(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'VarNamingTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_with_VarNamingFixedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingFixedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='VarNamingFixedVocabGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(VarNamingFixedVocab.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3,
                                max_name_length=8),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='VarNamingLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_full_name_accuracy',),
              n_batch=256,
              debug=True)


class TestTrainModelOnFITBTaskMemorizeMinibatch(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_minibatch_memorize_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        self.n_graphs_for_minibatch = 5
        self.minibatch_size = 20
        for file in os.listdir(self.gml_dir)[:self.n_graphs_for_minibatch]:
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = FITBTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'FITBTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_memorize_minibatch_with_FITBFixedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBFixedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBFixedVocabGGNN',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBFixedVocabGGNN.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=3),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .001},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=7,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_with_FITBFixedVocabDTNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBFixedVocabDTNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBFixedVocabDTNN',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBFixedVocabDTNN.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=3),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .003},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=7,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_with_FITBFixedVocabRGCN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBFixedVocabRGCN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBFixedVocabRGCN',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBFixedVocabRGCN.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=3),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .003},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=7,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_with_FITBFixedVocabGAT(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBFixedVocabGAT',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBFixedVocabGAT',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBFixedVocabGAT.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              n_multi_attention_heads=2,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=2),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .00075},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=8,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.7)


class TestTrainModelOnVarNamingTaskMemorizeMinibatch(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_minibatch_memorize_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        self.n_graphs_for_minibatch = 5
        self.minibatch_size = 20
        for file in os.listdir(self.gml_dir)[:self.n_graphs_for_minibatch]:
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'VarNamingTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_memorize_minibatch_with_VarNamingFixedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingFixedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingFixedVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingFixedVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .005},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=7,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_with_VarNamingCharCNNGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingCharCNNGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=30),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingCharCNNGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingCharCNNGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .0005},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=5,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.9)

    def test_train_model_on_task_memorize_minibatch_with_VarNamingNameGraphVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingNameGraphVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=30),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingNameGraphVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingNameGraphVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingGraphVocabLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .005},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=7,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_no_subtoken_edges_with_VarNamingNameGraphVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingNameGraphVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=30,
                                                           add_edges=False),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingNameGraphVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingNameGraphVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingGraphVocabLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .0005},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=15,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_edit_distance_with_VarNamingNameGraphVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingNameGraphVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingNameGraphVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingNameGraphVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingGraphVocabLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .0005},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=9,
                                     evaluation_metrics=('evaluate_edit_distance',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertLessEqual(wordwise_accuracy, 2)
