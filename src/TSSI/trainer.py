from __future__ import annotations
from itertools import chain
from typing import Dict, List, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys
import numpy

from src.utils.mi_estimators import *

from src.TSSI.model import TSSIModel
from src.data.data_generation import demand
from src.data.data_class import TrainDataSetTorch, TestDataSetTorch, concat_dataset
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred, linear_reg_weight_loss
from config import Config

exp_num = 0

class TSSITrainer(object):

    def __init__(self, networks: List[Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False):
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        # 对于 PyTorch，设置为确定性模式
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # self.gpu_flg = False
        # configure training params
        self.lam1: float = train_params["lam1"]
        self.lam2: float = train_params["lam2"]
        self.lam3: float = train_params["lam3"]
        self.lam4: float = train_params["lam4"]
        self.distance_dim: int = train_params["distance_dim"]
        self.stage1_iter: int = train_params["stage1_iter"]
        self.stage1_S1_iter: int = train_params["stage1_S1_iter"]
        self.stage2_iter: int = train_params["stage2_iter"]
        self.covariate_iter: int = train_params["covariate_iter"]
        self.mi_iter: int = train_params["mi_iter"]
        self.odds_iter: int = train_params["odds_iter"]
        self.n_epoch: int = train_params["n_epoch"]
        self.add_stage1_intercept: bool = True
        self.add_stage2_intercept: bool = True
        self.treatment_weight_decay: float = train_params["treatment_weight_decay"]
        self.instrumental_weight_decay: float = train_params["instrumental_weight_decay"]
        self.covariate_weight_decay: float = train_params["covariate_weight_decay"]
        self.selection_weight_decay: float = train_params["selection_weight_decay"]
        self.r1_weight_decay: float = train_params["r1_weight_decay"]
        self.r0_weight_decay: float = train_params["r0_weight_decay"]
        self.s1_weight_decay: float = train_params["s1_weight_decay"]
        self.odds_weight_decay: float = train_params["odds_weight_decay"]
        self.S1_weight_decay: float = train_params["S1_weight_decay"]
        self.S0_weight_decay: float = train_params["S0_weight_decay"]
        self.y_weight_decay: float = train_params["y_weight_decay"]
        self.y1_weight_decay: float = train_params["y1_weight_decay"]
        self.lam_y: float = train_params["lam_y"]
        self.lam_a: float = train_params["lam_a"]
        self.selection_weight_decay: float = train_params["selection_weight_decay"]

        # build networks
        self.treatment_net: nn.Module = networks[0]
        self.instrumental_net: nn.Module = networks[1]
        self.selection_net: nn.Module = networks[2]
        self.covariate_net: Optional[nn.Module] = networks[3]
        self.r1_net: nn.Module = networks[4]
        self.r0_net: nn.Module = networks[5]
        self.phi_net: nn.Module = networks[6]
        self.s1_net: nn.Module = networks[7]
        self.odds_net: nn.Module = networks[8]
        self.S1_net: nn.Module = networks[9]
        self.y_net: nn.Module = networks[10]
        self.y1_net: nn.Module = networks[11]

        if self.gpu_flg:
            self.treatment_net.to("cuda:0")
            self.instrumental_net.to("cuda:0")
            if self.covariate_net is not None:
                self.covariate_net.to("cuda:0")
            self.selection_net.to("cuda:0")
            self.r1_net.to("cuda:0")
            self.r0_net.to("cuda:0")
            self.phi_net.to("cuda:0")
            self.s1_net.to("cuda:0")
            self.odds_net.to("cuda:0")
            self.S1_net.to("cuda:0")
            self.y_net.to("cuda:0")
            self.y1_net.to("cuda:0")
        
        # 创建adam优化器
        self.treatment_opt = torch.optim.Adam(self.treatment_net.parameters(),
                                              weight_decay=self.treatment_weight_decay)
        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(), 
                                                 weight_decay=self.instrumental_weight_decay)
        self.s1_opt = torch.optim.Adam(chain(self.s1_net.parameters(), self.phi_net.parameters()),
                                       weight_decay=self.s1_weight_decay)
        self.S1_opt = torch.optim.Adam(self.S1_net.parameters(),
                                       weight_decay=self.S1_weight_decay)
        self.selection_opt = torch.optim.Adam(self.selection_net.parameters(),
                                        weight_decay=self.selection_weight_decay)

        if self.covariate_net:
            self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),
                                                  weight_decay=self.covariate_weight_decay)

    def train(self, rand_seed: int = 42, verbose: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        global exp_num
        exp_num = exp_num + 1
        # 加载数据集
        # train_data, unselected_train_data, test_data, unselected_test_data = demand(Config.sample_num * 10, rand_seed)
        # train_data_predS, unselected_train_data_predS, test_data_predS, unselected_test_data_predS = demand(Config.sample_num * 10, rand_seed)

        # train_1st_t, train_2nd_t, train_3rd_t = concat_dataset(train_data,
        #                                                     unselected_train_data), train_data, unselected_train_data
        # train_1st_t = TrainDataSetTorch.from_numpy(train_1st_t)
        # train_2nd_t = TrainDataSetTorch.from_numpy(train_2nd_t)
        # train_3rd_t = TrainDataSetTorch.from_numpy(train_3rd_t)
        # test_data_t = TestDataSetTorch.from_numpy(test_data)
        # unselected_test_data_t = TestDataSetTorch.from_numpy(unselected_test_data)
    
        
        # train_1st_t_predS, train_2nd_t_predS, train_3rd_t_predS = concat_dataset(train_data_predS,
        #                                                     unselected_train_data_predS), train_data_predS, unselected_train_data_predS
        # train_1st_t_predS = TrainDataSetTorch.from_numpy(train_1st_t_predS)
        # train_2nd_t_predS = TrainDataSetTorch.from_numpy(train_2nd_t_predS)
        # train_3rd_t_predS = TrainDataSetTorch.from_numpy(train_3rd_t_predS)
        # test_data_t_predS = TestDataSetTorch.from_numpy(test_data_predS)
        # unselected_test_data_t_predS = TestDataSetTorch.from_numpy(unselected_test_data_predS)

        # if self.gpu_flg:
        #     train_1st_t = train_1st_t.to_gpu()
        #     train_2nd_t = train_2nd_t.to_gpu()
        #     train_3rd_t = train_3rd_t.to_gpu()
        #     test_data_t = test_data_t.to_gpu()
        #     train_1st_t_predS = train_1st_t_predS.to_gpu()
        #     unselected_test_data_t = unselected_test_data_t.to_gpu()
        # # 调整正则化超参数，与数据规模成比例
        # self.lam1 *= train_1st_t[0].size()[0]
        # self.lam2 *= train_2nd_t[0].size()[0]
        # self.lam3 *= train_3rd_t[0].size()[0]
        # self.lam_y = train_2nd_t[0].size()[0]

        writer = SummaryWriter()
        
        filename = None
        if len(sys.argv) == 2:
            filename = sys.argv[1]
        lista1 = []
        listb1 = []
        listc1 = []
        listd1 = []
        lista2 = []
        listb2 = []
        listc2 = []
        listd2 = []
        listS = []
        t = []
        # self.predS_update(train_1st_t_predS)

        for idx in range(200):
            train_data, unselected_train_data, test_data, unselected_test_data = demand(Config.sample_num * 10, rand_seed)
            train_data_predS, unselected_train_data_predS, test_data_predS, unselected_test_data_predS = demand(Config.sample_num * 10, rand_seed)

            train_1st_t, train_2nd_t, train_3rd_t = concat_dataset(train_data,
                                                                unselected_train_data), train_data, unselected_train_data
            train_1st_t = TrainDataSetTorch.from_numpy(train_1st_t)
            train_2nd_t = TrainDataSetTorch.from_numpy(train_2nd_t)
            train_3rd_t = TrainDataSetTorch.from_numpy(train_3rd_t)
            test_data_t = TestDataSetTorch.from_numpy(test_data)
            unselected_test_data_t = TestDataSetTorch.from_numpy(unselected_test_data)
        
            train_1st_t_predS, train_2nd_t_predS, train_3rd_t_predS = concat_dataset(train_data_predS,
                                                                unselected_train_data_predS), train_data_predS, unselected_train_data_predS
            train_1st_t_predS = TrainDataSetTorch.from_numpy(train_1st_t_predS)
            train_2nd_t_predS = TrainDataSetTorch.from_numpy(train_2nd_t_predS)
            train_3rd_t_predS = TrainDataSetTorch.from_numpy(train_3rd_t_predS)
            test_data_t_predS = TestDataSetTorch.from_numpy(test_data_predS)
            unselected_test_data_t_predS = TestDataSetTorch.from_numpy(unselected_test_data_predS)

            if self.gpu_flg:
                train_1st_t = train_1st_t.to_gpu()
                train_2nd_t = train_2nd_t.to_gpu()
                train_3rd_t = train_3rd_t.to_gpu()
                test_data_t = test_data_t.to_gpu()
                train_1st_t_predS = train_1st_t_predS.to_gpu()
                unselected_test_data_t = unselected_test_data_t.to_gpu()
            # 调整正则化超参数，与数据规模成比例
            if idx == 0:
                self.lam1 *= train_1st_t[0].size()[0]
                self.lam2 *= train_2nd_t[0].size()[0]
                self.lam3 *= train_3rd_t[0].size()[0]
                self.lam_y = train_2nd_t[0].size()[0]

            writer = SummaryWriter()
            for tt in range(self.n_epoch): # 2SIS
            #     self.stage1_update(train_1st_t, tt, writer)
            #     # if self.covariate_net:
            #     #     self.update_covariate_net(train_1st_t, train_2nd_t, tt, writer)
                self.stage2_update(train_1st_t, train_2nd_t, tt, writer)
            writer.close()
            mdl = TSSIModel(self.treatment_net, self.instrumental_net, self.selection_net,
                            self.covariate_net, self.r1_net, self.r0_net, self.odds_net, self.phi_net, self.S1_net, self.y_net, self.y1_net,
                            self.add_stage1_intercept, self.add_stage2_intercept,
                            self.odds_iter, self.selection_weight_decay,
                            self.r1_weight_decay, self.r0_weight_decay, self.odds_weight_decay, self.y_weight_decay, self.y1_weight_decay, self.lam_y, self.distance_dim)
            loss = mdl.fit_t(train_1st_t, train_2nd_t, train_3rd_t, train_1st_t_predS, self.lam1, self.lam2, self.lam3) # shadow variable -> Selection Bias

            if self.gpu_flg:
                torch.cuda.empty_cache()

            oos_loss: numpy.ndarray = mdl.evaluate_t(test_data_t)
            unselected_loss: numpy.ndarray = mdl.evaluate_t(unselected_test_data_t)
            res = oos_loss, unselected_loss, loss
            print("Epoch{} s=1: {:.2f}, {:.2f}, {:.2f}, {:.2f} s=0: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                idx, 
                float(res[0][0]), 
                float(res[0][1]), 
                float(res[0][2]), 
                float(res[0][3]), 
                float(res[1][0]), 
                float(res[1][1]), 
                float(res[1][2]), 
                float(res[1][3])
            ))
            
            loss_a1_values, loss_b1_values, loss_c1_values, loss_d1_values = res[0]
            loss_a2_values, loss_b2_values, loss_c2_values, loss_d2_values = res[1]
            loss_S_values = res[2]
            lista1.append(loss_a1_values)
            listb1.append(loss_b1_values)
            listc1.append(loss_c1_values)
            listd1.append(loss_d1_values)
            lista2.append(loss_a2_values)
            listb2.append(loss_b2_values)
            listc2.append(loss_c2_values)
            listd2.append(loss_d2_values)
            listS.append(loss_S_values)
            t.append(idx)
            

            # 第一个图：显示模型a和b的loss
            plt.figure(figsize=(12, 6))
            plt.plot(t, lista1, marker='o', linestyle='-', color='b', label='Y Loss')
            plt.plot(t, listb1, marker='s', linestyle='-', color='g', label='Y with W Loss')
            # plt.plot(t, listc1, marker='x', linestyle='-', color='r', label='Y Balanced Loss')
            plt.plot(t, listd1, marker='^', linestyle='-', color='m', label='Y Linear Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            if filename is not None:
                plt.savefig(f'result/{filename}_s1_{exp_num}')

            # 第二个图：显示模型c和d的loss
            plt.figure(figsize=(12, 6))
            plt.plot(t, lista2, marker='o', linestyle='-', color='b', label='Y Loss')
            plt.plot(t, listb2, marker='s', linestyle='-', color='g', label='Y with W Loss')
            # plt.plot(t, listc2, marker='x', linestyle='-', color='r', label='Y Balanced Loss')
            plt.plot(t, listd2, marker='^', linestyle='-', color='m', label='Y Linear Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            if filename is not None:
                plt.savefig(f'result/{filename}_s0_{exp_num}')


            plt.figure(figsize=(12, 6))
            plt.plot(t, listS, marker='o', linestyle='-', color='b', label='Spred Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            if filename is not None:
                plt.savefig(f'result/{filename}_Spred_{exp_num}')
        return oos_loss, unselected_loss, loss

    def stage1_update(self, train_1st_t: TrainDataSetTorch, epoch: int, writer: SummaryWriter):
        self.treatment_net.train(False)
        self.instrumental_net.train(True)
        self.phi_net.train(True)
        self.s1_net.train(True)
        self.S1_net.train(False)
        self.selection_net.train(False)
        
        bce_func = nn.BCELoss()
        if self.covariate_net:
            self.covariate_net.train(False)
        mi_estimator = eval("CLUB")(self.distance_dim, self.distance_dim, self.distance_dim * 2) 
        if self.gpu_flg:
            mi_estimator = mi_estimator.to("cuda:0")
        mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-4) 
        treatment_feature = self.treatment_net(train_1st_t.treatment).detach() 
        for i in range(self.stage1_iter):
            self.instrumental_opt.zero_grad()
            instrumental_feature = self.instrumental_net(train_1st_t.instrumental) 
            covariate_feature = self.covariate_net(train_1st_t.covariate)
            feature_t = TSSIModel.augment_stage1_feature(instrumental_feature, self.add_stage1_intercept) 
            loss_t = self.lam_a * linear_reg_loss(treatment_feature, feature_t, self.lam1)
            loss_t.backward() 
            self.instrumental_opt.step() 
            writer.add_scalar('InstrumentalNet Train loss', loss_t, epoch * self.stage1_iter + i)
            for j in range(self.mi_iter): 
                mi_estimator.train(True)
                phi_feature = self.phi_net(train_1st_t.instrumental) 
                mi_loss = mi_estimator.learning_loss(phi_feature, train_1st_t.treatment) 
                mi_optimizer.zero_grad()
                mi_loss.backward()
                mi_optimizer.step()
            mi_estimator.train(False) 
            phi_feature = self.phi_net(train_1st_t.instrumental) 
            s_pred = self.s1_net(torch.cat((train_1st_t.treatment, train_1st_t.covariate, phi_feature), 1)) 
            loss_s = bce_func(s_pred, train_1st_t.selection) + self.lam4 * mi_estimator(phi_feature, train_1st_t.treatment) 
            self.s1_opt.zero_grad()
            loss_s.backward()
            self.s1_opt.step()
            writer.add_scalar('Phi Train loss', loss_s, epoch * self.stage1_iter + i)
    
    def predS_update(self, train_1st_data_t_predS: TrainDataSetTorch):
        loss_func = nn.BCELoss()
        writer = SummaryWriter()
        self.treatment_net.train(False)
        self.instrumental_net.train(True)
        self.phi_net.train(True)
        self.s1_net.train(True)
        self.S1_net.train(False)
        self.selection_net.train(True)
        instrumental_1st_feature_predS = self.instrumental_net(train_1st_data_t_predS.instrumental).detach()
        feature = TSSIModel.augment_stage1_feature(instrumental_1st_feature_predS, self.add_stage1_intercept)
        treatment_1st_feature_predS = self.treatment_net(train_1st_data_t_predS.treatment).detach()
        stage1_weight = fit_linear(treatment_1st_feature_predS, feature, self.lam1) 
        predicted_treatment_1st_feature_predS = linear_reg_pred(feature, stage1_weight)
        selection_1st_d_predS = train_1st_data_t_predS.selection
        phi_1st_feature_predS = self.phi_net(train_1st_data_t_predS.instrumental).detach()
        for e in range(100):
            self.selection_opt.zero_grad()
            selection_pred = self.selection_net(torch.cat((predicted_treatment_1st_feature_predS, phi_1st_feature_predS, train_1st_data_t_predS.covariate), 1))
            loss_selection = loss_func(selection_pred, selection_1st_d_predS)
            loss_selection.backward()
            self.selection_opt.step()
            writer.add_scalar('SelectionNet Train loss', loss_selection, e)

        


    def stage2_update(self, train_1st_t: TrainDataSetTorch, train_2nd_t: TrainDataSetTorch, epoch: int, writer: SummaryWriter):
        self.treatment_net.train(True)
        self.instrumental_net.train(False)
        self.phi_net.train(False)
        self.S1_net.train(False)
        self.y_net.train(False)
        self.y1_net.train(False)
        self.selection_net.train(False)

        if self.covariate_net:
            self.covariate_net.train(False)

        # have instrumental features
        instrumental_1st_feature = self.instrumental_net(train_1st_t.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_t.instrumental).detach()

        phi_2nd_feature = self.phi_net(train_2nd_t.instrumental).detach()

        covariate_2nd_feature = None
        # have covariate features
        if self.covariate_net:
            covariate_2nd_feature = self.covariate_net(train_2nd_t.covariate).detach()
            covariate_1st_feature = self.covariate_net(train_1st_t.covariate).detach()

        for i in range(self.stage2_iter):
            self.treatment_opt.zero_grad()
            treatment_1st_feature = self.treatment_net(train_1st_t.treatment)
            treatment_2nd_feature = self.treatment_net(train_2nd_t.treatment)
            res = TSSIModel.fit_2sls(treatment_1st_feature,
                                     treatment_2nd_feature,
                                     instrumental_1st_feature,
                                     instrumental_2nd_feature,
                                     phi_2nd_feature,
                                     covariate_1st_feature,
                                     covariate_2nd_feature,
                                     train_2nd_t.outcome,
                                     self.lam1, self.lam2,
                                     self.add_stage1_intercept,
                                     self.add_stage2_intercept)
            loss = res["stage2_loss"]
            loss.backward()
            self.treatment_opt.step()
            writer.add_scalar('TreatmentNet Train loss', loss, epoch * self.stage1_iter + i)

    def update_covariate_net(self, train_1st_data: TrainDataSetTorch, train_2nd_data: TrainDataSetTorch, epoch: int, writer: SummaryWriter):
        # have instrumental features
        self.selection_net.train(False)
        self.instrumental_net.train(False)
        instrumental_1st_feature = self.instrumental_net(train_1st_data.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_data.instrumental).detach()

        self.treatment_net.train(False)
        treatment_1st_feature = self.treatment_net(train_1st_data.treatment).detach()
        treatment_2nd_feature = self.treatment_net(train_2nd_data.treatment).detach()

        feature = TSSIModel.augment_stage1_feature(instrumental_1st_feature, self.add_stage1_intercept)
        stage1_weight = fit_linear(treatment_1st_feature, feature, self.lam1) 

        feature = TSSIModel.augment_stage1_feature(instrumental_2nd_feature, self.add_stage1_intercept)
        predicted_treatment_feature = linear_reg_pred(feature, stage1_weight).detach() 

        self.covariate_net.train(True)
        self.phi_net.train(False)
        phi_feature = self.phi_net(train_2nd_data.instrumental).detach()
        for i in range(self.covariate_iter): 
            self.covariate_opt.zero_grad()
            covariate_feature = self.covariate_net(train_2nd_data.covariate)
            # stage2 - y1 regression
            feature = TSSIModel.augment_stage_y1_feature(predicted_treatment_feature,
                                                         phi_feature,
                                                         covariate_feature,
                                                         self.add_stage2_intercept)
            loss = linear_reg_loss(train_2nd_data.outcome, feature, self.lam2)
            loss.backward()
            self.covariate_opt.step()
            writer.add_scalar('CovariateNet Train loss', loss, epoch * self.stage1_iter + i)
