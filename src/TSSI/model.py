from typing import Optional
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, outer_prod, add_const_col, fit_weighted_linear
from src.data.data_class import TrainDataSetTorch, TestDataSetTorch, concat_dataset


class TSSIModel:
    stage1_weight: torch.Tensor
    stage2_y1_weight: torch.Tensor
    stage2_y0_weight: torch.Tensor

    def __init__(self,
                 treatment_net: nn.Module,
                 instrumental_net: nn.Module,
                 selection_net: nn.Module,
                 covariate_net: Optional[nn.Module],
                 r1_net: nn.Module,
                 r0_net: nn.Module,
                 odds_net: nn.Module,
                 phi_net: nn.Module,
                 S1_net: nn.Module,
                 y_net: nn.Module,
                 y1_net: nn.Module,
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool,
                 odds_iter: int,
                 selection_weight_decay: float,
                 r1_weight_decay: float,
                 r0_weight_decay: float,
                 odds_weight_decay: float,
                 y_weight_decay: float,
                 y1_weight_decay: float
                 ):
        self.treatment_net = treatment_net
        self.instrumental_net = instrumental_net
        self.selection_net = selection_net
        self.odds_net = odds_net
        self.covariate_net = covariate_net
        self.r0_net = r0_net
        self.r1_net = r1_net
        self.phi_net = phi_net
        self.S1_net = S1_net
        self.y_net = y_net
        self.y1_net = y1_net
        self.add_stage1_intercept = add_stage1_intercept
        self.add_stage2_intercept = add_stage2_intercept
        self.odds_iter = odds_iter
        self.selection_opt = torch.optim.Adam(self.selection_net.parameters(),
                                              weight_decay=selection_weight_decay)
        self.odds_opt = torch.optim.Adam(self.odds_net.parameters(),
                                         weight_decay=odds_weight_decay)
        self.selection_opt = torch.optim.Adam(self.selection_net.parameters(),
                                              weight_decay=selection_weight_decay)
        self.r1_opt = torch.optim.Adam(self.r1_net.parameters(),
                                       weight_decay=r1_weight_decay)
        self.r0_opt = torch.optim.Adam(self.r0_net.parameters(),
                                       weight_decay=r0_weight_decay)
        self.y_opt = torch.optim.Adam(self.y_net.parameters(),
                                       weight_decay=y_weight_decay)
        self.y1_opt = torch.optim.Adam(self.y1_net.parameters(),
                                      weight_decay=y1_weight_decay)
        self.condition_dim = 2

    @staticmethod
    def augment_stage1_feature(instrumental_feature: torch.Tensor,
                               add_stage1_intercept: bool):

        feature = instrumental_feature
        if add_stage1_intercept:
            feature = add_const_col(feature)
        return feature
    
    def augment_stage1_feature_plus(instrumental_feature: torch.Tensor,
                               add_stage1_intercept: bool):

        feature = instrumental_feature
        if add_stage1_intercept:
            feature = add_const_col(feature)
        return feature

    @staticmethod
    def augment_stage2_feature(treatment_feature: torch.Tensor,
                               covariate_feature: torch.Tensor,
                               add_stage2_intercept: bool):
        feature = treatment_feature
        if add_stage2_intercept:
            feature = add_const_col(feature)
        if covariate_feature is not None:
            feature_tmp = covariate_feature
            if add_stage2_intercept:
                feature_tmp = add_const_col(feature_tmp)
            feature = outer_prod(feature, feature_tmp)
            feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def augment_stage_y1_feature(treatment_feature: torch.Tensor,
                                 residual: torch.Tensor,
                                 covariate_feature: Optional[torch.Tensor],
                                 add_stage2_intercept: bool):
        feature = treatment_feature
        if add_stage2_intercept:
            feature = add_const_col(feature)

        if covariate_feature is not None:
            feature_tmp = covariate_feature
            if add_stage2_intercept:
                feature_tmp = add_const_col(feature_tmp)
            feature = outer_prod(feature, feature_tmp)
            feature = torch.flatten(feature, start_dim=1)

        feature = torch.cat((feature, residual), 1)

        return feature
    
    @staticmethod
    def augment_stage_y1_feature_plus(treatment_feature: torch.Tensor,
                                 residual: torch.Tensor,
                                 covariate_feature: Optional[torch.Tensor],
                                 add_stage2_intercept: bool):
        feature = torch.cat((treatment_feature, residual, covariate_feature), 1)
        if add_stage2_intercept:
            feature = add_const_col(feature)
        # feature = treatment_feature
        # if add_stage2_intercept:
        #     feature = add_const_col(feature)

        # if covariate_feature is not None:
        #     feature_tmp = covariate_feature
        #     if add_stage2_intercept:
        #         feature_tmp = add_const_col(feature_tmp)
        #     feature = outer_prod(feature, feature_tmp)
        #     feature = torch.flatten(feature, start_dim=1)
        #     feature = outer_prod(feature, residual)
        #     feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def fit_2sls(treatment_1st_feature: torch.Tensor,
                 treatment_2nd_feature: torch.Tensor,
                 instrumental_1st_feature: torch.Tensor,
                 instrumental_2nd_feature: torch.Tensor,
                 phi_2nd_feature: torch.Tensor,
                 covariate_2nd_feature: Optional[torch.Tensor],
                 outcome_2nd_t: torch.Tensor,
                 lam1: float, lam2: float,
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool,
                 ):

        # stage1
        feature = TSSIModel.augment_stage1_feature(instrumental_1st_feature, add_stage1_intercept)
        stage1_weight = fit_linear(treatment_1st_feature, feature, lam1)

        # predicting for stage 2
        feature = TSSIModel.augment_stage1_feature(instrumental_2nd_feature, add_stage1_intercept)
        predicted_treatment_2nd_feature = linear_reg_pred(feature, stage1_weight) # T^

        # stage2 - y1 regression
        feature = TSSIModel.augment_stage_y1_feature(predicted_treatment_2nd_feature,
                                                     phi_2nd_feature,
                                                     covariate_2nd_feature,
                                                     add_stage2_intercept)

        stage2_weight = fit_linear(outcome_2nd_t, feature, lam2)
        pred = linear_reg_pred(feature, stage2_weight) # Y^
        stage2_loss = torch.norm((outcome_2nd_t - pred)) ** 2 + lam2 * torch.norm(stage2_weight) ** 2

        return dict(stage1_weight=stage1_weight,
                    stage2_weight=stage2_weight,
                    stage2_loss=stage2_loss)

    def fit_odds(self,
                 treatment_1st_feature: torch.Tensor,
                 treatment_2nd_feature: torch.Tensor,
                 treatment_3rd_feature: torch.Tensor,
                 instrumental_1st_feature: torch.Tensor,
                 instrumental_2nd_feature: torch.Tensor,
                 instrumental_3rd_feature: torch.Tensor,
                 phi_1st_feature: torch.Tensor,
                 phi_2nd_feature: torch.Tensor,
                 covariate_1st_feature: Optional[torch.Tensor],
                 covariate_2nd_feature: Optional[torch.Tensor],
                 covariate: Optional[torch.Tensor],
                 odds_2nd_feature: torch.Tensor,
                 odds_2nd_predicted_feature: torch.Tensor,
                 selection_probability: torch.Tensor, 
                 r1_feature: torch.Tensor,
                 r0_feature: torch.Tensor,
                 outcome_2nd_t: torch.Tensor,
                 lam2: float,
                 lam3: float,
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool,
                 ):
        # residual for stage 2
        feature = TSSIModel.augment_stage1_feature(instrumental_2nd_feature, add_stage1_intercept)
        predicted_treatment_2nd_feature = linear_reg_pred(feature, self.stage1_weight)
        feature = TSSIModel.augment_stage1_feature(instrumental_3rd_feature, add_stage1_intercept)
        predicted_treatment_3rd_feature = linear_reg_pred(feature, self.stage1_weight)

        writer = SummaryWriter()
        # stage2 - f(R | X, S)
        loss_func = nn.MSELoss()
        self.r0_net.train(True)
        self.r1_net.train(True)
        for e in range(self.odds_iter):
            self.r0_opt.zero_grad()
            r0_pred_0 = self.r0_net(r0_feature)
            sign_loss = torch.mean((torch.sign(r0_pred_0) != torch.sign(instrumental_3rd_feature)).float())
            loss_r0 = loss_func(r0_pred_0, instrumental_3rd_feature) + lam3 * sign_loss
            loss_r0.backward()
            self.r0_opt.step()
            self.r1_opt.zero_grad()
            r1_pred = self.r1_net(r1_feature)
            sign_loss = torch.mean((torch.sign(r1_pred) != torch.sign(instrumental_2nd_feature)).float())
            loss_r1 = loss_func(r1_pred, instrumental_2nd_feature) + lam3 * sign_loss
            loss_r1.backward()
            self.r1_opt.step()
            writer.add_scalar('Z0 Train loss', loss_r0, e)
            writer.add_scalar('Z1 Train loss', loss_r1, e)

        self.r0_net.train(False)
        self.r1_net.train(False)
        with torch.no_grad():
            pred_s0 = self.r0_net(r1_feature).detach()
            pred_s1 = self.r1_net(r1_feature).detach()

        # E[OR(X,Y)|X, R, S=1] = f(R | X, S=1) / f(R | X, S=0)
        odds_2nd_d = pred_s0 / pred_s1

        # stage2 - OR~(X,Y) 
        self.odds_net.train(True)
        for e in range(self.odds_iter):
            self.odds_opt.zero_grad()
            odds_pred = self.odds_net(odds_2nd_feature)
            loss_odds = loss_func(odds_pred, odds_2nd_d) + lam3 * torch.mean(
                torch.max(torch.zeros(odds_pred.shape).to(odds_pred.device), odds_pred) ** 2)
            loss_odds.backward()
            self.odds_opt.step()
            writer.add_scalar('OR Train loss', loss_odds, e)
        self.odds_net.train(False)
        pred_or_tilde = self.odds_net(odds_2nd_feature).detach() # OR~(X,Y) = OR(X,Y) * OR～(X,Y=0)
        pred_or_tilde_mean = self.odds_net(odds_2nd_predicted_feature).detach() # E[OR~(X,Y)|X,R,S=1] = E[OR(T,Y)|V,X,S=1] * OR~(X,Y=0)

        # stage2 - OR(X,Y)/E[OR(X,Y) | X, R, S=1]
        pred_or_mean = pred_or_tilde / pred_or_tilde_mean

        # W
        phi_feature = instrumental_2nd_feature
        S1_pred = self.S1_net(torch.cat((predicted_treatment_2nd_feature, covariate, phi_feature), 1)).detach()
        ratio = (1 - S1_pred) / S1_pred
        p_Y = 1 / (1 + ratio * pred_or_mean)
        W = 1 / p_Y
        W_new = W.squeeze(1)
        W_real = 1 / selection_probability
        W_real = W_real.squeeze(1)

        # condition_feature = torch.concat((residual_2nd_feature, phi_2nd_feature), 1)
        # self.condition_dim = condition_feature.shape[1]
        # feature = TSSIModel.augment_stage_y1_feature(predicted_treatment_2nd_feature,
        #                                              phi_2nd_feature,
        #                                              covariate_2nd_feature,
        #                                              add_stage2_intercept)
        # stage2_weight = fit_weighted_linear(outcome_2nd_t, feature, lam2, W_real)
        # # pred_2nd_s1_outcome = linear_reg_pred(feature, stage2_weight)

        # return dict(stage2_weight=stage2_weight)
        
        # predict Y
        self.odds_net.train(False)
        self.treatment_net.train(False)
        self.instrumental_net.train(False)
        self.covariate_net.train(False)
        self.phi_net.train(False)
        self.S1_net.train(False)
        self.y_net.train(True) 
        self.y1_net.train(True)
        for e in range(self.odds_iter):
            self.y_net.zero_grad()
            feature = torch.cat((predicted_treatment_2nd_feature, phi_feature, covariate_2nd_feature), 1).detach()
            outcome_y = self.y_net(feature)
            # loss_y = torch.sum(W * (outcome_2nd_t - outcome_y) ** 2)
            loss_func = nn.MSELoss(reduction='none')
            mse_loss = loss_func(outcome_y, outcome_2nd_t)
            loss_y = torch.sum(W_real * mse_loss)
            loss_y.backward()
            self.y_opt.step()
            writer.add_scalar('Y Train loss', loss_y, e)

            feature = torch.cat((predicted_treatment_2nd_feature, phi_feature, covariate_2nd_feature), 1).detach()
            self.y1_net.zero_grad()
            outcome_y = self.y1_net(feature)
            loss_y1 = torch.sum((outcome_2nd_t - outcome_y) ** 2)
            loss_y1.backward()
            self.y1_opt.step()
            writer.add_scalar('Y1 Train loss', loss_y1, e) 
        writer.close()
        


    def fit_selection(self,
                      treatment: torch.Tensor,
                      phi: torch.Tensor,
                      covariate: torch.Tensor,
                      selection_1st_d: torch.Tensor):
        loss_func = nn.BCELoss()
        writer = SummaryWriter()
        self.selection_net.train(True)
        for e in range(self.odds_iter):
            self.selection_opt.zero_grad()
            selection_pred = self.selection_net(torch.cat((treatment, phi, covariate), 1))
            loss_selection = loss_func(selection_pred, selection_1st_d)
            loss_selection.backward()
            self.selection_opt.step()
            writer.add_scalar('SelectionNet Train loss', loss_selection, e)

    def fit_t(self,
              train_1st_data_t: TrainDataSetTorch,
              train_2nd_data_t: TrainDataSetTorch,
              train_3rd_data_t: TrainDataSetTorch,
              lam1: float, lam2: float, lam3: float):
        self.treatment_net.train(False)
        self.covariate_net.train(False)
        self.instrumental_net.train(False)
        self.phi_net.train(False)
        self.S1_net.train(False)
        treatment_1st_feature = self.treatment_net(train_1st_data_t.treatment).detach()
        treatment_2nd_feature = self.treatment_net(train_2nd_data_t.treatment).detach()
        treatment_3rd_feature = self.treatment_net(train_3rd_data_t.treatment).detach()
        instrumental_1st_feature = self.instrumental_net(train_1st_data_t.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_data_t.instrumental).detach()
        instrumental_3rd_feature = self.instrumental_net(train_3rd_data_t.instrumental).detach()
        selection_probability = train_2nd_data_t.selection_probability
        outcome_2nd_t = train_2nd_data_t.outcome
        selection_1st_d = train_1st_data_t.selection
        phi_1st_feature = self.phi_net(train_1st_data_t.instrumental).detach()
        phi_2nd_feature = self.phi_net(train_2nd_data_t.instrumental).detach()
        covariate_1st_feature = None
        covariate_2nd_feature = None
        covariate = train_2nd_data_t.covariate
        if self.covariate_net is not None:
            covariate_1st_feature = self.covariate_net(train_1st_data_t.covariate).detach()
            covariate_2nd_feature = self.covariate_net(train_2nd_data_t.covariate).detach()

        res = TSSIModel.fit_2sls(treatment_1st_feature,
                                 treatment_2nd_feature,
                                 instrumental_1st_feature,
                                 instrumental_2nd_feature,
                                 phi_2nd_feature,
                                 covariate_2nd_feature,
                                 outcome_2nd_t,
                                 lam1, lam2,
                                 self.add_stage1_intercept,
                                 self.add_stage2_intercept)
        self.stage1_weight = res["stage1_weight"]
        self.stage2_y1_weight = res["stage2_weight"]
        # predict for stage 2 odds
        feature = TSSIModel.augment_stage1_feature(instrumental_2nd_feature, self.add_stage1_intercept)
        predicted_treatment_2nd_feature = linear_reg_pred(feature, self.stage1_weight)
        feature = TSSIModel.augment_stage_y1_feature(predicted_treatment_2nd_feature,
                                                     phi_2nd_feature, 
                                                     covariate_2nd_feature,
                                                     self.add_stage2_intercept)
        predicted_2nd_outcome = linear_reg_pred(feature, self.stage2_y1_weight)
        z1_feature = torch.cat(
            (train_2nd_data_t.treatment, train_2nd_data_t.covariate,
             self.phi_net(train_2nd_data_t.instrumental).detach()), 1)
        z0_feature = torch.cat(
            (train_3rd_data_t.treatment, train_3rd_data_t.covariate,
             self.phi_net(train_3rd_data_t.instrumental).detach()), 1)
        odds_feature = torch.cat(
            (train_2nd_data_t.treatment, train_2nd_data_t.covariate,
             self.phi_net(train_2nd_data_t.instrumental).detach(), train_2nd_data_t.outcome), 1)
        odds_predicted_feature = torch.cat(
            (train_2nd_data_t.treatment, train_2nd_data_t.covariate,
             self.phi_net(train_2nd_data_t.instrumental).detach(), predicted_2nd_outcome), 1)
        selection_probability = train_2nd_data_t.selection_probability
        self.fit_odds(treatment_1st_feature,
                                 treatment_2nd_feature,
                                 treatment_3rd_feature,
                                 instrumental_1st_feature,
                                 instrumental_2nd_feature,
                                 instrumental_3rd_feature,
                                 phi_1st_feature,
                                 phi_2nd_feature,
                                 covariate_1st_feature,
                                 covariate_2nd_feature,
                                 covariate, 
                                 odds_feature,
                                 odds_predicted_feature,
                                 selection_probability,
                                 z1_feature,
                                 z0_feature,
                                 outcome_2nd_t,
                                 lam2,
                                 lam3,
                                 self.add_stage1_intercept,
                                 self.add_stage2_intercept
                                 )
        # self.stage2_weight = res_odds["stage2_weight"]

    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor],
                  instrumental: Optional[torch.Tensor], selection_probability: Optional[torch.Tensor]):
        treatment_feature = self.treatment_net(treatment)
        covariate_feature = None
        instrumental_feature = self.instrumental_net(instrumental).detach()
        if self.covariate_net:
            covariate_feature = self.covariate_net(covariate)
        if instrumental is not None:
            feature = TSSIModel.augment_stage1_feature(instrumental_feature, self.add_stage1_intercept)
            predicted_treatment_feature = linear_reg_pred(feature, self.stage1_weight)
            phi_feature = self.phi_net(instrumental)
        else:
            condition_feature = torch.zeros((len(treatment), self.condition_dim))
        feature = torch.cat((predicted_treatment_feature, phi_feature, covariate_feature), 1)
        return self.y_net(feature)
    
    def predict_t_1(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor],
                  instrumental: Optional[torch.Tensor], selection_probability: Optional[torch.Tensor], outcome: Optional[torch.Tensor]):
        treatment_feature = self.treatment_net(treatment)
        covariate_feature = None
        instrumental_feature = self.instrumental_net(instrumental).detach()
        if self.covariate_net:
            covariate_feature = self.covariate_net(covariate)
        if instrumental is not None:
            feature = TSSIModel.augment_stage1_feature(instrumental_feature, self.add_stage1_intercept)
            predicted_treatment_feature = linear_reg_pred(feature, self.stage1_weight)
            phi_feature = self.phi_net(instrumental)
        else:
            condition_feature = torch.zeros((len(treatment), self.condition_dim))

        feature = TSSIModel.augment_stage_y1_feature(predicted_treatment_feature,
                                                     phi_feature,
                                                     covariate_feature,
                                                     self.add_stage2_intercept)
        pred = linear_reg_pred(feature, self.stage2_y1_weight)
        W = 1 / selection_probability.squeeze(1)
        stage2_weight = fit_weighted_linear(outcome, feature, 0.1, W)
        pred_w = linear_reg_pred(feature, stage2_weight)
        # return pred1, pred
        feature = torch.cat((predicted_treatment_feature, phi_feature, covariate_feature), 1)
        return self.y1_net(feature), pred, pred_w

    def evaluate_t(self, test_data: TestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred1 = self.predict_t(test_data.treatment, test_data.covariate, test_data.instrumental, test_data.selection_probability)
        res1 = (torch.norm((target - pred1)) ** 2) / target.size()[0]
        with torch.no_grad():
            pred2, pred_linear, pred_weight = self.predict_t_1(test_data.treatment, test_data.covariate, test_data.instrumental, test_data.selection_probability, target)
        res2 = (torch.norm((target - pred2)) ** 2) / target.size()[0]
        res_linear = (torch.norm((target - pred_linear)) ** 2) / target.size()[0]
        res_weight = (torch.norm((target - pred_weight)) ** 2) / target.size()[0]
        return res1.detach().cpu().numpy(), res2.detach().cpu().numpy(), res_linear.detach().cpu().numpy(), res_weight.detach().cpu().numpy()