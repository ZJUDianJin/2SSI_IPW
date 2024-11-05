from torch import nn

        # self.treatment_net: nn.Module = networks[0]
        # self.instrumental_net: nn.Module = networks[1]
        # self.selection_net: nn.Module = networks[2]
        # self.covariate_net: Optional[nn.Module] = networks[3]
        # self.r1_net: nn.Module = networks[4]
        # self.r0_net: nn.Module = networks[5]
        # self.phi_net: nn.Module = networks[6]
        # self.s1_net: nn.Module = networks[7]
        # self.odds_net: nn.Module = networks[8]
        # self.S1_net: nn.Module = networks[9]
        # self.y_net: nn.Module = networks[10]
        # self.y1_net: nn.Module = networks[11]

class Config:
    config = {
        "model_structure": [
            # treatment_net
            nn.Sequential(nn.Linear(1, 4)), 
            # instrumental_net
            nn.Sequential(
                          nn.Linear(3, 1)
                          # nn.Linear(3, 128),
                          # nn.ReLU(),
                          # nn.Linear(128, 64),
                          # nn.ReLU(),
                          # nn.Linear(64, 1),
                          # nn.BatchNorm1d(3)
                          ),
            # selection_net
            nn.Sequential(
                          nn.Linear(5, 128),
                          nn.BatchNorm1d(128),
                          nn.ReLU(),
                          nn.Linear(128, 1),
                          nn.Sigmoid()
                          ),
            # covariate_net
            nn.Sequential(
                          nn.Linear(2, 1)
                          # nn.Linear(2, 128),
                          # nn.ReLU(),
                          # nn.Linear(128, 64),
                          # nn.ReLU(),
                          # nn.Linear(64, 32),
                          # nn.BatchNorm1d(32),
                          # nn.ReLU(),
                          # nn.Linear(32, 1),
                          # nn.ReLU()
                          ),
            # r1_net
            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          ),
            # r0_net
            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          ),
            # phi_net
            nn.Sequential(nn.Linear(3, 1),
                          nn.Sigmoid()
                          ),
            # s1_net
            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          nn.Sigmoid()
                          ),
            # odds_net   
            nn.Sequential(nn.Linear(5, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          nn.Softplus()
                          ),
            # S1_net
            nn.Sequential(nn.Linear(7, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          nn.Sigmoid()
                          ),
            # y_net
            nn.Sequential(nn.Linear(6, 128),
                          nn.ReLU(),
                          nn.Linear(128, 32),
                          nn.BatchNorm1d(32),
                          nn.ReLU(),
                          nn.Linear(32, 1), 
                          ),
            # y1_net
            nn.Sequential(nn.Linear(6, 1),
                          # nn.ReLU(),
                          # nn.Linear(128, 32),
                          # nn.BatchNorm1d(32),
                          # nn.ReLU(),
                          # nn.Linear(32, 1), 
                          ),
        ],
        "train_params": {
            "distance_dim": 1,
            "lam1": 0.1,
            "lam2": 0.1,
            "lam3": 0.1,
            "stage1_iter": 20,
            "stage1_S1_iter": 5,
            "covariate_iter": 20,
            "mi_iter": 20,
            "odds_iter": 100,
            "stage2_iter": 1,
            "lam4": 0.1,
            "n_epoch": 100,
            "treatment_weight_decay": 0.0,
            "instrumental_weight_decay": 0.0,
            "covariate_weight_decay": 0.1,
            "s1_weight_decay": 0.0,
            "odds_weight_decay": 0.0,
            "selection_weight_decay": 0.0,
            "r0_weight_decay": 0.0,
            "r1_weight_decay": 0.0,
            "S1_weight_decay": 0.0,
            "S0_weight_decay": 0.0,
            "y_weight_decay": 0.0,
            "y1_weight_decay": 0.0
        }
    }
    experiment_num = 50
    c_strength = 10
    u_strength = 10
    sample_num = 5000
