from torch import nn

class Config:
    config = {
        "model_structure": [
            # treatment_net
            nn.Sequential(nn.Linear(1, 1)), 
            # instrumental_net
            nn.Sequential(
                          nn.Linear(3, 1)
                          # nn.ReLU(),
                          # nn.Linear(3, 128),
                          # nn.ReLU(),
                          # nn.Linear(128, 64),
                          # nn.BatchNorm1d(64),
                          # nn.ReLU(),
                          # nn.Linear(64, 1),
                        #   nn.BatchNorm1d(3)
                          ),
            # selection_net
            nn.Sequential(
                          nn.Linear(4, 1),
                          # nn.BatchNorm1d(128),
                          # nn.ReLU(),
                          # nn.Linear(128, 1),
                          nn.Sigmoid()
                          ),
            # covariate_net
            nn.Sequential(nn.Linear(2, 128),
                          nn.ReLU(),
                          nn.Linear(128, 32),
                          nn.BatchNorm1d(32),
                          nn.ReLU(),
                          nn.Linear(32, 1),
                          ),
              ## bigXnet
            # nn.Sequential(nn.Linear(2, 256),
            #               nn.ReLU(),
            #               nn.Linear(256, 128),
            #               nn.BatchNorm1d(128),
            #               nn.ReLU(),
            #               nn.Linear(128, 32),
            #               nn.BatchNorm1d(32),
            #               nn.ReLU(),
            #               nn.Linear(32, 1),
            #               ),
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
            nn.Sequential(nn.Linear(5, 1),
                          # nn.BatchNorm1d(16),
                          # nn.ReLU(),
                          # nn.Linear(16, 1),
                          nn.Softplus()
                          ),
            # S1_net
            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          nn.Sigmoid()
                          ),
            # y_net
            nn.Sequential(nn.Linear(3, 1)
                        #   nn.ReLU(),
                        #   nn.Linear(128, 32),
                        #   # nn.Linear(16, 1)
                        # #   nn.Linear(128, 32),
                        #   nn.BatchNorm1d(32),
                        #   nn.ReLU(),
                        #   nn.Linear(32, 1)
                          ),
            # y1_net
            nn.Sequential(nn.Linear(3, 1)
                        #   nn.ReLU(),
                        #   nn.Linear(128, 32),
                        #   # nn.Linear(16, 1)
                        # #   nn.Linear(128, 32),
                        #   nn.BatchNorm1d(32),
                        #   nn.ReLU(),
                        #   nn.Linear(32, 1)
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
            "y1_weight_decay": 0.0,
            "lam_y": 0.1,
            "lam_a": 20000
        }
    }
    experiment_num = 200
    c_strength = 1
    u_strength = 10
    sample_num = 5000
