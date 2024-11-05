import traceback
import numpy as np
from src.TSSI.trainer import TSSITrainer
from config import Config
import matplotlib.pyplot as plt
import os

if not os.path.exists('result'):
    os.makedirs('result')

if __name__ == '__main__':
    try:
        res_list = []
        print("begin")
        lista = []
        listb = []
        listc = []
        listd = []
        t = []
        for idx in range(Config.experiment_num):
            model = TSSITrainer(Config.config['model_structure'], Config.config['train_params'], True)
            res = model.train(rand_seed=42)
            res_list.append(res)
            print("Epoch ", idx, " : ", res)
            # p.savetxt("res/y_result.txt", res)
            
            loss_a_values, loss_b_values, _ = res[0]
            loss_c_values, loss_d_values, _ = res[1]
            lista.append(loss_a_values)
            listb.append(loss_b_values)
            listc.append(loss_c_values)
            listd.append(loss_d_values)
            t.append(idx)

        # 第一个图：显示模型a和b的loss
        plt.figure(figsize=(12, 6))
        plt.plot(t, lista, marker='o', linestyle='-', color='b', label='Model A Loss')
        plt.plot(t, listb, marker='s', linestyle='-', color='g', label='Model B Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('result/Loss_simlple_net_1.png')
        plt.show()

        # 第二个图：显示模型c和d的loss
        plt.figure(figsize=(12, 6))
        plt.plot(t, listc, marker='x', linestyle='-', color='r', label='Model C Loss')
        plt.plot(t, listd, marker='^', linestyle='-', color='m', label='Model D Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('result/Loss_simlple_net_2.png')
        plt.show()
        res_list = np.array(res_list)

        bias = np.abs(np.mean(res_list, axis=0)).reshape(res_list.shape[1], 1)
        sd = np.std(res_list, axis=0).reshape(res_list.shape[1], 1)
        print(idx, " - bias:", bias, " sd", sd)
        np.savetxt('res/result.txt', np.concatenate((bias, sd), 0))


    except Exception as e:
        print('Exception: ' + str(e))
        print()
        traceback.print_exc()
