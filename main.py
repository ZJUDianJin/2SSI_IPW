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
        lista1 = []
        listb1 = []
        listc1 = []
        listd1 = []
        lista2 = []
        listb2 = []
        listc2 = []
        listd2 = []
        t = []
        for idx in range(Config.experiment_num):
            model = TSSITrainer(Config.config['model_structure'], Config.config['train_params'], True)
            res = model.train(rand_seed=42)
            res_list.append(res)
            print("Epoch ", idx, " : s=1:", int(res[0][0]), int(res[0][1]), int(res[0][2]), "s=0:", int(res[1][0]), int(res[1][1]), int(res[1][2]))
            # p.savetxt("res/y_result.txt", res)
            loss_a1_values, loss_b1_values, loss_c1_values, loss_d1_values = res[0]
            loss_a2_values, loss_b2_values, loss_c2_values, loss_d2_values = res[1]
            lista1.append(loss_a1_values)
            listb1.append(loss_b1_values)
            listc1.append(loss_c1_values)
            listd1.append(loss_d1_values)
            lista2.append(loss_a2_values)
            listb2.append(loss_b2_values)
            listc2.append(loss_c2_values)
            listd2.append(loss_d2_values)
            t.append(idx)

        # 第一个图：显示模型a和b的loss
        plt.figure(figsize=(12, 6))
        plt.plot(t, lista1, marker='o', linestyle='-', color='b', label='Model A1 Loss')
        plt.plot(t, listb1, marker='s', linestyle='-', color='g', label='Model B1 Loss')
        plt.plot(t, listc1, marker='x', linestyle='-', color='r', label='Model C1 Loss')
        plt.plot(t, listd1, marker='^', linestyle='-', color='m', label='Model D1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('result/Loss_simlple_net_1.png')
        plt.show()

        # 第二个图：显示模型c和d的loss
        plt.figure(figsize=(12, 6))
        plt.plot(t, lista2, marker='o', linestyle='-', color='b', label='Model A2 Loss')
        plt.plot(t, listb2, marker='s', linestyle='-', color='g', label='Model B2 Loss')
        plt.plot(t, listc2, marker='x', linestyle='-', color='r', label='Model C2 Loss')
        plt.plot(t, listd2, marker='^', linestyle='-', color='m', label='Model D2 Loss')
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
