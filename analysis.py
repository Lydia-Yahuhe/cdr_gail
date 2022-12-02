import math
import xlwt
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from flightEnv.cmd import CmdCount
from flightSim.utils import distance

# from parameters import *

font_size = 14
legend_size = font_size - 6
axis_size = font_size - 2


class Analyzer:
    def __init__(self, bins, folder, expert_train, expert_test):
        self.bins = bins
        self.folder = folder
        self.expert_train = expert_train
        self.expert_test = expert_test
        self.titles = ['steps', 'LP_Rew', 'IP_SR', 'IP_REW', 'AD', 'ACR', 'RMSE']
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    def analysis(self, path, keyword='exp', steps=[10000, 100001, 10000]):
        values1 = self.__analysis(keyword='train_' + keyword,
                                  prefix=path + 'dqn_{}_evaluate'.format(keyword),
                                  steps=steps,
                                  expert_path=self.expert_train)
        values2 = self.__analysis(keyword='test_' + keyword,
                                  prefix=path + 'dqn_{}_test'.format(keyword),
                                  steps=steps,
                                  expert_path=self.expert_test)

        work_book = xlwt.Workbook("UTF-8")
        sheet = work_book.add_sheet(keyword)
        for row, (v_train, v_test) in enumerate(zip(values1, values2)):
            for column, ele in enumerate(v_train + v_test):
                sheet.write(row, column, ele)
        work_book.save("analysis_{}.xls".format(keyword))

    def __analysis(self, keyword, prefix, steps, expert_path):
        f1, ax1 = plt.subplots(figsize=(10, 5))
        f2 = plt.figure(figsize=(10, 10))
        ax2 = f2.add_subplot(2, 1, 1)
        ax3 = f2.add_subplot(2, 1, 2)

        # Learned Policy
        e_policy = np.load(expert_path)
        e_rew = e_policy['rews']
        e_num = list(e_policy['num'])
        e_acs = e_policy['acs']

        mean_list_delta = []
        var_list_delta = []
        table_data_list = [self.titles]
        for i, step in enumerate(steps):
            result = self.__get_action_distribution(prefix + '_{}.npz'.format(step), e_rew, e_num, e_acs)
            table_data, f_data, f1_data, f2_data = result
            table_data_list.append([step] + [round(v, 3) for v in table_data if isinstance(v, float)])

            # f(动作使用分布直方图)
            f, ax = plt.subplots(figsize=(10, 5))
            ax.hist(x=f_data, bins=self.bins, range=(0, CmdCount), density=True,
                    align='left', label=['专家', '模型_{}'.format(step)])
            ax.set_xlabel('动作下标范围', fontsize=axis_size)
            ax.set_ylabel("频率", fontsize=axis_size)
            ax.set_title("动作使用分布图", fontsize=font_size)
            ax.set_ylim(ymin=0.0, ymax=0.12)
            ax.legend(fontsize=legend_size, loc='upper right')
            ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            f.savefig(self.folder + 'hist_{}_{}.pdf'.format(keyword, step))
            plt.close(fig=f)

            # f1
            [x1, x2] = f1_data
            if i == 0:
                ax1.plot(x1, label='专家', color='red', linewidth=3, linestyle='dashed')
            ax1.plot(x2, linewidth=1)

            # 动作下标的变化趋势图
            plot = np.linspace(-CmdCount + 1, CmdCount - 1, num=2 * CmdCount).astype(int)
            pdf = st.gaussian_kde(f2_data).pdf(plot)
            ax2.plot(plot, pdf, label='{}'.format(step))

            mean_list_delta.append(np.mean(f2_data))
            var_list_delta.append(np.var(f2_data))

        # 动作使用分布的相似程度（KL散度）
        ax1.set_xlabel('动作下标范围', fontsize=axis_size)
        ax1.set_ylabel("概率密度", fontsize=axis_size)
        ax1.set_ylim(ymin=0.0, ymax=2e-2)
        ax1.set_title("以概率密度函数表示的分布相似度（KL散度）", fontsize=font_size)
        ax1.legend(fontsize=legend_size, loc='upper right')
        ax1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        ax2.set_xlabel('差值范围', fontsize=axis_size)
        ax2.set_ylabel("概率密度", fontsize=axis_size)
        ax2.set_ylim(ymin=0.0, ymax=2.0e-2)
        ax2.set_title("模型动作下标与专家动作下标之差的分布图", fontsize=font_size)
        ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        # ax2.legend(fontsize=legend_size, loc='upper right')

        l1, = ax3.plot(steps, mean_list_delta, 'r')
        ax3.axhline(linestyle='--', label='均值为0')
        ax3.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        ax3_1 = ax3.twinx()
        l2, = ax3_1.plot(steps, var_list_delta, 'g')
        ax3.legend([l1, l2], ['均值', '方差'], fontsize=legend_size, loc='upper right')
        ax3_1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax3.set_xlabel('迭代次数', fontsize=axis_size)
        ax3.set_ylabel("均值", fontsize=axis_size)
        ax3_1.set_ylabel("方差", fontsize=axis_size)
        ax3.set_ylim(ymin=-40, ymax=40)
        ax3_1.set_ylim(ymin=0, ymax=4e3)
        ax3.set_title("动作下标差值分布的参数（均值与方差）", fontsize=font_size)

        f1.subplots_adjust(wspace=0.3, hspace=0.25)  # 调整两幅子图的间距
        f2.subplots_adjust(wspace=0.3, hspace=0.25)  # 调整两幅子图的间距

        f1.savefig(self.folder + 'f1_' + keyword + '.pdf')
        f2.savefig(self.folder + 'f2_' + keyword + '.pdf')
        return table_data_list

    def __get_action_distribution(self, learned_path, e_rew, e_num, e_acs):
        # Imitation Policy
        p_policy = np.load(learned_path)
        p_rew = p_policy['rews']
        p_num = list(p_policy['num'])
        p_acs = p_policy['acs']

        size = len(e_num)

        # LP_REW和IP_SR, IP_REW
        lp_two = [size, np.mean(e_rew)]
        ip_two = [len(p_rew) / size * 100, np.mean(p_rew)]

        kl, pdf = self.__compute_kl(e_acs, p_acs)  # Action Distribution (AD)

        delta_error, root_error, count = [], [], 0
        for i, num in enumerate(p_num):
            if num not in e_num:
                continue

            e_action = int(e_acs[e_num.index(num)])
            p_action = int(p_acs[i])
            count += int(e_action == p_action)
            delta = p_action - e_action
            delta_error.append(delta)
            root_error.append(delta ** 2)
        r_mse = np.sqrt(np.mean(root_error))  # Root Mean Square Error (RMSE)
        acr = count / size * 100  # ACR

        return lp_two + ip_two + [kl, acr, r_mse], [e_acs, p_acs], pdf, delta_error

    def __compute_kl(self, x, y):
        bins = self.bins
        plot = np.linspace(0, bins - 1, num=bins).astype(int)
        x_pdf = st.gaussian_kde(x).pdf(plot)
        y_pdf = st.gaussian_kde(y).pdf(plot)
        kl = round(st.entropy(x_pdf, y_pdf), 3)
        return kl, [x_pdf, y_pdf]
