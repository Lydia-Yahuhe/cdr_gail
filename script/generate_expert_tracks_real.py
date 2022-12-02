import matplotlib.pyplot as plt

from withTracks.rdp.file_processor import get_fpl_list
from withTracks.rdp.model import AgentSetReal


def main():
    # 从excel中提取轨迹和航班信息
    alt_limit = 6000.0
    date_limit = [str(date) for date in range(20210611, 20210621)]
    print(date_limit)
    fpl_set = get_fpl_list(alt_limit=alt_limit, date_limit=date_limit)
    return

    remain = None
    for date in date_limit:
        fpl_dict = fpl_set[date]
        fpl_list, starts = fpl_dict['fpl_list'], fpl_dict['starts']

        # flight_times = []
        # for fpl in fpl_list:
        #     [dep, arr] = fpl.from_to.split('-')
        #     if dep.startswith('ZH') or arr.startswith('ZH'):
        #         continue
        #
        #     tracks = fpl.real_tracks
        #     flight_time = tracks[-1][0]-tracks[0][0]
        #     print(fpl.from_to, flight_time)
        #
        #     key = flight_time // 300 + 1
        #     flight_times.append(key)
        #
        # plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        # plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
        # plt.hist(flight_times, align='left', density=True)  # 绘制x的密度直方图
        # plt.xlabel('武汉扇区内飞行时间/秒')
        # plt.ylabel('频数')
        # plt.show()
        #
        # break

        # 构建agent set类
        agent_set = AgentSetReal(fpl_list, starts, remain=remain)
        print('------------------------------------------------')
        if remain is None:
            print('>>>', date, len(fpl_list), min(starts), max(starts), len(agent_set.agents))
        else:
            print('>>>', date, len(fpl_list), min(starts), max(starts), len(agent_set.agents), len(remain))

        # agentSet运行
        while not agent_set.all_done():
            agent_set.do_step()

        # 可视化历史轨迹和计划轨迹
        agent_set.visual(save_path='AgentSet_{}'.format(date))
        print('>>> Agent set is visualized!')

        # 可视化流量
        agent_set.flow_visual('flow_{}_{}'.format(alt_limit, date))
        print('>>> Air flow is presented!')

        remain = agent_set.ac_en


if __name__ == '__main__':
    main()
