"""
python_avoid_APF.py
人工势场法避障的python实现
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


def avoid_APF(P_start, V_start, P_aim, mymap, Kg=0.5, kr=20,
              Q_search=20, epsilon=2, Vl=2, Ul=2, dt=0.2, draw_ontime=False):
    """
    :param P_start: 初始位置
    :param V_start: 初始速度
    :param P_aim: 目标点
    :param mymap: 储存有障碍物信息的0-1二值化地图，0表示障碍物
    :param Kg: 避障控制器参数（引力）
    :param kr: 避障控制器参数（斥力）
    :param Q_search: 搜索障碍物距离阈值
    :param epsilon: 误差上限
    :param Vl: 速率上限
    :param Ul: 控制器输出上限
    :param dt: 迭代时间
    :param draw_ontime: 是否动态绘图
    :return: 无
    """

    def distance(A, B):
        return math.hypot(A[0] - B[0], A[1] - B[1])

    def myatan(a, b):
        # 改动：当 a=b=0 时返回 0.0，避免 NoneType
        if a == 0 and b == 0:
            return 0.0
        if a == 0:
            return math.pi/2 if b > 0 else math.pi*3/2
        if b == 0:
            return 0.0 if a > 0 else -math.pi
        if b > 0:
            return math.atan(b / a) if a > 0 else (math.atan(b / a) + math.pi)
        # b < 0
        return (math.atan(b / a + 2*math.pi) if a > 0 else math.atan(b / a) + math.pi)

    def isClockwise(a, b):
        # 改动：先拦截 None
        if a is None or b is None:
            return False
        da = b - a
        # 如果夹角在 (0, π) 或者 小于 -π，都算逆时针（不顺时针）
        if 0 < da < math.pi or -2*math.pi < da < -math.pi:
            return False
        return True

    # 读取初始状态
    P_start = np.array(P_start)        # 初始位置
    V_start = np.array(V_start)        # 初始速度
    pos_record = [P_start]
    size_x, size_y = mymap.shape

    # 设置绘图参数
    plt.axis('equal')
    plt.xlim(0, 1200)
    plt.ylim(0, 800)
    plt.imshow(mymap)
    plt.plot([P_start[0], P_aim[0]], [P_start[1], P_aim[1]], 'o')

    direction_search = np.array([-2, -1, 0, 1, 2]) * math.pi/4

    P_curr = P_start.copy()
    V_curr = V_start.copy()
    ob_flag = False
    pos_num = 0

    while distance(P_curr, P_aim) > epsilon:
        angle_curr = myatan(V_curr[0], V_curr[1])
        Frep = np.zeros(2)

        # 计算斥力
        for dir in direction_search:
            angle_search = angle_curr + dir
            for d in range(Q_search):
                x = int(P_curr[0] + d * math.sin(angle_search))
                y = int(P_curr[1] + d * math.cos(angle_search))
                if not (0 <= x < size_x and 0 <= y < size_y and mymap[x, y] == 1):
                    d_search = distance(P_curr, [x, y])
                    Frep += kr * (1/d_search - 1/Q_search) / (d_search**3) * \
                            (P_curr - np.array([x, y])) * (distance([x, y], P_aim)**2)
                    break

        # 计算引力
        Fatt = -Kg * (P_curr - P_aim)

        # 局部极小值检测与处理
        if pos_num >= 1:
            p0 = pos_record[pos_num - 1]
            p1 = pos_record[pos_num - 2]
            Vra = (distance(p0, P_aim) - distance(p1, P_aim)) / dt
            if abs(Vra) < 0.6 * Vl:
                if not ob_flag:
                    angle_g = myatan(Fatt[0], Fatt[1])
                    angle_r = myatan(Frep[0], Frep[1])
                    theta = (15 * math.pi/180) if isClockwise(angle_g, angle_r) else (-15 * math.pi/180)
                    ob_flag = True
                Frep = np.array([
                    math.cos(theta)*Frep[0] - math.sin(theta)*Frep[1],
                    math.sin(theta)*Frep[0] + math.cos(theta)*Frep[1]
                ])
            else:
                ob_flag = False

            # 改进引力
            l = Vl
            Kv = 3 * l / (2 * l + abs(Vra))
            Kd = 15 * math.exp(-(distance(P_curr, P_aim) - 3)**2 / 2) + 1
            Ke = 3
            Fatt = Kv * Kd * Ke * Fatt

        # 合力、限幅及更新
        U = Fatt + Frep
        if np.linalg.norm(U, ord=np.inf) > Ul:
            U = Ul * U / np.linalg.norm(U, ord=np.inf)
        V_curr = V_curr + U * dt
        if np.linalg.norm(V_curr) > Vl:
            V_curr = Vl * V_curr / np.linalg.norm(V_curr)
        P_curr = P_curr + V_curr * dt

        print(P_curr, V_curr, distance(P_curr, P_aim))
        pos_record.append(P_curr.copy())
        pos_num += 1

        if draw_ontime:
            plt.plot(
                [pos_record[pos_num-1][0], pos_record[pos_num][0]],
                [pos_record[pos_num-1][1], pos_record[pos_num][1]], 'r'
            )
            plt.pause(0.0001)
            plt.ioff()

    # 绘制最终轨迹
    traj = np.array(pos_record).T
    plt.plot(traj[0], traj[1], '--')
    plt.show()


if __name__ == "__main__":
    # 读取地图图像并二值化
    img = cv2.imread('../map_image/scene_fixed.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=1)
    dst = cv2.erode(dst, None, iterations=4) / 255
    avoid_APF(P_start=[15, 15], V_start=[0, 2], P_aim=[600, 500], Q_search=15, mymap=dst)
