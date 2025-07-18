import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


def a_star(mymap, start, goal):
    """
    A* 算法实现
    :param mymap: 二值化地图，0 表示障碍物，1 表示空闲区域
    :param start: 起点 (x, y)
    :param goal: 终点 (x, y)
    :return: 路径列表
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4邻域
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < mymap.shape[0] and 0 <= neighbor[1] < mymap.shape[1]:
                if mymap[neighbor[0], neighbor[1]] == 0:  # 0 表示障碍物
                    continue
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # 如果没有找到路径


def avoid_A_star(P_start, P_aim, mymap):
    """
    使用 A* 算法进行路径规划
    :param P_start: 起点 (x, y)
    :param P_aim: 终点 (x, y)
    :param mymap: 二值化地图，0 表示障碍物，1 表示空闲区域
    :return: 无
    """
    # 地图尺寸
    size_x = mymap.shape[0]
    size_y = mymap.shape[1]

    # 设置绘图参数
    plt.axis('equal')
    plt.xlim(0, 1200)
    plt.ylim(0, 800)

    # 绘制地图（障碍物和航路点）
    plt.imshow(mymap)
    plt.plot([P_start[0], P_aim[0]], [P_start[1], P_aim[1]], 'o')

    # 运行 A* 算法
    path = a_star(mymap, P_start, P_aim)

    if path:
        print("找到路径:", path)
        # 绘制路径
        for i in range(len(path) - 1):
            plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'r')
        plt.show()
    else:
        print("未找到路径")


if __name__ == "__main__":
    # 读取地图图像并二值化
    img = cv2.imread('../map_image/scene_fixed.png')
    if img is None:
        print("无法加载地图图片，请检查路径是否正确。")
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=1)
    dst = cv2.erode(dst, None, iterations=4) / 255

    # 检查起点和终点是否为障碍物
    P_start = (15, 15)
    P_aim = (600, 500)
    print("起点是否为障碍物:", dst[P_start[0], P_start[1]] == 0)
    print("终点是否为障碍物:", dst[P_aim[0], P_aim[1]] == 0)

    # 运行 A* 算法
    avoid_A_star(P_start=P_start, P_aim=P_aim, mymap=dst)