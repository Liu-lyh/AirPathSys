import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import sortedcontainers


# 树节点
class rrt_treenode():
    def __init__(self, value, parent=None, cost_from_start=0, dis_to_aim=0):
        self.value = value
        self.children = []
        self.cost = cost_from_start
        self.dis = dis_to_aim
        self.parent = parent

    def __lt__(self, other):
        return self.cost + self.dis < other.cost + other.dis

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node.value)
            node = node.parent
        return list(reversed(path_back))


# 优先级队列
class PriorityQueue(object):
    def __init__(self):
        self._queue = sortedcontainers.SortedList([])

    def push(self, node):
        self._queue.add(node)

    def pop(self):
        return self._queue.pop(index=0)

    def size(self):
        return len(self._queue)


def A_star(root, eposilon):
    q = PriorityQueue()
    q.push(root)
    while q.size() != 0:
        node_curr = q.pop()
        for child in node_curr.children:
            if child.dis < eposilon:  # 到达目标
                return child.path()
            q.push(child)
    return []


def avoid_RRT(mymap, start, aim, p_sample, maxlimit, step, eposilon, isshow, Q_save):
    """
    :param mymap: 0-1二值化地图，0表示障碍物
    :param start: 起始点
    :param aim: 目标点
    :param p_sample: 随机采样概率
    :param maxlimit: 最大迭代次数
    :param step: 搜索步长
    :param eposilon: 误差上限
    :param isshow: 是否显示搜索结果
    :param Q_save: 安全距离
    :return: 储存有二维信息的航路点列表
    """

    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    start = np.array(start)
    aim = np.array(aim)
    mapsize = mymap.shape
    dir_save = []  # 安全距离计算
    for i in range(-Q_save, Q_save + 1):
        for j in range(-Q_save, Q_save + 1):
            if distance([i, j], [0, 0]) <= Q_save:
                dir_save.append(np.array([i, j]))

    tree_map = []
    for i in range(mapsize[0]):
        tree_map.append([])
        for j in range(mapsize[1]):
            if mymap[i][j] == 0:
                tree_map[i].append(0)
            else:
                tree_map[i].append(None)
    if tree_map[start[0]][start[1]] is None:
        root = rrt_treenode(start, dis_to_aim=distance(start, aim))
        tree_map[start[0]][start[1]] = root
    else:
        print("出发点处存在障碍物！")
        return []

    # 开始迭代
    p_record = [start]
    count = 0
    isArrive = False
    while count < maxlimit and not isArrive:
        # 获取采样点
        sample = [0, 0]
        while tree_map[sample[0]][sample[1]] is not None:
            if np.random.rand() < p_sample:
                sample = np.array([np.random.randint(0, mapsize[0]),
                                   np.random.randint(0, mapsize[1])])
            else:
                sample = aim

        min_dis = float('inf')
        p_curr = start
        for p_tmp in p_record:
            dis = distance(p_tmp, sample)
            if dis < min_dis:
                p_curr = p_tmp
                min_dis = dis

        # 向采样点前进，获取点p_next
        direction = (sample - p_curr) / np.linalg.norm((sample - p_curr))
        p_next = p_curr + direction * step

        # 判断p_next位置是否可行
        flag = False
        if 0 <= p_next[0] < mapsize[0] and 0 <= p_next[1] < mapsize[1] \
                and tree_map[int(p_next[0])][int(p_next[1])] is None:
            flag = True  # True表示该点可行

        if flag:
            for dir in dir_save:
                p_search = p_next + dir
                if not (0 <= p_search[0] < mapsize[0] and 0 <= p_search[1] < mapsize[1]
                        and tree_map[int(p_search[0])][int(p_search[1])] != 0):
                    flag = False
                    break

        if flag:
            d_next = distance(p_curr, p_next)
            direction = (p_next - p_curr) / np.linalg.norm((p_next - p_curr))
            for d_search in range(1, int(d_next) + 1):
                p_search = p_curr + d_search * direction
                if not (0 <= p_search[0] < mapsize[0] and 0 <= p_search[1] < mapsize[1]
                        and tree_map[int(p_search[0])][int(p_search[1])] != 0):
                    flag = False
                    break

        if flag:
            parenttree = tree_map[int(p_curr[0])][int(p_curr[1])]
            newtree = rrt_treenode(p_next, parent=parenttree,
                                   cost_from_start=parenttree.cost + step,
                                   dis_to_aim=distance(p_next, aim))
            tree_map[int(p_next[0])][int(p_next[1])] = newtree
            parenttree.children.append(newtree)
            p_record.append(p_next)

            plt.plot(p_next[0], p_next[1], 'o', c='blue')
            plt.plot([p_curr[0], p_next[0]], [p_curr[1], p_next[1]], c='green')
            e = newtree.dis
            if e < eposilon:
                isArrive = True

        count += 1

    # 绘制地图
    plt.axis('equal')
    plt.xlim(0, mapsize[0])
    plt.ylim(0, mapsize[1])
    plt.imshow(mymap.T)
    plt.plot(start[0], start[1], 'x', c='y')
    plt.plot(aim[0], aim[1], 'x', c='m')

    # 未到达目标
    if not isArrive:
        if isshow:
            plt.show()
        return []

    # A*搜索最优路径
    path = A_star(root, eposilon)

    # 绘制路径
    if isshow:
        path_plot = np.array(path).T
        plt.plot(path_plot[0], path_plot[1], c='red')
        plt.show()

    return path


if __name__ == "__main__":
    img = cv2.imread('../map_image/scene_fixed.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, mymap = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mymap = cv2.dilate(mymap, None, iterations=1).T
    mymap = cv2.erode(mymap, None, iterations=4) / 255
    path = []

    # 优化后的路径规划
    while len(path) == 0:
        path = avoid_RRT(mymap, [0, 0], [800, 300], 0.1, 5000, 100, 50, True, 10)

    for i in range(len(path)):
        print(i, ': (%.3f, %.3f)' % (path[i][0], path[i][1]))

    if len(path) != 0:
        print('Find!')
    else:
        print('False!')

    # 增加高度信息
    path_for_airsim = []
    for p in path:
        path_for_airsim.append([p[0] / 10, p[1] / 10, -3])  # 单位调整
    path_for_airsim.append([38, 70, -3])
    path_for_airsim = np.array(path_for_airsim)
    np.save('path_for_airsim', path_for_airsim)
