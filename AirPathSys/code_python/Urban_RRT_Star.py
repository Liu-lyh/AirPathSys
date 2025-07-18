import airsim
import numpy as np
import cv2
import math
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from airsim import Vector3r
rcParams['font.sans-serif'] = ['SimSun']    # 宋体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ---------- FlightVisualizer：显示航路点与实时轨迹 ----------
class FlightVisualizer:
    def __init__(self, client):
        self.client = client
        self.trail_points = []

    def draw_waypoints(self, waypoints, goal_points=[]):
        # 航路点（绿色）
        self.client.simPlotPoints(
            waypoints,
            color_rgba=[0.0, 1.0, 0.0, 1.0],
            size=15,
            is_persistent=True
        )
        # 目标点（黄色）
        self.client.simPlotPoints(
            goal_points,
            color_rgba=[1.0, 1.0, 0.0, 1.0],
            size=30,
            is_persistent=True
        )

    def update_realtime_trail(self, position):
        p = Vector3r(position.x_val, position.y_val, position.z_val)
        self.trail_points.append(p)
        if len(self.trail_points) >= 2:
            self.client.simPlotLineStrip(
                self.trail_points[-2:],
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                thickness=10,
                is_persistent=True
            )


# ---------- 图像增强与去噪 ----------
def enhance_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)
    binar = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
    )
    return cv2.medianBlur(binar, 3)


# ---------- RRT* 算法 ----------
class Node:
    def __init__(self, pt, parent=None, cost=float('inf')):
        self.pt = pt  # (y,x)
        self.parent = parent
        self.cost = cost  # 到达该节点的代价（路径长度）


def collision_free(p1, p2, grid):
    steps = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    # 如果两点重合，直接返回True，避免除零错误
    if steps == 0:
        return True

    for i in range(steps + 1):
        t = i / steps
        y = int(p1[0] * (1 - t) + p2[0] * t)
        x = int(p1[1] * (1 - t) + p2[1] * t)
        if grid[y, x] != 0:
            return False
    return True


def rrt_star_planner(grid, start, goal, max_iter=5000, eps=30, rewire_radius=50):
    rows, cols = grid.shape
    tree = [Node(start, None, 0)]  # 初始树，起点的代价为0
    for _ in range(max_iter):
        # 1. 随机采样一个点
        rnd = (np.random.randint(rows), np.random.randint(cols))

        # 2. 找到当前树中离随机点最近的节点
        nearest_node = min(tree, key=lambda n: (n.pt[0] - rnd[0]) ** 2 + (n.pt[1] - rnd[1]) ** 2)
        dy, dx = rnd[0] - nearest_node.pt[0], rnd[1] - nearest_node.pt[1]
        dist = math.hypot(dy, dx)

        # 3. 如果距离为零（即采样点和最近的节点重合），跳过当前迭代
        if dist == 0:
            continue  # 重新采样，跳过当前计算

        new_pt = (int(nearest_node.pt[0] + dy / dist * eps),
                  int(nearest_node.pt[1] + dx / dist * eps))

        # 4. 如果新节点不越界并且没有碰撞，则添加到树中
        if not (0 <= new_pt[0] < rows and 0 <= new_pt[1] < cols):
            continue
        if not collision_free(nearest_node.pt, new_pt, grid):
            continue

        new_node = Node(new_pt, nearest_node, nearest_node.cost + dist)

        # 5. 在树中选择合适的父节点，并重新连接优化路径
        neighbors = [node for node in tree if
                     math.hypot(new_pt[0] - node.pt[0], new_pt[1] - node.pt[1]) < rewire_radius]
        for neighbor in neighbors:
            if collision_free(neighbor.pt, new_pt, grid):
                new_cost = neighbor.cost + math.hypot(new_pt[0] - neighbor.pt[0], new_pt[1] - neighbor.pt[1])
                if new_cost < new_node.cost:
                    new_node.parent = neighbor
                    new_node.cost = new_cost

        # 将新节点添加到树中
        tree.append(new_node)

        # 6. 检查是否达到了目标
        if math.hypot(new_pt[0] - goal[0], new_pt[1] - goal[1]) < eps:
            path = [goal]
            cur = new_node
            while cur:
                path.append(cur.pt)
                cur = cur.parent
            return path[::-1], tree

    return [], tree  # 如果在最大迭代次数内没有找到路径，返回空路径



# ---------- 插值 & 等距抽点 ----------
def interpolate_path(path, factor=24):
    out = []
    for i in range(len(path) - 1):
        y1, x1 = path[i];
        y2, x2 = path[i + 1]
        for t in np.linspace(0, 1, factor, endpoint=False):
            X = int(x1 * (1 - t) + x2 * t)
            Y = int(y1 * (1 - t) + y2 * t)
            out.append((Y, X))
    out.append(path[-1])
    return out


def extract_waypoints_by_distance(path, num_points):
    dists = [0]
    for i in range(1, len(path)):
        y1, x1 = path[i - 1];
        y2, x2 = path[i]
        dists.append(dists[-1] + math.hypot(y2 - y1, x2 - x1))
    total = dists[-1]
    if total == 0:
        return [path[0]] * num_points
    step = total / (num_points - 1)
    targets = [i * step for i in range(num_points)]
    waypts = [];
    idx = 0
    for td in targets:
        while idx < len(dists) - 1 and dists[idx + 1] < td:
            idx += 1
        if idx == len(dists) - 1:
            waypts.append(path[idx])
        else:
            r = (td - dists[idx]) / (dists[idx + 1] - dists[idx])
            y = path[idx][0] * (1 - r) + path[idx + 1][0] * r
            x = path[idx][1] * (1 - r) + path[idx + 1][1] * r
            waypts.append((y, x))
    return waypts


# ---------- RRT 树 + 最终路径可视化 ----------
def visualize_rrt_and_path(grid, nodes, final_path, start, goal, viz, w, h, goal_points=[]):
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='gray')
    # 1) 画骨架
    for node in nodes:
        if node.parent is not None:
            y1, x1 = node.pt
            y2, x2 = node.parent.pt
            plt.plot([x1, x2], [y1, y2],
                     '-', color='gray',
                     linewidth=1.5, alpha=0.6)
    # 2) 画最终路径
    ys, xs = zip(*final_path)
    plt.plot(xs, ys, '-', color='cyan',
             linewidth=3.0, label='最终RRT*路径')
    # 3) 标记起终点
    plt.scatter([start[1]], [start[0]],
                c='green', s=100, marker='o', label='起点')
    plt.scatter([goal[1]], [goal[0]],
                c='red', s=100, marker='X', label='目标')

    # 标注所有目标点
    for i, goal_point in enumerate(goal_points):
        if i == len(goal_points) - 1:
            plt.scatter(goal_point[1], goal_point[0], c='red', s=100, marker='x', label="Goal")
        else:
            # 修改为黄色星形
            plt.scatter(goal_point[1], goal_point[0], c='yellow', s=100, marker='*',
                        label="中间目标" if i == 0 else "")

    plt.gca().invert_yaxis()
    plt.legend(loc='lower right')
    plt.title("RRT* 树 + 最终路径")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 保存图像
    output_path = './rrt_star_path_with_all_goals.png'
    plt.savefig(output_path)
    plt.show()

    # 在Airsim中绘制扩展点（灰色）
    tree3d = [
        Vector3r((x - w // 2) * 0.3, -(y - h // 2) * 0.3, -5)
        for (y, x) in final_path
    ]
    viz.client.simPlotPoints(
        tree3d, color_rgba=[0.5, 0.5, 0.5, 1.0], size=10, is_persistent=True
    )

    return output_path  # 返回保存的路径


# ---------- 航路点追踪飞行 ----------
def move_by_path_tracking(client, waypoints, goal_points, Va, z_val=-5, epsilon=3.0, dt=0.1):
    viz = FlightVisualizer(client)
    viz.draw_waypoints(waypoints, goal_points)
    goal_idx = 0
    total_wp = len(waypoints)
    for i, wp in enumerate(waypoints):
        print(f"→ 航路点 {i + 1}/{total_wp}: x={wp.x_val:.2f}, y={wp.y_val:.2f}, z={wp.z_val:.2f}")
        if goal_idx < len(goal_points):
            g = goal_points[goal_idx]
            if math.hypot(wp.x_val - g.x_val, wp.y_val - g.y_val) <= epsilon:
                print(f"🚁 已到达目标点 {goal_idx + 1}")
                goal_idx += 1
        while True:
            st = client.simGetGroundTruthKinematics(vehicle_name="Drone1")
            pos = st.position
            viz.update_realtime_trail(pos)
            dx, dy = wp.x_val - pos.x_val, wp.y_val - pos.y_val
            if math.hypot(dx, dy) <= epsilon:
                break
            hd = math.atan2(dy, dx)
            client.moveByVelocityZAsync(
                Va * math.cos(hd), Va * math.sin(hd), z_val,
                duration=dt, vehicle_name="Drone1"
            ).join()
    return viz.trail_points


# ---------- 渐进降落 ----------
def gradual_landing(client, start_h, target_height=-4.0, landing_speed=0.2):
    cz = start_h
    while cz > target_height:
        cz -= landing_speed
        client.moveToZAsync(cz, velocity=landing_speed).join()
        time.sleep(0.2)
    while abs(client.simGetGroundTruthKinematics(vehicle_name="Drone1")
                      .position.z_val - target_height) > 0.1:
        time.sleep(1)


# ---------- 3D 可视化飞行轨迹（English labels） ----------
def plot_3d_flight_model(trajectory, goals):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*trajectory)
    ax.plot(xs, ys, zs, c='red', lw=2, label='Flight Trajectory')
    gx, gy, gz = zip(*goals)
    ax.scatter(gx, gy, gz, c='blue', s=80, marker='*', label='目标点')
    ax.set_xlabel('X (m)');
    ax.set_ylabel('Y (m)');
    ax.set_zlabel('Z (m)')
    ax.set_title('3D 多目标飞行路径');
    ax.legend()
    plt.show()


# ---------- 主流程 ----------
def main():
    print("🔧 地图预处理中...")
    grid = (enhance_image("../map_image/scene.png") == 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    grid_inf = (cv2.dilate((grid * 255).astype(np.uint8), kernel, 1) > 128).astype(np.uint8)

    plt.figure(figsize=(6, 4))
    plt.imshow(grid_inf, cmap='gray')
    plt.title("膨胀栅格图");
    plt.show()

    h, w = grid_inf.shape
    start_px = (h // 2, w // 2)
    goals_px = [(650, 300), (370, 600), (120, 800)]
    all_pts = [start_px] + goals_px

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")

    total_path = []
    all_nodes = []

    # 使用 RRT* 算法进行路径规划
    for a, b in zip(all_pts, all_pts[1:]):
        print(f"🧠 RRT* 规划：{a} → {b}")
        seg, nodes = rrt_star_planner(grid_inf, a, b, max_iter=3000, eps=30)  # 只解包两个值
        if not seg:
            print("❌ RRT* 未能找到路径，退出")
            return
        total_path += seg
        all_nodes += nodes

    # 2D 可视化：RRT* 树 + 最终路径
    print("🔍 绘制 RRT* 树骨架 & 最终路径...")
    viz = FlightVisualizer(client)
    visualize_rrt_and_path(grid_inf,
                           all_nodes,
                           total_path,
                           start_px,
                           goals_px[-1], viz, w, h, goals_px)

    # 生成真正飞行航路点
    interp = interpolate_path(total_path, factor=36)
    pixs = extract_waypoints_by_distance(interp, 60)
    flight_wps = [
        Vector3r((x - w // 2) * 0.3, -(y - h // 2) * 0.3, -5)
        for y, x in pixs
    ]
    goal_wps = [
        Vector3r((gx - w // 2) * 0.3, -(gy - h // 2) * 0.3, -5)
        for gy, gx in goals_px
    ]

    # 起飞前可视化航路点
    viz.draw_waypoints(flight_wps, goal_wps)

    # 起飞
    print("🛫 起飞中，请稍候...")
    client.takeoffAsync().join()
    client.moveToZAsync(-5, 1).join()
    time.sleep(1)
    print("✈️ 起飞完毕，开始航路点追踪")

    # 飞行追踪
    print("🚁 开始飞行…")
    trail = move_by_path_tracking(client, flight_wps, goal_wps,
                                  Va=3, z_val=-5)

    # 缓慢降落
    print("✅ 到达终点 降落中…")
    gradual_landing(client, start_h=-5)

    # 3D 轨迹
    print("🔍 绘制三维飞行轨迹…")
    traj = [(p.x_val, p.y_val, p.z_val) for p in trail]
    goals = [(g.x_val, g.y_val, g.z_val) for g in goal_wps]
    plot_3d_flight_model(traj, goals)


if __name__ == "__main__":
    main()

