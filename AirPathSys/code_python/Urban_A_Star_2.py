import airsim
import numpy as np
import cv2
import heapq
import math
import time
import matplotlib.pyplot as plt
from airsim import Vector3r
from mpl_toolkits.mplot3d import Axes3D


# ---------- FlightVisualizer：显示航路点与实时轨迹 ----------
class FlightVisualizer:
    def __init__(self, client):
        self.client = client
        self.trail_points = []

    def draw_waypoints(self, waypoints, goal_points=[]):
        # 绘制航路点
        self.client.simPlotPoints(
            waypoints,
            color_rgba=[0.0, 1.0, 0.0, 1.0],  # 绿色标记航路点
            size=15,
            is_persistent=True
        )

        # 绘制目标点（大一点，黄色）
        self.client.simPlotPoints(
            goal_points,
            color_rgba=[1.0, 1.0, 0.0, 1.0],  # 黄色标记目标点
            size=30,  # 更大的目标点
            is_persistent=True
        )

    def update_realtime_trail(self, position):
        self.trail_points.append(Vector3r(position.x_val, position.y_val, position.z_val))
        if len(self.trail_points) >= 2:
            self.client.simPlotLineStrip(
                self.trail_points[-2:],
                color_rgba=[1.0, 0.0, 0.0, 1.0],  # 红色飞行轨迹
                thickness=10,
                is_persistent=True
            )


# ---------- 图像增强与去噪 ----------
def enhance_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    denoised = cv2.medianBlur(binary, 3)
    return denoised


# ---------- A* 路径规划 ----------
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    f_score = {start: h(start)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # 重建路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = current[0] + dx, current[1] + dy
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] == 0:
                tentative_g = g_score[current] + 1
                if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                    g_score[(ny, nx)] = tentative_g
                    f = tentative_g + h((ny, nx))
                    heapq.heappush(open_set, (f, (ny, nx)))
                    came_from[(ny, nx)] = current
    return []


# ---------- 插值 ----------
def interpolate_path(path, factor=24):
    interpolated = []
    for i in range(len(path) - 1):
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        for t in np.linspace(0, 1, factor, endpoint=False):
            X = int(x1 * (1 - t) + x2 * t)
            Y = int(y1 * (1 - t) + y2 * t)
            interpolated.append((Y, X))
    interpolated.append(path[-1])
    return interpolated


# ---------- 可视化路径（增强版，带标注并保存） ----------
def visualize_path(grid, path, start, goals, filename='path_on_grid_map.png'):
    """
    grid:      二值化后的网格（0/1）
    path:      像素坐标列表 [(y0,x0), (y1,x1), ...]
    start:     起点 (y,x)
    goals:     目标点列表 [(y1,x1), (y2,x2), ...]
    filename:  保存图像的文件名
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='gray')

    # 1) 画路径
    ys, xs = zip(*path)
    plt.plot(xs, ys, '-', color='cyan', linewidth=2, label='Path')

    # 2) 标注起点
    sy, sx = start
    plt.scatter(sx, sy, c='green', s=100, marker='o', label='Start')

    # 3) 标注所有目标（倒数最后一个为最终目标，其余为中间目标）
    for i, (gy, gx) in enumerate(goals):
        if i < len(goals) - 1:
            plt.scatter(gx, gy, c='yellow', s=150, marker='*',
                        label='Intermediate Goal' if i == 0 else None)
        else:
            plt.scatter(gx, gy, c='red', s=150, marker='X', label='Final Goal')

    plt.title("Path on Grid Map")
    plt.legend(loc='upper right')
    plt.axis('off')  # 如果不想显示坐标轴可以取消注释

    # 4) 保存图片
    plt.savefig(filename, bbox_inches='tight', dpi=200)

    # 5) 显示
    plt.show()


# ---------- 等距航路点抽取 ----------
def extract_waypoints_by_distance(path, num_points):
    distances = [0]
    for i in range(1, len(path)):
        y1, x1 = path[i - 1]
        y2, x2 = path[i]
        distances.append(distances[-1] + math.hypot(y2 - y1, x2 - x1))
    total = distances[-1]
    if total == 0:
        return [path[0]] * num_points
    step = total / (num_points - 1)
    targets = [i * step for i in range(num_points)]
    waypoints = []
    idx = 0
    for td in targets:
        while idx < len(distances) - 1 and distances[idx + 1] < td:
            idx += 1
        if idx == len(distances) - 1:
            waypoints.append(path[idx])
        else:
            r = (td - distances[idx]) / (distances[idx + 1] - distances[idx])
            y = path[idx][0] * (1 - r) + path[idx + 1][0] * r
            x = path[idx][1] * (1 - r) + path[idx + 1][1] * r
            waypoints.append((y, x))
    return waypoints


# ---------- 航路点追踪飞行 ----------
def move_by_path_tracking(client, waypoints, goal_points, Va, z_val=-5, epsilon=1.0, dt=0.1):
    vehicle_name = "Drone1"
    viz = FlightVisualizer(client)
    viz.draw_waypoints(waypoints, goal_points)  # 绘制航路点和目标点
    total_waypoints = len(waypoints)

    # 当前目标点的索引
    goal_idx = 0

    for i, wp in enumerate(waypoints):
        print(f"→ 飞行到航路点 {i + 1}/{total_waypoints}: x={wp.x_val:.2f}, y={wp.y_val:.2f}, z={wp.z_val:.2f}")

        # 判断是否到达目标点
        if goal_idx < len(goal_points):
            goal = goal_points[goal_idx]
            # 检查目标点是否已经到达
            if math.hypot(wp.x_val - goal.x_val, wp.y_val - goal.y_val) <= epsilon:
                print(f"🚁 到达目标点 {goal_idx + 1}")
                goal_idx += 1  # 更新目标点索引

        while True:
            state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
            pos = state.position
            viz.update_realtime_trail(pos)

            dx, dy = wp.x_val - pos.x_val, wp.y_val - pos.y_val
            if math.hypot(dx, dy) <= epsilon:
                break

            heading = math.atan2(dy, dx)
            client.moveByVelocityZAsync(
                Va * math.cos(heading),
                Va * math.sin(heading),
                z_val,
                duration=dt,
                vehicle_name=vehicle_name
            ).join()


# ---------- 渐进降落 ----------
def gradual_landing(client, start_height, target_height=-4.0, landing_speed=0.2):
    current_z = start_height
    while current_z > target_height:
        current_z -= landing_speed
        client.moveToZAsync(current_z, velocity=landing_speed).join()
        time.sleep(0.2)
    while True:
        state = client.simGetGroundTruthKinematics(vehicle_name="Drone1")
        if abs(state.position.z_val - target_height) < 0.1:
            break
        time.sleep(1)


# ---------- 绘制3D轨迹 ----------
def plot_trajectory(data, goal_points):
    x, y, z = zip(*data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制飞行轨迹
    ax.plot(x, y, z, color='red', label='Flight Trajectory', linewidth=2)

    # 绘制目标点
    for goal in goal_points:
        ax.scatter([goal[0]], [goal[1]], [goal[2]], c='blue', s=100, marker='*', label='Goal Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Multi-Goal Flight Path')
    ax.legend()
    plt.show()


def main():
    # 1. 图像预处理
    print("🔧 图像增强与去噪中...")
    enhanced = enhance_image("../map_image/scene.png")

    print("🧪 二值化 & 栅格化中...")
    grid = (enhanced == 0).astype(np.uint8)

    print("🔧 障碍膨胀中...")
    obs = (grid * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    inflated = cv2.dilate(obs, kernel, iterations=1)
    grid_inf = (inflated > 128).astype(np.uint8)

    plt.imshow(grid_inf, cmap='gray')
    plt.title("Inflated Grid")
    plt.show()

    # 2. 起点 & 目标点列表（多个目标）
    h, w = grid_inf.shape
    start_px = (h // 2, w // 2)
    goals_px = [(120, 800), (650, 300), (370, 600)]

    # 合并起点和目标点，进行一次路径规划
    all_points = [start_px] + goals_px  # 将起点和所有目标点合并

    # 3. 依次规划、飞行
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")

    total_path_px = []

    # 一次性规划路径：从起点到所有目标点
    for i in range(len(all_points) - 1):
        print(f"🧠 规划路径：{all_points[i]} → {all_points[i + 1]} …")
        path_px = a_star(grid_inf, all_points[i], all_points[i + 1])
        if not path_px:
            print(f"❌ 从 {all_points[i]} 到 {all_points[i + 1]} 路径不存在，退出")
            return
        total_path_px.extend(path_px)  # 合并所有路径

    # 插值生成更细的路径，这里将航路点数量修改为 36
    interp_px = interpolate_path(total_path_px, factor=36)
    visualize_path(grid_inf,  # 膨胀后的网格
                   interp_px,  # 插值后的完整路径
                   start_px,  # 起点
                   goals_px)  # 目标点列表（可以包含多个中间和最终目标）

    # 抽取航路点 & 转三维
    pix_waypoints = extract_waypoints_by_distance(interp_px, num_points=36)  # 每段路径生成36个航路点
    flight_waypoints = []
    for y, x in pix_waypoints:
        wx = (x - w // 2) * 0.3
        wy = -(y - h // 2) * 0.3
        wz = -5
        flight_waypoints.append(Vector3r(wx, wy, wz))

    # 目标点列表（仍然需要为所有目标点创建目标点）
    all_goal_points = []
    for goal_px in goals_px:
        goal_points = Vector3r((goal_px[1] - w // 2) * 0.3, -(goal_px[0] - h // 2) * 0.3, -5)
        all_goal_points.append(goal_points)

    # 传递目标点给绘制方法
    viz = FlightVisualizer(client)
    viz.draw_waypoints(flight_waypoints, all_goal_points)  # 绘制所有目标点

    # 飞行：根据航路点追踪飞行
    print("🚁 飞行这一段 …")
    move_by_path_tracking(client, flight_waypoints, all_goal_points, Va=3, z_val=-5)  # 传递 goal_points

    # 4. 最后降落
    print("✅ 全部目标到达，开始降落")
    gradual_landing(client, start_height=-5)

    # 5. 绘制3D轨迹
    trajectory_data = [(wp.x_val, wp.y_val, wp.z_val) for wp in flight_waypoints]
    plot_trajectory(trajectory_data, all_goal_points)


if __name__ == "__main__":
    main()
