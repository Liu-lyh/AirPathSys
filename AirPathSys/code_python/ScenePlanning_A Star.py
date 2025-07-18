import airsim
import numpy as np
import cv2
import heapq
import math
import time
import matplotlib.pyplot as plt
from airsim import Vector3r


# ---------- FlightVisualizer：显示航路点与实时轨迹 ----------
class FlightVisualizer:
    def __init__(self, client):
        self.client = client
        self.trail_points = []

    def draw_waypoints(self, waypoints):
        self.client.simPlotPoints(
            waypoints,
            color_rgba=[0.0, 1.0, 0.0, 1.0],  # 绿色标记航路点
            size=15,
            is_persistent=True
        )

    def update_realtime_trail(self, position):
        self.trail_points.append(Vector3r(position.x_val, position.y_val, position.z_val))
        if len(self.trail_points) >= 2:
            self.client.simPlotLineStrip(
                self.trail_points[-2:],  # 绘制前两个点的轨迹
                color_rgba=[1.0, 0.0, 0.0, 1.0],  # 红色飞行轨迹
                thickness=10,
                is_persistent=True
            )


# ---------- 渐进降落 ----------
def gradual_landing(client, start_height, target_height=-4.0, landing_speed=0.2):
    """
    渐进式安全降落
    :param start_height: 当前高度（AirSim中z为负值）
    :param target_height: 目标高度（米）
    :param landing_speed: 下降速度（m/s）
    """
    current_z = start_height
    print(f"\n开始渐进降落（初始高度：{-current_z:.1f}m）")

    # 缓慢下降直到到达目标高度
    while current_z > target_height:
        current_z -= landing_speed
        client.moveToZAsync(current_z, velocity=landing_speed).join()  # 确保每次命令执行
        print(f"当前高度：{-current_z:.1f}m")

        time.sleep(0.2)  # 稍微延迟，确保每次命令的平稳执行

    # 到达目标高度后停止并打印一次
    print(f"💡 已经到达目标高度：{-current_z:.1f} 米，开始悬停...")

    # 在目标高度悬停
    while True:
        state = client.simGetGroundTruthKinematics(vehicle_name="Drone1")
        pos = state.position
        if abs(pos.z_val - target_height) < 0.1:  # 如果高度差小于0.1米，认为到达目标高度
            print(f"✅ 悬停在 {target_height} 米，保持当前高度...")
            break
        time.sleep(1)  # 每秒检查一次高度


# ---------- 航路点追踪飞行 ----------
def move_by_path_tracking(client, waypoints, Va, z_val=-5, epsilon=1.0, dt=0.1):
    vehicle_name = "Drone1"
    viz = FlightVisualizer(client)
    viz.draw_waypoints(waypoints)  # 绘制航路点
    for i, target in enumerate(waypoints):
        print(f"\n追踪航路点 {i + 1}/{len(waypoints)}: ({target.x_val:.2f}, {target.y_val:.2f}, {target.z_val:.2f})")
        while True:
            state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
            pos = state.position
            viz.update_realtime_trail(pos)  # 更新飞行轨迹
            dx = target.x_val - pos.x_val
            dy = target.y_val - pos.y_val
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= epsilon:
                print(f"✓ 到达航路点 {i + 1}")
                break
            heading = math.atan2(dy, dx)
            Vx = Va * math.cos(heading)
            Vy = Va * math.sin(heading)
            client.moveByVelocityZAsync(vx=Vx, vy=Vy, z=z_val, duration=dt, vehicle_name=vehicle_name).join()

    # 到达最后一个航路点后，悬停在目标高度
    print("💡 到达最后一个航路点，开始悬停在目标高度...")
    state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    pos = state.position

    # 让无人机在最后一个航路点悬停
    print(f"当前高度：{pos.z_val} 米，开始悬停...")
    gradual_landing(client, start_height=pos.z_val, target_height=-4.0, landing_speed=0.2)


# ---------- 图像二值化 ----------
def optimize_thresholding(image_path):
    """
    图像二值化，使用自适应阈值法
    :param image_path: 图像路径
    :return: 二值化后的图像
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)  # 自适应阈值
    return adaptive


# ---------- A* 路径规划 ----------
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny = current[0] + dx
            nx = current[1] + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                if grid[ny, nx] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                    g_score[(ny, nx)] = tentative_g
                    f = tentative_g + abs(goal[0] - ny) + abs(goal[1] - nx)
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


# ---------- 可视化路径 ----------
def visualize_path(grid, path, start, goal):
    vis = grid.copy().astype(np.float32)
    for x, y in path:
        vis[int(x), int(y)] = 0.5
    vis[start] = 0.7
    vis[goal] = 0.9
    plt.imshow(vis, cmap='gray')
    plt.title("Path on Grid Map")
    plt.savefig("../map_image/path_visualization.png")
    plt.close()


# ---------- 等距航路点抽取 ----------
def extract_waypoints_by_distance(path, num_points):
    distances = [0]
    for i in range(1, len(path)):
        y1, x1 = path[i - 1]
        y2, x2 = path[i]
        d = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(distances[-1] + d)
    total_distance = distances[-1]
    if total_distance == 0:
        return [path[0]] * num_points

    step = total_distance / (num_points - 1)
    target_ds = [i * step for i in range(num_points)]

    waypoints = []
    idx = 0
    for td in target_ds:
        while idx < len(distances) - 1 and distances[idx + 1] < td:
            idx += 1
        if idx == len(distances) - 1:
            waypoints.append(path[idx])
        else:
            ratio = (td - distances[idx]) / (distances[idx + 1] - distances[idx])
            y = path[idx][0] * (1 - ratio) + path[idx + 1][0] * ratio
            x = path[idx][1] * (1 - ratio) + path[idx + 1][1] * ratio
            waypoints.append((y, x))
    return waypoints


# 主程序：路径规划与飞行
def main():
    img_rgb = cv2.imread("../map_image/scene.png")

    # 图像锐化
    print("🔧 图像锐化中...")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_rgb, -1, kernel)
    cv2.imwrite("../map_image/scene_sharpened.png", sharpened)

    # 图像二值化
    print("🧪 图像二值化中...")
    binary = optimize_thresholding("../map_image/scene_sharpened.png")
    cv2.imwrite("../map_image/scene_fixed.png", binary)
    grid = (binary == 0).astype(np.uint8)

    # 对障碍物膨胀（增大障碍范围）
    obstacle_img = (grid * 255).astype(np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)
    obstacle_inflated = cv2.dilate(obstacle_img, kernel_dilate, iterations=1)
    grid_inflated = (obstacle_inflated > 128).astype(np.uint8)

    # A* 路径规划
    h, w = grid_inflated.shape
    scale = 0.3  # 每像素对应的米数
    start_px = (h // 2, w // 2)  # 图像中心，代表 (0,0,-5)
    goal_px = (102, 400)  # 红点像素（终点）

    print("🧠 A* 路径规划中...")
    path = a_star(grid_inflated, start_px, goal_px)
    if not path:
        print("❌ 无法找到路径")
        exit()

    path = interpolate_path(path, factor=24)

    # 可视化路径
    visualize_path(grid_inflated, path, start_px, goal_px)

    # 等距采样航路点
    num_waypoints = 12  # 包括起点和终点
    waypoints_px = extract_waypoints_by_distance(path, num_waypoints)

    # 将图像像素坐标转换为世界坐标
    flight_waypoints = []
    for (y, x) in waypoints_px:
        wx = (x - w // 2) * scale
        wy = -(y - h // 2) * scale
        wz = -5
        flight_waypoints.append(airsim.Vector3r(wx, wy, wz))

    # 创建AirSim客户端并开始飞行
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name="Drone1")
    client.armDisarm(True, vehicle_name="Drone1")

    # 追踪航路点飞行
    print("🚁 航路点追踪飞行开始...")
    move_by_path_tracking(client, flight_waypoints, Va=3, z_val=-5, epsilon=1.0, dt=0.1)

    # 飞行完成后，最后一个航路点悬停
    print("✅ 飞行任务完成，系统继续运行并悬停在最后一个航路点。")
    # 保持悬停状态，不退出系统


if __name__ == "__main__":
    main()
