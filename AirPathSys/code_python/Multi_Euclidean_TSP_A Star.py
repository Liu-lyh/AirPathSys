import airsim
import numpy as np
import cv2
import heapq
import math
import time
import matplotlib.pyplot as plt
from airsim import Vector3r


class FlightVisualizer:
    def __init__(self, client):
        self.client = client
        self.trail_points = []

    def draw_waypoints(self, waypoints, special_points=None):
        self.client.simPlotPoints(
            waypoints,
            color_rgba=[0.0, 1.0, 0.0, 1.0],  # 绿色
            size=15,
            is_persistent=True
        )
        if special_points:
            self.client.simPlotPoints(
                special_points,
                color_rgba=[1.0, 1.0, 0.0, 1.0],  # 黄色
                size=25,
                is_persistent=True
            )

    def update_realtime_trail(self, position):
        self.trail_points.append(Vector3r(position.x_val, position.y_val, position.z_val))
        if len(self.trail_points) >= 2:
            self.client.simPlotLineStrip(
                self.trail_points[-2:],
                color_rgba=[1.0, 0.0, 0.0, 1.0],  # 红色轨迹
                thickness=10,
                is_persistent=True
            )


def gradual_landing(client, start_height, target_height=-4.0, landing_speed=0.2):
    current_z = start_height
    print(f"\n开始渐进降落（初始高度：{-current_z:.1f}m）")
    while current_z > target_height:
        current_z -= landing_speed
        client.moveToZAsync(current_z, velocity=landing_speed).join()
        print(f"当前高度：{-current_z:.1f}m")
        time.sleep(0.2)
    print(f"💡 已经到达目标高度：{-current_z:.1f} 米，开始悬停...")
    while True:
        state = client.simGetGroundTruthKinematics(vehicle_name="Drone1")
        pos = state.position
        if abs(pos.z_val - target_height) < 0.1:
            print(f"✅ 悬停在 {target_height} 米，保持当前高度...")
            break
        time.sleep(1)


def move_by_path_tracking(client, waypoints, special_points, Va, z_val=-5, epsilon=1.0, dt=0.1):
    vehicle_name = "Drone1"
    viz = FlightVisualizer(client)
    viz.draw_waypoints(waypoints, special_points=special_points)
    for i, target in enumerate(waypoints):
        print(f"\n追踪航路点 {i + 1}/{len(waypoints)}: ({target.x_val:.2f}, {target.y_val:.2f}, {target.z_val:.2f})")
        while True:
            state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
            pos = state.position
            viz.update_realtime_trail(pos)
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

    print("💡 到达最后一个航路点，开始悬停在目标高度...")
    state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    pos = state.position
    print(f"当前高度：{pos.z_val} 米，开始悬停...")
    gradual_landing(client, start_height=pos.z_val, target_height=-4.0, landing_speed=0.2)


def optimize_thresholding(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return adaptive

def remove_consecutive_duplicates(path):
    """移除路径中连续重复的点，避免插值计算除以0"""
    if not path:
        return path
    result = [path[0]]
    for pt in path[1:]:
        if pt != result[-1]:
            result.append(pt)
    return result


def extract_waypoints_by_distance(path, num_points):
    """根据路径距离间隔等分提取若干个航路点，避免距离为0除错"""
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
            denom = distances[idx + 1] - distances[idx]
            if denom == 0:
                waypoints.append(path[idx])  # 连续点重合时直接取当前点
            else:
                ratio = (td - distances[idx]) / denom
                y = path[idx][0] * (1 - ratio) + path[idx + 1][0] * ratio
                x = path[idx][1] * (1 - ratio) + path[idx + 1][1] * ratio
                waypoints.append((y, x))

    return waypoints

from scipy.spatial.distance import cdist

def optimize_goal_sequence(goal_list_px, start_px):
    coords = [start_px] + goal_list_px
    dist_matrix = cdist(coords, coords, metric='euclidean')
    n = len(goal_list_px)
    visited = [False] * (n + 1)
    visited[0] = True
    sequence = []
    current = 0
    for _ in range(n):
        nearest = None
        min_dist = float('inf')
        for j in range(1, n + 1):
            if not visited[j] and dist_matrix[current][j] < min_dist:
                min_dist = dist_matrix[current][j]
                nearest = j
        sequence.append(goal_list_px[nearest - 1])
        visited[nearest] = True
        current = nearest
    return sequence


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
                    f = tentative_g + math.sqrt((goal[0] - ny) ** 2 + (goal[1] - nx) ** 2)  # 欧几里得距离
                    heapq.heappush(open_set, (f, (ny, nx)))
                    came_from[(ny, nx)] = current
    return []


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


def visualize_path(grid, path, start, goal):
    vis = grid.copy().astype(np.float32)
    for x, y in path:
        vis[int(x), int(y)] = 0.5
    vis[start] = 0.7
    vis[goal] = 0.9
    plt.imshow(vis, cmap='gray')
    plt.title("Multi-Goal Path on Grid Map")
    plt.savefig("../map_image/path_visualization.png")
    plt.close()


def main():
    img_rgb = cv2.imread("../map_image/scene.png")
    print("🔧 图像锐化中...")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_rgb, -1, kernel)
    cv2.imwrite("../map_image/scene_sharpened.png", sharpened)

    print("🧪 图像二值化中...")
    binary = optimize_thresholding("../map_image/scene_sharpened.png")
    cv2.imwrite("../map_image/scene_fixed.png", binary)
    grid = (binary == 0).astype(np.uint8)

    obstacle_img = (grid * 255).astype(np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)
    obstacle_inflated = cv2.dilate(obstacle_img, kernel_dilate, iterations=1)
    grid_inflated = (obstacle_inflated > 128).astype(np.uint8)

    h, w = grid_inflated.shape
    scale = 0.3
    start_px = (h // 2, w // 2)

    goal_list_px = [
        (102, 900),
        (80, 500),
        (500, 400),
    ]
    goal_list_px = optimize_goal_sequence(goal_list_px, start_px)

    print("🧠 开始多目标路径规划...")
    full_path = []
    current_start = start_px
    for idx, goal_px in enumerate(goal_list_px):
        print(f"\n🧭 规划第 {idx + 1} 段：从 {current_start} 到 {goal_px}")
        segment_path = a_star(grid_inflated, current_start, goal_px)
        if not segment_path:
            print(f"❌ 无法从 {current_start} 到达 {goal_px}")
            return
        segment_path = interpolate_path(segment_path, factor=24)
        if full_path and segment_path[0] == full_path[-1]:
            segment_path = segment_path[1:]
        full_path += segment_path
        current_start = goal_px

    visualize_path(grid_inflated, full_path, start_px, goal_list_px[-1])
    full_path = remove_consecutive_duplicates(full_path)
    waypoints_px = extract_waypoints_by_distance(full_path, num_points=30)

    flight_waypoints = []
    for (y, x) in waypoints_px:
        wx = (x - w // 2) * scale
        wy = -(y - h // 2) * scale
        wz = -5
        flight_waypoints.append(airsim.Vector3r(wx, wy, wz))

    special_targets = []
    for px in goal_list_px:
        wy = -(px[0] - h // 2) * scale
        wx = (px[1] - w // 2) * scale
        wz = -5
        special_targets.append(airsim.Vector3r(wx, wy, wz))

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name="Drone1")
    client.armDisarm(True, vehicle_name="Drone1")
    print("🛫 起飞中...")
    client.takeoffAsync(vehicle_name="Drone1").join()
    client.moveToZAsync(-5, 2, vehicle_name="Drone1").join()  # 提前飞到目标高度

    print("🚁 航路点追踪飞行开始...")
    move_by_path_tracking(client, flight_waypoints, special_points=special_targets, Va=3, z_val=-5, epsilon=1.0, dt=0.1)
    print("✅ 全部目标点任务完成，系统悬停中...")


if __name__ == "__main__":
    main()
