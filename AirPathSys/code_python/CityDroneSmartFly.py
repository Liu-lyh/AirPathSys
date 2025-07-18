import airsim
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
# 设置中文字体
rcParams['font.sans-serif'] = ['SimSun']    # 宋体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 起飞
client.takeoffAsync().join()
client.moveToZAsync(-2, 1).join()
time.sleep(1)

# 多个目标点
GOALS = [
    # np.array([60.0, 40.0, -5.0]),
    # np.array([-20.0, 60.0, -5.0]),
    # np.array([50.0, -40.0, -5.0])
    np.array([10.0, 10.0, -5.0]),
    np.array([42.0, 5.0, -5.0]),
    np.array([60.0, -30.0, -3.0])
]

SAFE_DIST = 5.0
FORWARD_VEL = 2.0
SIDE_VEL = 1.5
UPWARD_VEL = 1.0
DT = 1.0
AVOID_FRAMES = 1  # 避障持续帧数

trajectory = []

# ✅ 可视化目标点
for goal in GOALS:
    client.simPlotPoints([airsim.Vector3r(*goal)], color_rgba=[0.0, 1.0, 0.0, 1.0], size=30.0, is_persistent=True)

# ---------- 工具函数 ----------
def get_min_dist(lidar_name):
    data = client.getLidarData(lidar_name)
    if len(data.point_cloud) < 3:
        return float('inf')
    points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
    return np.min(np.linalg.norm(points, axis=1))

def record_position():
    pos = client.getMultirotorState().kinematics_estimated.position
    point = (pos.x_val, pos.y_val, pos.z_val)
    trajectory.append(point)
    return np.array(point)

def update_sim_trajectory(traj):
    points = [airsim.Vector3r(x, y, z) for (x, y, z) in traj]
    client.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness=10.0, is_persistent=True)

def plot_trajectory(data):
    x, y, z = zip(*data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='red', label='Flight Trajectory', linewidth=2)
    for goal in GOALS:
        ax.scatter([goal[0]], [goal[1]], [goal[2]], c='blue', s=100, marker='*', label='目标点')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D 多目标飞行路径')
    ax.set_zlim(-10, 0)         # 手动设置Z轴范围（可选）
    ax.invert_zaxis()           # ✅ 反转Z轴方向（让0在下，负值在上）
    ax.legend()
    plt.show()


def rotate_towards(goal):
    state = client.getMultirotorState()
    curr_pos = state.kinematics_estimated.position
    dx = goal[0] - curr_pos.x_val
    dy = goal[1] - curr_pos.y_val
    yaw_rad = math.atan2(dy, dx)
    yaw_deg = math.degrees(yaw_rad)
    print(f"🔄 正在转向下一个目标 Yaw: {yaw_deg:.1f}°")
    client.rotateToYawAsync(yaw_deg).join()
    time.sleep(0.5)

# ---------- 主飞行循环 ----------
print("🚁 启动多目标避障飞行")

for idx, GOAL in enumerate(GOALS):
    is_last = (idx == len(GOALS) - 1)

    if idx > 0:
        rotate_towards(GOAL)

    print(f"\n🎯 正在前往目标 {idx + 1}: {GOAL.tolist()}")

    # 状态变量
    avoid_counter = 0
    avoid_direction = None

    try:
        while True:
            pos = record_position()
            update_sim_trajectory(trajectory)

            dist_to_goal = np.linalg.norm(pos - GOAL)
            print(f"📍 距离目标点：{dist_to_goal:.2f} 米")
            if dist_to_goal < 2.0:
                print("✅ 到达目标点！")
                break

            direction = GOAL - pos
            direction = direction / np.linalg.norm(direction)
            vx, vy, vz = direction * FORWARD_VEL

            if avoid_counter > 0:
                print(f"⏱️ 避障中（剩余 {avoid_counter} 帧）")
                if avoid_direction == 'left':
                    vy = -SIDE_VEL
                elif avoid_direction == 'right':
                    vy = SIDE_VEL
                elif avoid_direction == 'up':
                    vz = -UPWARD_VEL
                avoid_counter -= 1
                client.moveByVelocityAsync(vx, vy, vz, DT).join()
                time.sleep(0.1)
                continue

            # 获取雷达数据
            front = get_min_dist("LidarFront")
            left = get_min_dist("LidarLeft")
            right = get_min_dist("LidarRight")

            if front < SAFE_DIST:
                print("⚠️ 前方障碍！")
                if left > right and left > SAFE_DIST:
                    avoid_direction = 'left'
                    print("↖️ 向左避障")
                elif right >= left and right > SAFE_DIST:
                    avoid_direction = 'right'
                    print("↗️ 向右避障")
                else:
                    avoid_direction = 'up'
                    print("⬆️ 上升避障")
                avoid_counter = AVOID_FRAMES
                continue

            print(f"✅ 路径畅通 vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
            client.moveByVelocityAsync(vx, vy, vz, DT).join()
            time.sleep(0.1)

        if is_last:
            target_z = pos[2] + 1.0
            print("⏬ 最终目标点，下降 1 米并悬停...")
            client.moveToZAsync(target_z, 0.5).join()
            client.hoverAsync().join()
            print("🛸 悬停完成")

    except KeyboardInterrupt:
        print("🛑 手动中断飞行")
        break

# ✅ 绘制最终轨迹
plot_trajectory(trajectory)
