import airsim
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    np.array([60.0, 40.0, -5.0]),
    np.array([-20.0, 60.0, -5.0]),
    np.array([50.0, -40.0, -5.0])
]

SAFE_DIST = 5.0  # 避障敏感度
FORWARD_VEL = 2.0
SIDE_VEL = 1.5
UPWARD_VEL = 1.0
DT = 1.0
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
        ax.scatter([goal[0]], [goal[1]], [goal[2]], c='blue', s=100, marker='*', label='Goal Point')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Multi-Goal Flight Path')
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


# ✅ 起飞后朝第一个目标点调整方向并推进一帧
rotate_towards(GOALS[0])
pos = np.array(record_position())
dir_vec = GOALS[0] - pos
dir_vec = dir_vec / np.linalg.norm(dir_vec)
client.moveByVelocityAsync(dir_vec[0] * FORWARD_VEL, dir_vec[1] * FORWARD_VEL, dir_vec[2] * FORWARD_VEL, DT).join()
time.sleep(0.3)

# ---------- 主飞行循环 ----------
print("🚁 启动多目标避障飞行")

for idx, GOAL in enumerate(GOALS):
    is_last = (idx == len(GOALS) - 1)

    if idx > 0:
        rotate_towards(GOAL)
        pos = np.array(record_position())
        next_dir = GOAL - pos
        next_dir = next_dir / np.linalg.norm(next_dir)
        vx, vy, vz = next_dir * FORWARD_VEL
        print("🚶 转向后推进一帧")
        client.moveByVelocityAsync(vx, vy, vz, DT).join()
        time.sleep(0.3)

    print(f"\n🎯 正在前往目标 {idx + 1}: {GOAL.tolist()}")

    try:
        while True:
            pos = record_position()
            update_sim_trajectory(trajectory)

            dist_to_goal = np.linalg.norm(pos - GOAL)
            print(f"📍 距离目标点：{dist_to_goal:.2f} 米")
            if dist_to_goal < 2.0:
                print("✅ 到达目标点！")
                break

            front = get_min_dist("LidarFront")
            left = get_min_dist("LidarLeft")
            right = get_min_dist("LidarRight")

            direction = GOAL - pos
            direction = direction / np.linalg.norm(direction)
            vx, vy, vz = direction * FORWARD_VEL

            if front < SAFE_DIST:
                print("⚠️ 前方障碍！")
                if left > right and left > SAFE_DIST:
                    vy = -SIDE_VEL
                    print("↖️ 向左避障")
                elif right >= left and right > SAFE_DIST:
                    vy = SIDE_VEL
                    print("↗️ 向右避障")
                else:
                    vx, vy, vz = 0, 0, -UPWARD_VEL
                    print("⬆️ 上升避障")

                # 避障动作 + 持续 1 帧
                for i in range(2):  # 一帧原动作 + 一帧持续动作
                    client.moveByVelocityAsync(vx, vy, vz, DT).join()
                    time.sleep(0.3)
            else:
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

# ✅ 最终轨迹图
plot_trajectory(trajectory)
