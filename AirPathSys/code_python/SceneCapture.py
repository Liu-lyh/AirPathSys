import airsim
import time
import numpy as np
import cv2
from airsim import Vector3r, to_quaternion


def takeoff_and_capture(vehicle_name="Drone1", camera_name="mapping_cam"):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)

    # 起飞并升至100米
    print("✅ 起飞...")
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    print("📷 上升到100米拍照...")
    client.moveToZAsync(-140, 3, vehicle_name=vehicle_name).join()
    time.sleep(2)  # 等待拍照前的准备

    # 设置相机朝下
    orientation = to_quaternion(np.radians(-90), 0, 0)
    client.simSetCameraPose(camera_name, airsim.Pose(Vector3r(0, 0, 0), orientation), vehicle_name)

    # 拍照
    print("📸 拍照中...")
    responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)],
                                    vehicle_name)
    if not responses or responses[0].height == 0:
        print("❌ 拍照失败")
        exit()

    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    cv2.imwrite("../map_image/scene.png", img_rgb)

    # 返回起点
    print("➡️ 返回起点 (0,0,-5)...")
    client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=vehicle_name).join()
    time.sleep(1)

    # 降落
    print("🛬 开始降落...")
    client.landAsync(vehicle_name=vehicle_name).join()

    client.armDisarm(False, vehicle_name)
    client.enableApiControl(False, vehicle_name)
    print("✅ 拍照任务完成，系统退出。")


if __name__ == "__main__":
    takeoff_and_capture()
