#!/usr/bin/env python3
"""
[REFERENCE IMPLEMENTATION - chưa test, cần ROS 2 thật để chạy]

Replay đúng dữ liệu đã mô phỏng (wheel_twist_sim.csv, imu_sim.csv) thành
2 topic ROS 2 thật (/wheel_twist, /imu/data), publish đúng tốc độ thời gian
thực (100Hz và 50Hz), để bạn cắm thẳng vào node `ekf_filter_node` đang chạy
với chính file `ekf.yaml` của bạn - kiểm chứng EKF THẬT trên dữ liệu ĐÃ BIẾT
TRƯỚC ground truth (ground_truth.csv), điều mà robot thật không có được.

Cách dùng (trên máy có ROS 2 - Humble/Jazzy/Kilted, cần bạn xác nhận đúng
distro để tôi kiểm tra API rclpy/message_filters nếu có khác biệt):

    ros2 run robot_localization ekf_node --ros-args --params-file ekf.yaml &
    python3 replay_to_ros2.py
    ros2 topic echo /odometry/filtered   # hoặc rviz2 để xem trực quan

Sau khi chạy xong, so sánh /odometry/filtered với ground_truth.csv bằng
script compare_ros2_vs_groundtruth.py (bên dưới, gợi ý logic, cần viết
riêng tùy cách bạn muốn ghi log - ví dụ dùng `ros2 bag record` rồi
`ros2 bag play` kết hợp đọc lại bằng rosbag2_py).
"""
import csv
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    return rows


class ReplayNode(Node):
    def __init__(self):
        super().__init__('sim_replay_node')
        self.pub_wheel = self.create_publisher(TwistWithCovarianceStamped, '/wheel_twist', 10)
        self.pub_imu = self.create_publisher(Imu, '/imu/data', 10)

        self.wheel_data = load_csv('wheel_twist_sim.csv')
        self.imu_data = load_csv('imu_sim.csv')
        self.get_logger().info(
            f"Nạp {len(self.wheel_data)} mẫu wheel, {len(self.imu_data)} mẫu imu")

        self.i_wheel = 0
        self.i_imu = 0
        self.t0 = time.time()

        # Chạy 2 timer riêng biệt, ĐÚNG tần số thật của từng cảm biến
        # (100Hz cho wheel, 50Hz cho imu) - mô phỏng đúng tính bất đồng bộ
        # đã bàn trong toàn bộ cuộc hội thoại, KHÔNG gộp chung 1 timer.
        self.create_timer(1.0 / 100.0, self.publish_wheel)
        self.create_timer(1.0 / 50.0, self.publish_imu)

    def publish_wheel(self):
        if self.i_wheel >= len(self.wheel_data):
            return
        row = self.wheel_data[self.i_wheel]
        self.i_wheel += 1

        msg = TwistWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.twist.linear.x = row['vx_computed']
        msg.twist.twist.linear.y = 0.0          # differential drive - không đo được
        msg.twist.twist.angular.z = row['vyaw_computed']

        # Field covariance 6x6 dạng flatten - CHỈ điền đúng 2 vị trí đường chéo
        # tương ứng vx (index 0) và vyaw (index 35), theo đúng thứ tự chuẩn
        # [vx,vy,vz,vroll,vpitch,vyaw] - khớp twist0_config đã review trước đó
        cov = [0.0] * 36
        cov[0] = row['vx_variance']
        cov[35] = row['vyaw_variance']
        msg.twist.covariance = cov

        self.pub_wheel.publish(msg)

    def publish_imu(self):
        if self.i_imu >= len(self.imu_data):
            return
        row = self.imu_data[self.i_imu]
        self.i_imu += 1

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.angular_velocity.z = row['vyaw_measured']

        # sensor_msgs/Imu yêu cầu orientation_covariance[0] = -1 nếu không
        # cung cấp orientation (theo convention chuẩn REP đã xác nhận)
        msg.orientation_covariance[0] = -1.0
        av_cov = [0.0] * 9
        av_cov[8] = row['vyaw_variance']   # vị trí (2,2) trong ma trận 3x3 = trục Z
        msg.angular_velocity_covariance = av_cov

        self.pub_imu.publish(msg)


def main():
    rclpy.init()
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
