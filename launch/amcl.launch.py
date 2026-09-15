import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'Pan_one'
    
    # Đường dẫn tới file map và amcl.yaml
    map_file = '/home/tinthuongtri/dev_ws/my_map_save.yaml'
    amcl_config_file = os.path.join(get_package_share_directory(pkg_name), 'config', 'amcl.yaml')

    return LaunchDescription([
        # 1. Node Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': True}, {'yaml_filename': map_file}]
        ),
        
        # 2. Node AMCL: Load cấu hình từ amcl.yaml
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[amcl_config_file, {'use_sim_time': True}]
        ),

        # 3. Node Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'autostart': True},
                {'node_names': ['map_server', 'amcl']}
            ]
        )
    ])