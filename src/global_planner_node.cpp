#include <memory>
#include <cmath>
#include <vector>
#include <queue>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

struct NodeCell {
    int x, y;
    float g_cost;
    float f_cost;
    bool operator>(const NodeCell& other) const {
        return f_cost > other.f_cost;
    }
};

class GlobalPlannerNode : public rclcpp::Node
{
public:
    GlobalPlannerNode() : Node("global_planner_node")
    {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        map_subscriber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&GlobalPlannerNode::map_callback, this, std::placeholders::_1));
        
        goal_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, std::bind(&GlobalPlannerNode::goal_callback, this, std::placeholders::_1));

        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>(
            "/global_path", rclcpp::QoS(1).transient_local());
            
        // Kênh phát Costmap ra RViz để trực quan hóa
        costmap_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/custom_costmap", rclcpp::QoS(1).transient_local());

        RCLCPP_INFO(this->get_logger(), "A* Planner (kem Costmap) da san sang!");
    }

private:
    nav_msgs::msg::OccupancyGrid::SharedPtr map_data_;
    std::vector<int8_t> costmap_data_; // Mảng chứa dữ liệu bản đồ đã thổi phồng
    int inflation_radius_cells_ = 6;   // Bán kính thổi phồng: 6 ô * 0.05m = 0.3 mét

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_publisher_;

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        map_data_ = msg;
        inflateMap(); // Chạy thuật toán thổi phồng ngay khi nhận được map
    }

    // =========================================================================
    // THUẬT TOÁN TẠO COSTMAP (INFLATION)
    // =========================================================================
    void inflateMap() {
        int width = map_data_->info.width;
        int height = map_data_->info.height;
        costmap_data_.assign(width * height, 0);

        std::vector<std::pair<int, int>> obstacles;

        // 1. Quét tìm toàn bộ bức tường gốc
        for (int i = 0; i < width * height; i++) {
            if (map_data_->data[i] >= 50) {
                costmap_data_[i] = 100; // Tường chết (Lethal)
                obstacles.push_back({i % width, i / width});
            } else if (map_data_->data[i] == -1) {
                costmap_data_[i] = -1;  // Chưa biết
            }
        }

        // 2. Lan tỏa độ nguy hiểm (Cost) ra xung quanh từng bức tường
        for (auto& obs : obstacles) {
            for (int dx = -inflation_radius_cells_; dx <= inflation_radius_cells_; dx++) {
                for (int dy = -inflation_radius_cells_; dy <= inflation_radius_cells_; dy++) {
                    int nx = obs.first + dx;
                    int ny = obs.second + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        float dist = std::hypot(dx, dy); // Định lý Pytago
                        
                        if (dist <= inflation_radius_cells_) {
                            int idx = ny * width + nx;
                            if (costmap_data_[idx] != 100 && costmap_data_[idx] != -1) {
                                // Tính toán giá trị nguy hiểm (Càng gần tường cost càng cao)
                                int cost = 99 - (dist / inflation_radius_cells_ * 99);
                                
                                // Nếu ô này đang bị đè bởi nhiều quầng sáng từ nhiều bức tường, lấy giá trị cao nhất
                                if (cost > costmap_data_[idx]) {
                                    costmap_data_[idx] = cost;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Đóng gói và phát lên RViz
        nav_msgs::msg::OccupancyGrid costmap_msg;
        costmap_msg.header = map_data_->header;
        costmap_msg.info = map_data_->info;
        costmap_msg.data = costmap_data_;
        costmap_publisher_->publish(costmap_msg);
        
        RCLCPP_INFO(this->get_logger(), "Da tinh toan xong Costmap va phat tren /custom_costmap");
    }

    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr goal_msg)
    {
        if (!map_data_ || costmap_data_.empty()) return;

        geometry_msgs::msg::TransformStamped transformStamped;
        try {
            transformStamped = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Loi TF: %s", ex.what());
            return;
        }

        double start_x = transformStamped.transform.translation.x;
        double start_y = transformStamped.transform.translation.y;
        double goal_x = goal_msg->pose.position.x;
        double goal_y = goal_msg->pose.position.y;

        int sgx, sgy, ggx, ggy;
        if (!worldToGrid(start_x, start_y, sgx, sgy) || !worldToGrid(goal_x, goal_y, ggx, ggy)) {
            RCLCPP_ERROR(this->get_logger(), "Toa do nam ngoai ban do!"); return;
        }

        // Kiểm tra đích đến trên COSTMAP (chứ không phải map gốc nữa)
        int goal_index = ggy * map_data_->info.width + ggx;
        if (costmap_data_[goal_index] >= 50) {
            RCLCPP_ERROR(this->get_logger(), "Dich den nam trong vung NGAY HIEM hoac TUONG!");
            return;
        }

        auto path_cells = a_star(sgx, sgy, ggx, ggy);

        if (!path_cells.empty()) {
            nav_msgs::msg::Path path_msg;
            path_msg.header.stamp = this->get_clock()->now();
            path_msg.header.frame_id = "map";

            for (auto cell : path_cells) {
                geometry_msgs::msg::PoseStamped pose;
                pose.header.frame_id = "map";
                pose.header.stamp = path_msg.header.stamp;
                pose.pose.position.x = gridToWorldX(cell.first);
                pose.pose.position.y = gridToWorldY(cell.second);
                pose.pose.position.z = 0.05; // Nâng Z lên để không bị đè
                pose.pose.orientation.w = 1.0; 
                path_msg.poses.push_back(pose);
            }
            path_publisher_->publish(path_msg);
            RCLCPP_INFO(this->get_logger(), "Phat duong di thanh cong!");
        }
    }

    std::vector<std::pair<int, int>> a_star(int start_x, int start_y, int goal_x, int goal_y)
    {
        int width = map_data_->info.width;
        int height = map_data_->info.height;
        int map_size = width * height;

        std::vector<int> came_from(map_size, -1);
        std::vector<float> g_score(map_size, std::numeric_limits<float>::infinity());
        std::priority_queue<NodeCell, std::vector<NodeCell>, std::greater<NodeCell>> open_set;

        int start_idx = start_y * width + start_x;
        g_score[start_idx] = 0;
        open_set.push({start_x, start_y, 0, heuristic(start_x, start_y, goal_x, goal_y)});

        int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

        while (!open_set.empty()) {
            NodeCell current = open_set.top();
            open_set.pop();

            int curr_idx = current.y * width + current.x;

            if (current.x == goal_x && current.y == goal_y) {
                std::vector<std::pair<int, int>> path;
                int trace_idx = curr_idx;
                while (trace_idx != -1) {
                    path.push_back({trace_idx % width, trace_idx / width});
                    trace_idx = came_from[trace_idx];
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            for (int i = 0; i < 8; ++i) {
                int nx = current.x + dx[i];
                int ny = current.y + dy[i];

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int neighbor_idx = ny * width + nx;

                    // KIỂM TRA VẬT CẢN TRÊN COSTMAP: Loại bỏ các ô nguy hiểm sát tường (>= 50)
                    int cost = costmap_data_[neighbor_idx];
                    if (cost >= 50 || cost == -1) continue; 

                    // Nếu đi vào vùng Cost > 0 (gần tường), phạt nặng chi phí để ép A* đi ra giữa phòng
                    float step_penalty = 1.0f + (cost / 100.0f) * 10.0f; 
                    float step_cost = (dx[i] != 0 && dy[i] != 0) ? 1.414f * step_penalty : 1.0f * step_penalty;
                    
                    float tentative_g_score = g_score[curr_idx] + step_cost;

                    if (tentative_g_score < g_score[neighbor_idx]) {
                        came_from[neighbor_idx] = curr_idx;
                        g_score[neighbor_idx] = tentative_g_score;
                        float f_score = tentative_g_score + heuristic(nx, ny, goal_x, goal_y);
                        open_set.push({nx, ny, tentative_g_score, f_score});
                    }
                }
            }
        }
        return {};
    }

    float heuristic(int x1, int y1, int x2, int y2) {
        return std::hypot(x1 - x2, y1 - y2);
    }

    bool worldToGrid(double wx, double wy, int &gx, int &gy) {
        gx = std::floor((wx - map_data_->info.origin.position.x) / map_data_->info.resolution);
        gy = std::floor((wy - map_data_->info.origin.position.y) / map_data_->info.resolution);
        return (gx >= 0 && gx < (int)map_data_->info.width && gy >= 0 && gy < (int)map_data_->info.height);
    }

    double gridToWorldX(int gx) {
        return (gx + 0.5) * map_data_->info.resolution + map_data_->info.origin.position.x;
    }
    double gridToWorldY(int gy) {
        return (gy + 0.5) * map_data_->info.resolution + map_data_->info.origin.position.y;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalPlannerNode>());
    rclcpp::shutdown();
    return 0;
}