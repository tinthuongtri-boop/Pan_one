"""
Mô phỏng ground-truth trajectory + tín hiệu Encoder (Servo Controller L2DB)
và IMU (HWT901B) tương ứng cho robot differential-drive.

Kịch bản chuyển động (thiết kế để bộc lộ rõ hành vi EKF):
  0-2s   : đứng yên (kiểm tra IMU drift lúc đứng yên - vấn đề nêu từ đầu hội thoại)
  2-5s   : tăng tốc thẳng   (v: 0 -> 0.3 m/s)
  5-10s  : đi thẳng đều     (v = 0.3 m/s)
  10-11s : XẢY RA WHEEL-SLIP trong lúc bắt đầu rẽ (bánh trái "ảo" quay nhanh hơn
           thực tế - đúng kịch bản Chu kỳ 2 đã tính tay ở các lượt trước)
  10-15s : vừa đi vừa rẽ    (v=0.3 m/s, ω=0.3 rad/s)
  15-18s : giảm tốc về 0
  18-20s : đứng yên

[ASSUMPTION] Toàn bộ thông số vật lý dưới đây (L, r, noise std...) là giá trị
minh họa dùng xuyên suốt cuộc hội thoại - PHẢI thay bằng số đo thực nghiệm/
datasheet thật của robot bạn trước khi dùng kết quả này để kết luận về robot
thật ngoài đời.
"""
import numpy as np
import pandas as pd

# ============ THAM SỐ VẬT LÝ [ASSUMPTION] ============
L = 0.30          # wheelbase (m)
r = 0.05          # bán kính bánh (m)
dt_true = 0.01    # bước tích phân ground-truth (100Hz, mô phỏng "liên tục")

# Tần số publish thực tế của 2 cảm biến (đúng theo phần cứng đã xác nhận)
SERVO_RATE_HZ = 100.0   # PDO servo, 10ms
IMU_RATE_HZ   = 50.0    # theo ekf.yaml (frequency: 50)

# ============ NHIỄU CẢM BIẾN [ASSUMPTION - cần thay bằng thực nghiệm thật] ============
ENCODER_RPM_NOISE_STD   = 1.2      # độ lệch chuẩn nhiễu rpm mỗi bánh (đo được ~1-2rpm là hợp lý)
ENCODER_TICK_PER_REV    = 4096     # theo object "Resolution" driver L2DB (mặc định datasheet)
GYRO_NOISE_STD_RAD_S    = 0.008    # độ lệch chuẩn nhiễu gyro (rad/s) - minh họa
GYRO_BIAS_RAD_S         = 0.006    # bias cố định của gyro (nguyên nhân "trôi khi đứng yên")

rng = np.random.default_rng(42)

# ============ 1. GROUND TRUTH: v(t), omega(t) theo kịch bản ============
def v_profile(t):
    if t < 2:      return 0.0
    if t < 5:      return 0.3 * (t-2)/3.0          # tăng tốc tuyến tính
    if t < 15:     return 0.3
    if t < 18:     return 0.3 * (1 - (t-15)/3.0)   # giảm tốc tuyến tính
    return 0.0

def omega_profile(t):
    if 10 <= t < 15:
        return 0.3
    return 0.0

t_arr = np.arange(0, 20, dt_true)
v_true     = np.array([v_profile(t) for t in t_arr])
omega_true = np.array([omega_profile(t) for t in t_arr])

# Tích phân ra ground-truth pose (dùng chính công thức differential-drive đã học)
X = np.zeros_like(t_arr); Y = np.zeros_like(t_arr); YAW = np.zeros_like(t_arr)
for i in range(1, len(t_arr)):
    X[i]   = X[i-1]   + v_true[i-1]*np.cos(YAW[i-1])*dt_true
    Y[i]   = Y[i-1]   + v_true[i-1]*np.sin(YAW[i-1])*dt_true
    YAW[i] = YAW[i-1] + omega_true[i-1]*dt_true

# ============ 2. Bánh trái/phải THẬT (inverse kinematics) ============
vL_true = v_true - omega_true*L/2.0
vR_true = v_true + omega_true*L/2.0

# ============ 3. TIÊM SỰ KIỆN WHEEL-SLIP (10.0s - 10.3s) ============
# Bánh trái "ảo" báo cáo quay CHẬM hơn thực tế -> omega_encoder bị lệch
slip_mask = (t_arr >= 10.0) & (t_arr < 10.3)
vL_reported = vL_true.copy()
vL_reported[slip_mask] -= 0.15   # bánh trái báo thiếu 0.15 m/s do trượt (mất độ bám)

# ============ 4. rpm mỗi bánh (những gì Servo Controller THỰC SỰ đo & gửi) ============
def to_rpm(v_wheel):
    return v_wheel / r * 60.0/(2*np.pi)

rpm_L_true = to_rpm(vL_reported)   # dùng giá trị ĐÃ CÓ SLIP làm "vật lý thật mà encoder thấy"
rpm_R_true = to_rpm(vR_true)

# ============ 5. LẤY MẪU theo đúng tần số publish thật của từng cảm biến, kèm nhiễu ============
def sample_and_noise(t_arr, signal_true, rate_hz, noise_std, rng, bias=0.0, quant=None):
    dt_sample = 1.0/rate_hz
    t_samples = np.arange(0, t_arr[-1], dt_sample)
    signal_sampled = np.interp(t_samples, t_arr, signal_true)
    noisy = signal_sampled + bias + rng.normal(0, noise_std, size=signal_sampled.shape)
    if quant is not None:
        noisy = np.round(noisy/quant)*quant
    return t_samples, noisy

# --- Servo Controller (rpm mỗi bánh, publish 100Hz qua CAN Bus 1) ---
t_servo, rpm_L_meas = sample_and_noise(t_arr, rpm_L_true, SERVO_RATE_HZ, ENCODER_RPM_NOISE_STD, rng)
_,       rpm_R_meas = sample_and_noise(t_arr, rpm_R_true, SERVO_RATE_HZ, ENCODER_RPM_NOISE_STD, rng)

# --- IMU gyro Z (rad/s, publish 50Hz qua CAN Bus 2 + hub UART-to-CAN) ---
t_imu, gyro_z_meas = sample_and_noise(t_arr, omega_true, IMU_RATE_HZ, GYRO_NOISE_STD_RAD_S, rng,
                                       bias=GYRO_BIAS_RAD_S)

# ============ 6. Tính v, omega_enc TỪ rpm đo được (đúng công thức Chiều FEEDBACK) ============
def rpm_to_wheel_v(rpm): return rpm * (2*np.pi/60.0) * r
vL_computed = rpm_to_wheel_v(rpm_L_meas)
vR_computed = rpm_to_wheel_v(rpm_R_meas)
v_computed     = (vL_computed + vR_computed)/2.0
omega_computed = (vR_computed - vL_computed)/L

# ============ 7. Xuất CSV — 3 file riêng: ground_truth, wheel_twist, imu ============
pd.DataFrame({
    "t": t_arr, "X_true": X, "Y_true": Y, "yaw_true": YAW,
    "v_true": v_true, "omega_true": omega_true,
}).to_csv("ground_truth.csv", index=False)

pd.DataFrame({
    "t": t_servo,
    "rpm_L_measured": rpm_L_meas, "rpm_R_measured": rpm_R_meas,
    "vx_computed": v_computed, "vyaw_computed": omega_computed,
    "vx_variance": ENCODER_RPM_NOISE_STD**2 * (2*np.pi/60*r/2)**2 * 2,   # ước lượng lan truyền nhiễu
    "vyaw_variance": ENCODER_RPM_NOISE_STD**2 * (2*np.pi/60*r/L)**2 * 2,
}).to_csv("wheel_twist_sim.csv", index=False)

pd.DataFrame({
    "t": t_imu, "vyaw_measured": gyro_z_meas,
    "vyaw_variance": GYRO_NOISE_STD_RAD_S**2,
}).to_csv("imu_sim.csv", index=False)

print("Đã tạo: ground_truth.csv, wheel_twist_sim.csv, imu_sim.csv")
print(f"Số mẫu ground_truth={len(t_arr)}, wheel={len(t_servo)}, imu={len(t_imu)}")
print(f"Wheel-slip tiêm vào khoảng t=[10.0, 10.3)s, bánh trái báo thiếu 0.15 m/s")
