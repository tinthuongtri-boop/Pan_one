"""
So sánh 2 kiến trúc EKF trên CÙNG 1 bộ dữ liệu mô phỏng (simulate_dataset.py):

  (A) EKF CỔ ĐIỂN (3-state, kiểu Barfoot/Thrun dạy ban đầu trong hội thoại)
      state = [X, Y, yaw]
      wheel (v, omega_enc) DÙNG LÀM CONTROL INPUT u_k -> Predict, KHÔNG qua Correct
      IMU: tích phân gyro thành theta_gyro cục bộ -> dùng làm MEASUREMENT sửa yaw

  (B) EKF AUGMENTED-STATE (5-state, kiểu robot_localization / đúng ekf.yaml đang dùng)
      state = [X, Y, yaw, vx, vyaw]
      wheel (v, omega_enc) là MEASUREMENT của state (vx, vyaw) - CÓ Correct
      imu (gyro_z) là MEASUREMENT của state (vyaw) - CÓ Correct, xử lý TUẦN TỰ

Toàn bộ công thức dùng đúng 10 bước Predict/Correct đã thống nhất trong hội thoại.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

L = 0.30  # phải khớp giá trị dùng lúc mô phỏng dữ liệu

gt      = pd.read_csv("ground_truth.csv")
wheel   = pd.read_csv("wheel_twist_sim.csv")
imu     = pd.read_csv("imu_sim.csv")

# Gộp toàn bộ sự kiện (wheel + imu) theo đúng thời gian, sắp xếp tăng dần
# -> mô phỏng đúng tính chất BẤT ĐỒNG BỘ đã bàn kỹ trong hội thoại
events = []
for _, row in wheel.iterrows():
    events.append((row["t"], "wheel", row["vx_computed"], row["vyaw_computed"],
                   row["vx_variance"], row["vyaw_variance"]))
for _, row in imu.iterrows():
    events.append((row["t"], "imu", None, row["vyaw_measured"], None, row["vyaw_variance"]))
events.sort(key=lambda e: e[0])

# ============================================================
# (0) ODOMETRY THUẦN TÚY — không Kalman, không fuse gì cả
#     (đúng đường "odometer reading" trong ảnh tham chiếu bạn gửi)
# ============================================================
x0 = np.array([0.0, 0.0, 0.0])
last_t_0 = 0.0
results0 = []
for _, row in wheel.iterrows():
    t = row["t"]; dt = t - last_t_0; last_t_0 = t
    v, w = row["vx_computed"], row["vyaw_computed"]
    x0 = x0 + np.array([v*np.cos(x0[2])*dt, v*np.sin(x0[2])*dt, w*dt])
    results0.append((t, x0[0], x0[1], x0[2]))
res0 = pd.DataFrame(results0, columns=["t","X","Y","yaw"])


# ============================================================
# (A) EKF CỔ ĐIỂN — 3-state, wheel=control input, IMU sửa yaw absolute
# ============================================================
xA = np.array([0.0, 0.0, 0.0])              # [X, Y, yaw]
PA = np.diag([1e-6, 1e-6, 1e-6])
R_process_A = np.diag([0.02, 0.02, 0.01])   # process noise (motion model không hoàn hảo)
Q_meas_A    = 0.02**2                        # measurement noise cho theta_gyro tích lũy

theta_gyro_local = 0.0   # IMU tự tích phân góc cục bộ (kiểu ví dụ dạy đầu tiên)
last_t_A = 0.0
last_imu_t_A = 0.0       # BUG FIX: phải track riêng thời điểm IMU gần nhất
                          # (không dùng chung last_t_A, vì wheel event xen giữa
                          #  làm dt bị "cắt cụt" sai, khiến theta_gyro_local
                          #  gần như không tích lũy được)
resultsA = []

def jacobianG_A(x, v, dt):
    _, _, yaw = x
    return np.array([
        [1, 0, -v*np.sin(yaw)*dt],
        [0, 1,  v*np.cos(yaw)*dt],
        [0, 0,  1]
    ])

for (t, src, vx_meas, vyaw_meas, vx_var, vyaw_var) in events:
    dt = t - last_t_A
    if dt < 0: dt = 0
    last_t_A = t

    if src == "wheel":
        # === PREDICT (wheel = control input, dùng NGAY, không Correct) ===
        v_ctrl, w_ctrl = vx_meas, vyaw_meas
        G = jacobianG_A(xA, v_ctrl, dt)
        xA = xA + np.array([
            v_ctrl*np.cos(xA[2])*dt,
            v_ctrl*np.sin(xA[2])*dt,
            w_ctrl*dt
        ])
        PA = G @ PA @ G.T + R_process_A*dt
        theta_gyro_local += 0  # không đổi ở bước wheel

    elif src == "imu":
        # IMU: tích phân cục bộ thành theta_gyro, rồi dùng làm MEASUREMENT sửa yaw
        dt_imu = t - last_imu_t_A     # BUG FIX: dt tính từ lần IMU TRƯỚC ĐÓ, không
        last_imu_t_A = t              # phải từ "sự kiện bất kỳ gần nhất" (có thể là wheel)
        theta_gyro_local += vyaw_meas * dt_imu
        H = np.array([0, 0, 1.0])
        z = theta_gyro_local
        zhat = xA[2]
        y = z - zhat
        S = H @ PA @ H.T + Q_meas_A
        K = PA @ H.T / S
        xA = xA + K*y
        PA = (np.eye(3) - np.outer(K, H)) @ PA

    resultsA.append((t, xA[0], xA[1], xA[2], PA[2,2]))

resA = pd.DataFrame(resultsA, columns=["t","X","Y","yaw","P_yaw"])

# ============================================================
# (B) EKF AUGMENTED-STATE — 5-state, xử lý TUẦN TỰ đúng robot_localization
# ============================================================
xB = np.array([0.0, 0.0, 0.0, 0.0, 0.0])    # [X, Y, yaw, vx, vyaw]
PB = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
Rp = np.diag([0.001, 0.001, 0.0005, 0.025, 0.02])  # process noise, khớp tinh thần ekf.yaml
last_t_B = 0.0
resultsB = []

def jacobianG_B(x, dt):
    X,Y,yaw,vx,vyaw = x
    G = np.eye(5)
    G[0,2] = -vx*np.sin(yaw)*dt
    G[0,3] =  np.cos(yaw)*dt
    G[1,2] =  vx*np.cos(yaw)*dt
    G[1,3] =  np.sin(yaw)*dt
    G[2,4] =  dt
    return G

for (t, src, vx_meas, vyaw_meas, vx_var, vyaw_var) in events:
    dt = t - last_t_B
    if dt < 0: dt = 0
    last_t_B = t

    # === PREDICT — LUÔN chạy trước mỗi Correct, dùng constant-velocity model ===
    X,Y,yaw,vx,vyaw = xB
    G = jacobianG_B(xB, dt)
    xB = np.array([
        X + vx*np.cos(yaw)*dt,
        Y + vx*np.sin(yaw)*dt,
        yaw + vyaw*dt,
        vx, vyaw
    ])
    PB = G @ PB @ G.T + Rp*dt

    # === CORRECT — tuần tự, tùy nguồn vừa đến ===
    if src == "wheel":
        H = np.array([[0,0,0,1,0],[0,0,0,0,1]])
        z = np.array([vx_meas, vyaw_meas])
        Qm = np.diag([vx_var, vyaw_var])
    else:  # imu
        H = np.array([[0,0,0,0,1]])
        z = np.array([vyaw_meas])
        Qm = np.array([[vyaw_var]])

    zhat = H @ xB
    y = z - zhat
    S = H @ PB @ H.T + Qm
    K = PB @ H.T @ np.linalg.inv(S)
    xB = xB + K @ y
    PB = (np.eye(5) - K @ H) @ PB

    resultsB.append((t, *xB, PB[2,2]))

resB = pd.DataFrame(resultsB, columns=["t","X","Y","yaw","vx","vyaw","P_yaw"])

# ============================================================
# (C) EKF AUGMENTED + BIAS ESTIMATION — 6-state
#     state = [X, Y, yaw, vx, vyaw, bias_gyro]
#
#     ĐÚNG mô hình vật lý thật: gyro_đo = vyaw_thật + bias  (KHÔNG phải = vyaw)
#     Wheel vẫn đo TRỰC TIẾP vyaw (không dính bias của IMU)
#     -> Có 2 nguồn ĐỘC LẬP cho cùng "vyaw thật", nên bias trở thành
#        QUAN SÁT ĐƯỢC (observable) và tự tách ra được, đúng cơ chế
#        trong ảnh tham chiếu "EKF-SLAM with bias estimation"
# ============================================================
xC = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # [X,Y,yaw,vx,vyaw,bias]
PC = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0.01])  # P0[bias] LỚN hơn hẳn - ta
                                                      # KHÔNG biết bias ban đầu là bao nhiêu
Rp_C = np.diag([0.001, 0.001, 0.0005, 0.025, 0.02, 1e-7])  # process noise của
                                                             # bias RẤT NHỎ - giả định
                                                             # bias gần như hằng số
                                                             # (chỉ trôi chậm theo nhiệt độ)
last_t_C = 0.0
resultsC = []
GATE_THRESHOLD_WHEEL = 9.0   # [ASSUMPTION] ngưỡng Mahalanobis^2 cho 2-DOF (~99%)
                              # để BẢO VỆ state bias khỏi bị "rò rỉ" bởi wheel-slip
                              # (đúng cơ chế reject_threshold đã bàn khi review ekf.yaml)
n_rejected = 0

def jacobianG_C(x, dt):
    X,Y,yaw,vx,vyaw,b = x
    G = np.eye(6)
    G[0,2] = -vx*np.sin(yaw)*dt
    G[0,3] =  np.cos(yaw)*dt
    G[1,2] =  vx*np.cos(yaw)*dt
    G[1,3] =  np.sin(yaw)*dt
    G[2,4] =  dt
    return G

for (t, src, vx_meas, vyaw_meas, vx_var, vyaw_var) in events:
    dt = t - last_t_C
    if dt < 0: dt = 0
    last_t_C = t

    X,Y,yaw,vx,vyaw,b = xC
    G = jacobianG_C(xC, dt)
    xC = np.array([
        X + vx*np.cos(yaw)*dt, Y + vx*np.sin(yaw)*dt,
        yaw + vyaw*dt, vx, vyaw, b
    ])
    PC = G @ PC @ G.T + Rp_C*dt

    if src == "wheel":
        H = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0]])
        z = np.array([vx_meas, vyaw_meas])
        Qm = np.diag([vx_var, vyaw_var])
    else:  # imu
        H = np.array([[0,0,0,0,1,1]])
        z = np.array([vyaw_meas])
        Qm = np.array([[vyaw_var]])

    zhat = H @ xC
    y = z - zhat
    S = H @ PC @ H.T + Qm

    # === GATING — tính Mahalanobis distance^2, CHỈ áp dụng cho wheel ===
    # (đây chính là chỗ chặn sự kiện wheel-slip "rò rỉ" vào state bias)
    d2 = y @ np.linalg.solve(S, y)
    if src == "wheel" and d2 > GATE_THRESHOLD_WHEEL:
        n_rejected += 1
        resultsC.append((t, *xC))   # KHÔNG Correct - chỉ giữ nguyên kết quả Predict
        continue

    K = PC @ H.T @ np.linalg.inv(S)
    xC = xC + K @ y
    PC = (np.eye(6) - K @ H) @ PC

    resultsC.append((t, *xC))

resC = pd.DataFrame(resultsC, columns=["t","X","Y","yaw","vx","vyaw","bias"])
print(f"[Gating] Số lần wheel measurement bị REJECT (Mahalanobis^2 > {GATE_THRESHOLD_WHEEL}): {n_rejected}")

# ============================================================
# VẼ ĐỒ THỊ SO SÁNH — phong cách giống ảnh tham chiếu (nhiều phương pháp)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (1) Quỹ đạo X-Y — đúng phong cách ảnh 2: Ground truth + nhiều phương pháp
ax = axes[0,0]
ax.plot(gt["X_true"], gt["Y_true"], 'k-', lw=2.5, label="Ground Truth")
ax.plot(res0["X"], res0["Y"], 'g:', lw=1.5, label="Odometry thuần (không fuse)")
ax.plot(resA["X"], resA["Y"], 'r-.', lw=1.3, label="EKF Cổ điển (3-state)")
ax.plot(resB["X"], resB["Y"], 'b--', lw=1.3, label="EKF Augmented (5-state, KHÔNG bias)")
ax.plot(resC["X"], resC["Y"], 'm-', lw=1.5, label="EKF Augmented + Bias (6-state)")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("Quỹ đạo X-Y — so sánh 5 phương pháp")
ax.legend(fontsize=8); ax.axis("equal"); ax.grid(alpha=0.3)

# (2) Yaw theo thời gian
ax = axes[0,1]
ax.plot(gt["t"], np.degrees(gt["yaw_true"]), 'k-', lw=2.5, label="Ground Truth")
ax.plot(res0["t"], np.degrees(res0["yaw"]), 'g:', lw=1.5, label="Odometry thuần")
ax.plot(resA["t"], np.degrees(resA["yaw"]), 'r-.', lw=1.3, label="EKF Cổ điển")
ax.plot(resB["t"], np.degrees(resB["yaw"]), 'b--', lw=1.3, label="EKF Augmented (không bias)")
ax.plot(resC["t"], np.degrees(resC["yaw"]), 'm-', lw=1.5, label="EKF Augmented + Bias")
ax.axvspan(10.0, 10.3, color='orange', alpha=0.3, label="Wheel-slip event")
ax.set_xlabel("t (s)"); ax.set_ylabel("Yaw (độ)"); ax.set_title("Yaw theo thời gian")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (3) Bias được ƯỚC LƯỢNG bởi EKF 6-state, so với bias THẬT đã tiêm vào
ax = axes[1,0]
ax.plot(resC["t"], resC["bias"], 'm-', lw=1.5, label="Bias ước lượng bởi EKF")
ax.axhline(0.006, color='k', linestyle='--', lw=1.5, label="Bias THẬT đã tiêm (0.006 rad/s)")
ax.set_xlabel("t (s)"); ax.set_ylabel("Gyro bias (rad/s)")
ax.set_title("EKF có tự ước lượng đúng bias không?")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (4) Sai số Yaw tuyệt đối theo thời gian - so 3 phương pháp EKF
ax = axes[1,1]
gtA = np.interp(resA["t"], gt["t"], gt["yaw_true"])
gtB = np.interp(resB["t"], gt["t"], gt["yaw_true"])
gtC = np.interp(resC["t"], gt["t"], gt["yaw_true"])
ax.plot(resA["t"], np.degrees(np.abs(resA["yaw"]-gtA)), 'r-.', lw=1.2, label="EKF Cổ điển")
ax.plot(resB["t"], np.degrees(np.abs(resB["yaw"]-gtB)), 'b--', lw=1.2, label="EKF Augmented (không bias)")
ax.plot(resC["t"], np.degrees(np.abs(resC["yaw"]-gtC)), 'm-', lw=1.5, label="EKF Augmented + Bias")
ax.set_xlabel("t (s)"); ax.set_ylabel("|Sai số Yaw| (độ)")
ax.set_title("Sai số Yaw tuyệt đối theo thời gian")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ekf_comparison.png", dpi=130)
print("Đã lưu ekf_comparison.png")

# ============ Bảng RMSE tổng kết cho CẢ 4 phương pháp ============
def rmse_xy_yaw(res, gt):
    gtx = np.interp(res["t"], gt["t"], gt["X_true"])
    gty = np.interp(res["t"], gt["t"], gt["Y_true"])
    gtyaw = np.interp(res["t"], gt["t"], gt["yaw_true"])
    return (np.sqrt(np.mean((res["X"]-gtx)**2)),
            np.sqrt(np.mean((res["Y"]-gty)**2)),
            np.degrees(np.sqrt(np.mean((res["yaw"]-gtyaw)**2))))

print("\n=== RMSE TOÀN QUỸ ĐẠO — càng nhỏ càng tốt ===")
print(f"{'Phương pháp':35s} {'RMSE_X(m)':>10s} {'RMSE_Y(m)':>10s} {'RMSE_yaw(°)':>12s}")
for name, res in [("Odometry thuần", res0), ("EKF Cổ điển", resA),
                   ("EKF Augmented (không bias)", resB), ("EKF Augmented + Bias", resC)]:
    ex, ey, eyaw = rmse_xy_yaw(res, gt)
    print(f"{name:35s} {ex:10.4f} {ey:10.4f} {eyaw:12.3f}")

resA.to_csv("result_classical_ekf.csv", index=False)
resB.to_csv("result_augmented_ekf.csv", index=False)
resC.to_csv("result_augmented_bias_ekf.csv", index=False)
res0.to_csv("result_odometry_only.csv", index=False)
