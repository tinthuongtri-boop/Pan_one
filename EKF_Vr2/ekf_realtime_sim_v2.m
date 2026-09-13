%% ============================================================
%  EKF REAL-TIME SIMULATION v2 - So sanh 4 phuong phap
%  Differential-drive robot: Servo Controller (wheel) + IMU (gyro yaw rate)
% ============================================================
%
%  4 phuong phap chay SONG SONG tren CUNG 1 du lieu mo phong, ve REAL-TIME:
%   (0) Odometry thuan     - khong Kalman, khong fuse gi ca
%   (A) EKF Co dien        - 3-state [X,Y,yaw], wheel = control input,
%                            IMU tich phan tich luy sua Yaw tuyet doi
%   (B) EKF Augmented      - 5-state [X,Y,yaw,vx,vyaw], KHONG uoc luong bias
%   (C) EKF Augmented+Bias - 6-state [X,Y,yaw,vx,vyaw,bias], CO gating
%                            Mahalanobis de bao ve state bias khoi bi
%                            "ro ri" boi wheel-slip (xem giai thich trong
%                            hoi thoai - day la diem quan trong nhat)
%
%  [ASSUMPTION] Toan bo thong so vat ly (L, r) va nhieu duoi day la GIA TRI
%  MINH HOA - PHAI thay bang so do thuc nghiem/datasheet that cua robot ban
%  truoc khi dung ket qua nay de ket luan ve he thong that.

clear; clc; close all;
rng(42);

%% ============ 1. THAM SO ============
L = 0.30; r = 0.05;
SERVO_RATE = 100; IMU_RATE = 50;
ENC_RPM_NOISE_STD = 1.2;
GYRO_NOISE_STD    = 0.008;
GYRO_BIAS         = 0.006;
GATE_THRESHOLD_WHEEL = 9.0;   % Mahalanobis^2, [ASSUMPTION] ~99% cho 2-DOF

REALTIME_PLAYBACK = true;
SPEEDUP    = 1.0;
PLOT_EVERY = 4;
T_end = 20;

%% ============ 2. QUY DAO CHO SAN (Ground Truth) ============
dt_gt = 0.01;
t_gt  = 0:dt_gt:T_end;
v_gt  = arrayfun(@vProfile, t_gt);
w_gt  = arrayfun(@omegaProfile, t_gt);
X_gt = zeros(size(t_gt)); Y_gt = zeros(size(t_gt)); YAW_gt = zeros(size(t_gt));
for i = 2:length(t_gt)
    X_gt(i)   = X_gt(i-1)   + v_gt(i-1)*cos(YAW_gt(i-1))*dt_gt;
    Y_gt(i)   = Y_gt(i-1)   + v_gt(i-1)*sin(YAW_gt(i-1))*dt_gt;
    YAW_gt(i) = YAW_gt(i-1) + w_gt(i-1)*dt_gt;
end

%% ============ 3. Banh trai/phai THAT + tiem Wheel-Slip ============
vL_true = v_gt - w_gt*L/2;
vR_true = v_gt + w_gt*L/2;
slipMask = (t_gt >= 10.0) & (t_gt < 10.3);
vL_true(slipMask) = vL_true(slipMask) - 0.15;

%% ============ 4. Lay mau + nhieu (dung tan so tung cam bien) ============
[t_wheel, rpmL_meas] = sampleAndNoise(t_gt, vToRpm(vL_true, r), SERVO_RATE, ENC_RPM_NOISE_STD, 0);
[~,       rpmR_meas] = sampleAndNoise(t_gt, vToRpm(vR_true, r), SERVO_RATE, ENC_RPM_NOISE_STD, 0);
[t_imu,   gyro_meas ] = sampleAndNoise(t_gt, w_gt,               IMU_RATE,   GYRO_NOISE_STD, GYRO_BIAS);

vL_c = rpmToV(rpmL_meas, r); vR_c = rpmToV(rpmR_meas, r);
vx_wheel_meas   = (vL_c + vR_c)/2;
vyaw_wheel_meas = (vR_c - vL_c)/L;

sigma_v        = ENC_RPM_NOISE_STD*(2*pi/60*r);
vx_var_wheel   = 2*(sigma_v/2)^2;
vyaw_var_wheel = 2*(sigma_v/L)^2;
vyaw_var_imu   = GYRO_NOISE_STD^2;

%% ============ 5. Gop + sap xep su kien theo thoi gian ============
nW = numel(t_wheel); nI = numel(t_imu);
evT     = [t_wheel(:); t_imu(:)];
evType  = [repmat({'wheel'}, nW,1); repmat({'imu'}, nI,1)];
evVx    = [vx_wheel_meas(:); nan(nI,1)];
evVyaw  = [vyaw_wheel_meas(:); gyro_meas(:)];
evVxVar = [repmat(vx_var_wheel, nW,1); nan(nI,1)];
evVyVar = [repmat(vyaw_var_wheel, nW,1); repmat(vyaw_var_imu, nI,1)];
[evT, sidx] = sort(evT);
evType=evType(sidx); evVx=evVx(sidx); evVyaw=evVyaw(sidx);
evVxVar=evVxVar(sidx); evVyVar=evVyVar(sidx);
N = numel(evT);

%% ============ 6. KHOI TAO CA 4 PHUONG PHAP ============
% (0) Odometry thuan
s0 = [0;0;0]; last_t0 = 0;

% (A) EKF Co dien - 3 state, wheel=input, IMU sua Yaw tuyet doi qua tich luy
sA = [0;0;0]; PA = diag([1e-6 1e-6 1e-6]);
Rp_A = diag([0.02 0.02 0.01]); Qm_A = 0.02^2;
thetaGyroLocal = 0; lastImuA = 0; lastTA = 0;

% (B) EKF Augmented - 5 state, KHONG bias
sB = [0;0;0;0;0]; PB = diag([1e-6 1e-6 1e-6 1e-6 1e-6]);
Rp_B = diag([0.001 0.001 0.0005 0.025 0.02]);
lastTB = 0;

% (C) EKF Augmented + Bias - 6 state, CO gating bao ve bias
sC = [0;0;0;0;0;0]; PC = diag([1e-6 1e-6 1e-6 1e-6 1e-6 0.01]);
Rp_C = diag([0.001 0.001 0.0005 0.025 0.02 1e-7]);
lastTC = 0; nRejected = 0;

hist_t=zeros(N,1); hist0=zeros(N,3); histA=zeros(N,3);
histB=zeros(N,5); histC=zeros(N,6);

%% ============ 7. FIGURE 1: QUY DAO (giong phong cach anh tham khao) ============
fig1 = figure('Name','Quy dao Real-time - So sanh 4 phuong phap','Position',[50 50 750 650]);
plot(X_gt, Y_gt, 'k-', 'LineWidth', 2.5); hold on; grid on; axis equal;
h0 = animatedline('Color',[0 0.6 0],'LineStyle',':','LineWidth',1.5);
hA = animatedline('Color','r','LineStyle','-.','LineWidth',1.3);
hB = animatedline('Color','b','LineStyle','--','LineWidth',1.3);
hC = animatedline('Color','m','LineStyle','-','LineWidth',1.8);
hPose = plot(0,0,'ko','MarkerFaceColor','y','MarkerSize',9);
legend('Ground Truth','Odometry thuan','EKF Co dien','EKF Augmented (khong bias)', ...
       'EKF Augmented + Bias','Vi tri hien tai (EKF+Bias)','Location','best');
xlabel('X (m)'); ylabel('Y (m)');
title('Quy dao Real-time - So sanh 4 phuong phap');

%% ============ 8. FIGURE 2: 5 STATE THEO THOI GIAN ============
fig2 = figure('Name','State theo thoi gian','Position',[820 50 700 850]);
labels = {'X (m)','Y (m)','yaw (rad)','v_x (m/s)','v_{yaw} (rad/s)'};
hGT2=gobjects(1,5); hB2=gobjects(1,5); hC2=gobjects(1,5);
for k=1:5
    subplot(5,1,k); hold on; grid on; xlim([0 T_end]);
    hGT2(k)=animatedline('Color','k','LineWidth',1.5);
    hB2(k) =animatedline('Color','b','LineWidth',1,'LineStyle','--');
    hC2(k) =animatedline('Color','m','LineWidth',1.3);
    ylabel(labels{k});
    if k==1, title('Den=Ground Truth | Xanh net dut=EKF Augmented | Tim=EKF+Bias'); end
    if k==5, xlabel('t (s)'); end
end

%% ============ 9. VONG LAP REAL-TIME ============
for i = 1:N
    t = evT(i);
    src = evType{i};

    % ---------- (0) Odometry thuan - CHI cap nhat khi co wheel ----------
    if strcmp(src,'wheel')
        dt0 = t - last_t0; last_t0 = t;
        s0 = s0 + [evVx(i)*cos(s0(3))*dt0; evVx(i)*sin(s0(3))*dt0; evVyaw(i)*dt0];
    end

    % ---------- (A) EKF Co dien ----------
    dtA = t - lastTA; if dtA<0, dtA=0; end; lastTA = t;
    if strcmp(src,'wheel')
        GA = jacobianG_A(sA, evVx(i), dtA);
        sA = sA + [evVx(i)*cos(sA(3))*dtA; evVx(i)*sin(sA(3))*dtA; evVyaw(i)*dtA];
        PA = GA*PA*GA' + Rp_A*dtA;
    else
        dtImuA = t - lastImuA; lastImuA = t;
        thetaGyroLocal = thetaGyroLocal + evVyaw(i)*dtImuA;
        H = [0 0 1];
        yInno = thetaGyroLocal - sA(3);
        S = H*PA*H' + Qm_A;
        K = PA*H'/S;
        sA = sA + K*yInno;
        PA = (eye(3)-K*H)*PA;
    end

    % ---------- (B) EKF Augmented (khong bias) ----------
    dtB = t - lastTB; if dtB<0, dtB=0; end; lastTB = t;
    [sB_pred, GB] = predictAugmented(sB, dtB);
    PB_pred = GB*PB*GB' + Rp_B*dtB;
    if strcmp(src,'wheel')
        H=[0 0 0 1 0;0 0 0 0 1]; z=[evVx(i);evVyaw(i)]; Qm=diag([evVxVar(i),evVyVar(i)]);
    else
        H=[0 0 0 0 1]; z=evVyaw(i); Qm=evVyVar(i);
    end
    yInno = z - H*sB_pred; S = H*PB_pred*H' + Qm; K = PB_pred*H'/S;
    sB = sB_pred + K*yInno; PB = (eye(5)-K*H)*PB_pred;

    % ---------- (C) EKF Augmented + Bias (CO gating) ----------
    dtC = t - lastTC; if dtC<0, dtC=0; end; lastTC = t;
    [sC_pred, GC] = predictAugmentedBias(sC, dtC);
    PC_pred = GC*PC*GC' + Rp_C*dtC;
    if strcmp(src,'wheel')
        H=[0 0 0 1 0 0;0 0 0 0 1 0]; z=[evVx(i);evVyaw(i)]; Qm=diag([evVxVar(i),evVyVar(i)]);
    else
        H=[0 0 0 0 1 1]; z=evVyaw(i); Qm=evVyVar(i);   % gyro THO = vyaw + bias
    end
    yInno = z - H*sC_pred; S = H*PC_pred*H' + Qm;
    d2 = yInno' * (S\yInno);    % Mahalanobis distance^2
    if strcmp(src,'wheel') && d2 > GATE_THRESHOLD_WHEEL
        nRejected = nRejected + 1;
        sC = sC_pred; PC = PC_pred;   % REJECT - chi giu Predict, khong Correct
    else
        K = PC_pred*H'/S;
        sC = sC_pred + K*yInno;
        PC = (eye(6)-K*H)*PC_pred;
    end

    hist_t(i)=t; hist0(i,:)=s0'; histA(i,:)=sA'; histB(i,:)=sB'; histC(i,:)=sC';

    % ---------- VE REAL-TIME ----------
    if mod(i,PLOT_EVERY)==0 || i==N
        figure(fig1);
        addpoints(h0, s0(1), s0(2)); addpoints(hA, sA(1), sA(2));
        addpoints(hB, sB(1), sB(2)); addpoints(hC, sC(1), sC(2));
        set(hPose,'XData',sC(1),'YData',sC(2));
        drawnow limitrate;

        gtIdx = min(round(t/dt_gt)+1, numel(t_gt));
        vals_gt = [X_gt(gtIdx) Y_gt(gtIdx) YAW_gt(gtIdx) v_gt(gtIdx) w_gt(gtIdx)];
        figure(fig2);
        for k=1:5
            addpoints(hGT2(k), t, vals_gt(k));
            addpoints(hB2(k),  t, sB(k));
            addpoints(hC2(k),  t, sC(k));
        end
        drawnow limitrate;
    end

    if REALTIME_PLAYBACK && i<N
        pause(max(0,(evT(i+1)-evT(i))/SPEEDUP));
    end
end

fprintf('\n[Gating] So lan wheel measurement bi REJECT (Mahalanobis^2 > %.1f): %d\n', ...
    GATE_THRESHOLD_WHEEL, nRejected);

%% ============ 10. THONG KE RMSE + LUU KET QUA ============
gt_i = interp1(t_gt, [X_gt' Y_gt' YAW_gt'], hist_t);
rmse0 = sqrt(mean((hist0-gt_i).^2,1));
rmseA = sqrt(mean((histA-gt_i).^2,1));
rmseB = sqrt(mean((histB(:,1:3)-gt_i).^2,1));
rmseC = sqrt(mean((histC(:,1:3)-gt_i).^2,1));

fprintf('\n=== RMSE (X,Y,yaw[rad]) ===\n');
fprintf('%-28s X=%.4f  Y=%.4f  yaw=%.4f rad (%.2f deg)\n','Odometry thuan',rmse0(1),rmse0(2),rmse0(3),rad2deg(rmse0(3)));
fprintf('%-28s X=%.4f  Y=%.4f  yaw=%.4f rad (%.2f deg)\n','EKF Co dien',rmseA(1),rmseA(2),rmseA(3),rad2deg(rmseA(3)));
fprintf('%-28s X=%.4f  Y=%.4f  yaw=%.4f rad (%.2f deg)\n','EKF Augmented (khong bias)',rmseB(1),rmseB(2),rmseB(3),rad2deg(rmseB(3)));
fprintf('%-28s X=%.4f  Y=%.4f  yaw=%.4f rad (%.2f deg)\n','EKF Augmented + Bias',rmseC(1),rmseC(2),rmseC(3),rad2deg(rmseC(3)));

exportgraphics(fig1, 'matlab_trajectory_result.png', 'Resolution', 150);
exportgraphics(fig2, 'matlab_states_result.png', 'Resolution', 150);

T = array2table([hist_t hist0 histA histB histC gt_i], 'VariableNames', ...
    {'t','X0','Y0','yaw0','XA','YA','yawA','XB','YB','yawB','vxB','vyawB', ...
     'XC','YC','yawC','vxC','vyawC','biasC','X_gt','Y_gt','yaw_gt'});
writetable(T, 'matlab_ekf_result.csv');
save('matlab_ekf_result.mat','T','rmse0','rmseA','rmseB','rmseC');
fprintf('\nDa luu: matlab_trajectory_result.png, matlab_states_result.png, matlab_ekf_result.csv/.mat\n');

%% ============================================================
%  LOCAL FUNCTIONS
%% ============================================================
function v = vProfile(t)
    if t<2, v=0; elseif t<5, v=0.3*(t-2)/3; elseif t<15, v=0.3;
    elseif t<18, v=0.3*(1-(t-15)/3); else, v=0; end
end
function w = omegaProfile(t)
    if t>=10 && t<15, w=0.3; else, w=0; end
end
function rpm = vToRpm(v,r), rpm = v./r.*60/(2*pi); end
function v = rpmToV(rpm,r), v = rpm.*(2*pi/60).*r; end
function [ts,noisy] = sampleAndNoise(t_full,sig_full,rateHz,noiseStd,bias)
    ts = (0:1/rateHz:t_full(end)-1/rateHz)';
    sig = interp1(t_full, sig_full, ts);
    noisy = sig + bias + noiseStd*randn(size(sig));
end
function G = jacobianG_A(x, v, dt)
    yaw = x(3);
    G = [1 0 -v*sin(yaw)*dt; 0 1 v*cos(yaw)*dt; 0 0 1];
end
function [s_pred, G] = predictAugmented(s, dt)
    X=s(1);Y=s(2);yaw=s(3);vx=s(4);vyaw=s(5);
    s_pred = s + [vx*cos(yaw)*dt; vx*sin(yaw)*dt; vyaw*dt; 0; 0];
    G = eye(5);
    G(1,3)=-vx*sin(yaw)*dt; G(1,4)=cos(yaw)*dt;
    G(2,3)= vx*cos(yaw)*dt; G(2,4)=sin(yaw)*dt;
    G(3,5)=dt;
end
function [s_pred, G] = predictAugmentedBias(s, dt)
    X=s(1);Y=s(2);yaw=s(3);vx=s(4);vyaw=s(5);b=s(6);
    s_pred = s + [vx*cos(yaw)*dt; vx*sin(yaw)*dt; vyaw*dt; 0; 0; 0];
    G = eye(6);
    G(1,3)=-vx*sin(yaw)*dt; G(1,4)=cos(yaw)*dt;
    G(2,3)= vx*cos(yaw)*dt; G(2,4)=sin(yaw)*dt;
    G(3,5)=dt;
end
