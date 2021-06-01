In order to ensure that the incremental update is implemented correctly, we can compare the output of the incremental and the non-incremental update.
While the weights may be different, the differences have to stay the same and the expectation values have to be numerically identical.
We are not changing the MC procedure, we only change the convention of calculating and storing the weights.

============================= NO INCREMENTAL UPDATE=============================

python manager.py eval 2 --meas_steps 10 --warmup_steps 10 --seed 42
2021-06-01 09:43:43,142 [INFO] Git hash: f1152fe65b040d322bc07d650ffa676ca2e5d3ff
2021-06-01 09:43:43,142 [INFO] ========= MC INFO ==========
2021-06-01 09:43:43,142 [INFO] Seed: 42
2021-06-01 09:43:43,142 [INFO] Warmup steps: 10
2021-06-01 09:43:43,142 [INFO] Measurement steps: 10
2021-06-01 09:43:43,143 [INFO] ============================
2021-06-01 09:43:43,143 [INFO] ======= SYSTEM INFO ========
2021-06-01 09:43:43,143 [INFO] L: 2
2021-06-01 09:43:43,143 [INFO] t: 0.0
2021-06-01 09:43:43,143 [INFO] y: 0.5
2021-06-01 09:43:43,143 [INFO] z: 0.5
2021-06-01 09:43:43,143 [INFO] g^2: 1.0
2021-06-01 09:43:43,143 [INFO] g_mag: None
2021-06-01 09:43:43,143 [INFO] g_gm: 0.0
2021-06-01 09:43:43,143 [INFO] ============================
link: 06, theta: 3.142, weight old: -10.429, weight new: -8.715, weight delta: 1.714, accept
link: 02, theta: 3.142, weight old: -8.715, weight new: -10.000, weight delta: -1.285, decline
link: 06, theta: 3.142, weight old: -8.715, weight new: -8.715, weight delta: 0.000, accept
link: 02, theta: 0.000, weight old: -8.715, weight new: -8.715, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: -8.715, weight new: -10.000, weight delta: -1.285, decline
link: 05, theta: 0.000, weight old: -8.715, weight new: -8.715, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: -8.715, weight new: -10.000, weight delta: -1.285, accept
link: 07, theta: 3.142, weight old: -10.000, weight new: -11.126, weight delta: -1.126, accept
link: 03, theta: 3.142, weight old: -11.126, weight new: -11.126, weight delta: 0.000, accept
link: 03, theta: 0.000, weight old: -11.126, weight new: -8.907, weight delta: 2.219, accept
link: 02, theta: 0.000, weight old: -8.907, weight new: -8.907, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: -8.907, weight new: -11.126, weight delta: -2.219, decline
link: 05, theta: 3.142, weight old: -8.907, weight new: -8.715, weight delta: 0.192, accept
link: 02, theta: 3.142, weight old: -8.715, weight new: -10.000, weight delta: -1.285, decline
link: 07, theta: 0.000, weight old: -8.715, weight new: -8.522, weight delta: 0.192, accept
link: 02, theta: 0.000, weight old: -8.522, weight new: -8.522, weight delta: 0.000, accept
link: 06, theta: 3.142, weight old: -8.522, weight new: -8.522, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: -8.522, weight new: -8.715, weight delta: -0.192, accept
link: 00, theta: 3.142, weight old: -8.715, weight new: -8.907, weight delta: -0.192, accept
link: 03, theta: 3.142, weight old: -8.907, weight new: -8.907, weight delta: 0.000, accept
<acceptance_prob> 0.8
<energy> 8.073624053499863
<mag_energy> 6.4
<el_energy> 1.6736240534998632
<wilson_00_11> -0.6
<polyakov_00_x> 0.6
2021-06-01 09:43:43,175 [INFO] ==== Acceptance prob =======
2021-06-01 09:43:43,175 [INFO] Acceptance probability: 0.8
2021-06-01 09:43:43,175 [INFO] ============================
2021-06-01 09:43:43,175 [INFO] ========== TIME ============
2021-06-01 09:43:43,175 [INFO] The simulation took 0.02190792141482234s
2021-06-01 09:43:43,175 [INFO] ============================

============================= NO INCREMENTAL UPDATE END =============================

============================= ====INCREMENTAL UPDATE ===============================
 python manager.py eval 2 --meas_steps 10 --warmup_steps 10 --seed 42
2021-06-01 09:44:01,444 [INFO] Git hash: 96f8491a932f1a08fae39be5b09e5e11cda7a85d
2021-06-01 09:44:01,444 [INFO] ========= MC INFO ==========
2021-06-01 09:44:01,444 [INFO] Seed: 42
2021-06-01 09:44:01,444 [INFO] Warmup steps: 10
2021-06-01 09:44:01,444 [INFO] Measurement steps: 10
2021-06-01 09:44:01,444 [INFO] ============================
2021-06-01 09:44:01,444 [INFO] ======= SYSTEM INFO ========
2021-06-01 09:44:01,444 [INFO] L: 2
2021-06-01 09:44:01,444 [INFO] t: 0.0
2021-06-01 09:44:01,444 [INFO] y: 0.5
2021-06-01 09:44:01,444 [INFO] z: 0.5
2021-06-01 09:44:01,444 [INFO] g^2: 1.0
2021-06-01 09:44:01,444 [INFO] g_mag: None
2021-06-01 09:44:01,445 [INFO] g_gm: 0.0
2021-06-01 09:44:01,445 [INFO] ============================
Weight recalculated!
link: 06, theta: 3.142, weight old: 0.661, weight new: 2.375, weight delta: 1.714, accept
link: 02, theta: 3.142, weight old: 2.375, weight new: 1.090, weight delta: -1.285, decline
link: 06, theta: 3.142, weight old: 2.375, weight new: 2.375, weight delta: 0.000, accept
link: 02, theta: 0.000, weight old: 2.375, weight new: 2.375, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: 2.375, weight new: 1.090, weight delta: -1.285, decline
link: 05, theta: 0.000, weight old: 2.375, weight new: 2.375, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: 2.375, weight new: 1.090, weight delta: -1.285, accept
link: 07, theta: 3.142, weight old: 1.090, weight new: -0.036, weight delta: -1.126, accept
link: 03, theta: 3.142, weight old: -0.036, weight new: -0.036, weight delta: 0.000, accept
link: 03, theta: 0.000, weight old: -0.036, weight new: 2.183, weight delta: 2.219, accept
link: 02, theta: 0.000, weight old: 2.183, weight new: 2.183, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: 2.183, weight new: -0.036, weight delta: -2.219, decline
link: 05, theta: 3.142, weight old: 2.183, weight new: 2.375, weight delta: 0.192, accept
link: 02, theta: 3.142, weight old: 2.375, weight new: 1.090, weight delta: -1.285, decline
link: 07, theta: 0.000, weight old: 2.375, weight new: 2.568, weight delta: 0.192, accept
link: 02, theta: 0.000, weight old: 2.568, weight new: 2.568, weight delta: 0.000, accept
link: 06, theta: 3.142, weight old: 2.568, weight new: 2.568, weight delta: 0.000, accept
link: 03, theta: 3.142, weight old: 2.568, weight new: 2.375, weight delta: -0.192, accept
link: 00, theta: 3.142, weight old: 2.375, weight new: 2.183, weight delta: -0.192, accept
link: 03, theta: 3.142, weight old: 2.183, weight new: 2.183, weight delta: 0.000, accept
<acceptance_prob> 0.8
<energy> 8.073624053499863
<mag_energy> 6.4
<el_energy> 1.6736240534998632
<wilson_00_11> -0.6
<polyakov_00_x> 0.6
2021-06-01 09:44:01,491 [INFO] ==== Acceptance prob =======
2021-06-01 09:44:01,492 [INFO] Acceptance probability: 0.8
2021-06-01 09:44:01,492 [INFO] ============================
2021-06-01 09:44:01,492 [INFO] ========== TIME ============
2021-06-01 09:44:01,492 [INFO] The simulation took 0.03438116190955043s
2021-06-01 09:44:01,492 [INFO] ============================

============================= ====INCREMENTAL UPDATE END===============================