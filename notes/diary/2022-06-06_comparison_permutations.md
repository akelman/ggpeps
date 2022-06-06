Check of the change of the permutation matrices

The generation of permutation matrices can be simplified. 
Instead of generating the permutation matrices by hand, we can simply give the input and output order of the modes and generate the permutation matrices automatically.
This approach is more human readable and comes without (almost) any runtime penalty since it will be only executed once.

We check that the two approaches are equivalent by comparing a run from the master branch (old permutation matrices) and a run from the mode_array branch (new permutation matrices).

Call: python manager.py exact 2 --ncopy 1 --nlayer 1 --seed 10101 --params 0.0 0.1 0.3 --pure-gauge

=================================================== master ====================================================

2022-06-06 11:27:30,245 [INFO] Git hash: 57c363a23849b2c55adfd99aaba33c99ab08d2ac
2022-06-06 11:27:30,245 [INFO] ========= MC INFO ==========
2022-06-06 11:27:30,245 [INFO] Seed: 10101
2022-06-06 11:27:30,245 [INFO] Warmup steps: 100000
2022-06-06 11:27:30,245 [INFO] Measurement steps: 100000
2022-06-06 11:27:30,246 [INFO] ============================
2022-06-06 11:27:30,246 [INFO] ======= SYSTEM INFO ========
2022-06-06 11:27:30,246 [INFO] L: 2
2022-06-06 11:27:30,246 [INFO] # of layers: 1
2022-06-06 11:27:30,246 [INFO] # of copies: 1
2022-06-06 11:27:30,247 [INFO] parameters: [[0.  0.1 0.3]]
2022-06-06 11:27:30,247 [INFO] g^2: 1.0
2022-06-06 11:27:30,247 [INFO] g^2_mag: None
2022-06-06 11:27:30,247 [INFO] g_gm: 0.0
2022-06-06 11:27:30,247 [INFO] Rebinning EOM: True
2022-06-06 11:27:30,247 [INFO] ============================
energy: 4.072364722631162
mag_energy: 4.062254366783926
el_energy: 0.010110355847236166
wilson_00_11: -0.01556359169598154
polyakov_00_x: 0.019976131650077306
grad_norm: [[ 0.         -1.12755423 -7.99162735]]
norm: 0.25158321005882434
mag_energy_grad: [[-0.         -0.0516987   0.83121559]]
el_energy_grad: [[-0.          0.06813144  0.22579896]]
energy_grad: [[-0.          0.01643273  1.05701455]]
2022-06-06 11:27:33,723 [INFO] ========== TIME ============
2022-06-06 11:27:33,724 [INFO] The simulation took 3.4707208820000233s
2022-06-06 11:27:33,724 [INFO] ============================


=================================================== mode_array ====================================================
2022-06-06 11:22:21,125 [INFO] Git hash: 9c05aa9bf1f53fae0d11485db4cba605e33f6dd7
2022-06-06 11:22:21,126 [INFO] ========= MC INFO ==========
2022-06-06 11:22:21,126 [INFO] Seed: 10101
2022-06-06 11:22:21,126 [INFO] Warmup steps: 100000
2022-06-06 11:22:21,126 [INFO] Measurement steps: 100000
2022-06-06 11:22:21,126 [INFO] ============================
2022-06-06 11:22:21,127 [INFO] ======= SYSTEM INFO ========
2022-06-06 11:22:21,127 [INFO] L: 2
2022-06-06 11:22:21,127 [INFO] # of layers: 1
2022-06-06 11:22:21,127 [INFO] # of copies: 1
2022-06-06 11:22:21,128 [INFO] parameters: [[0.  0.1 0.3]]
2022-06-06 11:22:21,128 [INFO] g^2: 1.0
2022-06-06 11:22:21,128 [INFO] g^2_mag: None
2022-06-06 11:22:21,128 [INFO] g_gm: 0.0
2022-06-06 11:22:21,128 [INFO] Rebinning EOM: True
2022-06-06 11:22:21,128 [INFO] ============================
energy: 4.072364722631162
mag_energy: 4.062254366783926
el_energy: 0.010110355847236166
wilson_00_11: -0.01556359169598154
polyakov_00_x: 0.019976131650077306
grad_norm: [[ 0.         -1.12755423 -7.99162735]]
norm: 0.25158321005882434
mag_energy_grad: [[-0.         -0.0516987   0.83121559]]
el_energy_grad: [[-0.          0.06813144  0.22579896]]
energy_grad: [[-0.          0.01643273  1.05701455]]
2022-06-06 11:22:26,316 [INFO] ========== TIME ============
2022-06-06 11:22:26,316 [INFO] The simulation took 5.176298364999866s
2022-06-06 11:22:26,316 [INFO] ============================

============================================ END OF LOG =========================================================

