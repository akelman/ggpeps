import matplotlib.pyplot as plt


class Args:
    def __init__(self, obs, fname):
        self.obs = obs
        self.fnames = fname
        self.obs = obs


def save_plot(func, fname, obs, name):
    args = Args(obs, fname)
    func(args, save_path=name)


if __name__ == "__main__":
    import glob
    import plot_binning_analysis
    import plot_timeseries

    # base_path = "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_2.5_el_1.2500_mag_0.2000_int_1.0_mass_1.0/"  # g =2.5
    # base_path = "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/"  # g = 0.5
    # base_path = "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_1.0_el_0.5000_mag_0.5000_int_1.0_mass_1.0/"  # g = 1.0
    # pattern_L6 = base_path + "L_6/L_*/**/*.gz"
    # pattern_L4 = base_path + "L_4/L_*/**/*.gz"
    # pattern_L2 = base_path + "L_2/L_*/**/*.gz"

    # file_list_L6 = glob.glob(pattern_L6, recursive=True)  # List of all .gz files in L_6
    # file_list_L4 = glob.glob(pattern_L4, recursive=True)  # List of all .gz files in L_4
    # file_list_L2 = glob.glob(pattern_L2, recursive=True)  # List of all .gz files in L_2

    # fname_lst = [file_list_L6, file_list_L4, file_list_L2]
    obs_lst = ["energy", "mag_energy", "el_energy", "mass_energy", "int_energy"]

    fname_lst = [
        [
            "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6/L_6_gf_T/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
            "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_T/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
        ],
        [
            "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6/L_6_gf_F/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
            "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_F/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
        ],
    ]
    for obs in obs_lst:
        save_plot(
            plot_timeseries.main,
            fname_lst[0],
            obs,
            "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_T/"
            + obs
            + "_gf_T_timeseries.png",
        )
        save_plot(
            plot_timeseries.main,
            fname_lst[1],
            obs,
            "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_F/"
            + obs
            + "_gf_F_timeseries.png",
        )
