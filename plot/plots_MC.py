import matplotlib.pyplot as plt


class Args:
    def __init__(self, obs, fname):
        self.obs = obs
        self.fname = fname
        self.obs = obs


def save_plot(func, fname, obs, name):
    args = Args(obs, fname)
    func(args, save_path=name)


if __name__ == "__main__":
    import glob
    import plot_binning_analysis
    import plot_timeseries

    base_path_single = (
        "G:/My Drive/Research/MC/trans_inv_mag_with_grad/single_plaquette/"  # g =2.5
    )
    base_path_all = "G:/My Drive/Research/MC/trans_inv_mag_with_grad/all_plaquettes/"
    # base_path = "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/"  # g = 0.5
    # base_path = "G:/My Drive/Research/MC/rows_gauge_test/g_1.0_el_0.5000_mag_0.5000_int_1.0_mass_1.0/"  # g = 1.0
    pattern_single = base_path_single + "/L_*/*.gz"
    pattern_all = base_path_all + "/L_*/*.gz"

    file_list_single = glob.glob(
        pattern_single, recursive=True
    )  # List of all .gz files in L_6
    file_list_all = glob.glob(
        pattern_all, recursive=True
    )  # List of all .gz files in L_4

    fname_lst = [
        [file_list_single[i], file_list_all[i]] for i in range(len(file_list_all))
    ]
    obs_lst = [
        "energy_grad",
        "energy",
        "mag_energy",
        "el_energy",
        "mass_energy",
        "int_energy",
    ]
    for fname in fname_lst:
        for obs in obs_lst:
            save_plot(
                plot_binning_analysis.main,
                fname,
                obs,
                fname[0][0 : len(base_path_single) + 9]
                + "\\"
                + obs
                + "_binning_analysis_600.pdf",
            )
    # with specific file names
    # fname_lst = [
    #     [
    #         "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6/L_6_gf_T/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
    #         "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_T/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
    #     ],
    #     [
    #         "G:/My Drive/Research/MC/test_gauge_fixing_and_system_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6/L_6_gf_F/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
    #         "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_F/data_mc_L_06-06_gel_0.250_gmag_1.000_gint_1.000_nlayer_02_wsteps_0100000_msteps_0100000.pkl.gz",
    #     ],
    # ]
    # for obs in obs_lst:
    #     save_plot(
    #         plot_timeseries.main,
    #         fname_lst[0],
    #         obs,
    #         "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_T/"
    #         + obs
    #         + "_gf_T_timeseries.png",
    #     )
    #     save_plot(
    #         plot_timeseries.main,
    #         fname_lst[1],
    #         obs,
    #         "G:/My Drive/Research/MC/test_step_size/g_0.5_el_0.2500_mag_1.0000_int_1.0_mass_1.0/L_6_update_size_10_gf_F/"
    #         + obs
    #         + "_gf_F_timeseries.png",
    #     )
