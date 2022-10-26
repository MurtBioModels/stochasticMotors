def rl_fu_bead2(family, kt, n_motors, n_it, epsilon, stat="probability"):
    """


    Parameters
    ----------

    Returns
    -------

    """
    # Dictionaries storing data (=values) per number of motors (=key)
    dict_run_length_bead = {}
    # Loop over range of motor team sizes
    for n in range(1, n_motors + 1):

        # Unpickle motor_0 object
        pickle_file_motor0 = open(f'Motor0_{family}_{epsilon}_{kt}kt_{n_it}it_{n}motors', 'rb')
        motor0 = pickle.load(pickle_file_motor0)
        pickle_file_motor0.close()
        #
        list_xbead = motor0.x_bead
        flattened_list = [val for sublist in list_xbead for val in sublist]
        size = len(flattened_list)
        index_list = [i + 1 for i, val in enumerate(flattened_list) if val == 0]
        split_list = [flattened_list[start: end] for start, end in
                      zip([0] + index_list, index_list + ([size] if index_list[-1] != size else []))]
        # print(split_list)
        list_rl = []
        for run in split_list:
            list_rl.append(max(run))
        list_rl = [i for i in list_rl if i != 0]
        dict_run_length_bead[n] = list_rl

    # Remove empty keys first
    # print(dict_run_length_bead)
    new_dict = {k: v for k, v in dict_run_length_bead.items() if v}

    # Transform dictionary to Pandas dataframe with equal dimensions
    df_run_lengths = pd.DataFrame({key: pd.Series(value) for key, value in new_dict.items()})
    # Melt columns (drop NaN rows)
    melted_run_lengths = pd.melt(df_run_lengths, value_vars=df_run_lengths.columns, var_name='N_motors').dropna()
    print(melted_run_lengths)

    # Plot distributions per size of motor team (N motors)
    g = sns.FacetGrid(melted_run_lengths, col="N_motors", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, binwidth=1, common_norm=False)
    g.set_axis_labels("Run length of bead [x]")
    g.set_titles("{col_name}")
    g.set(xticks=np.arange(0, 900, 100))
    g.set(xlim=(0, 900))
    plt.savefig(f'DistRLBead_{family}_{epsilon}_{kt}Kt.png')
    plt.show()

    return


def dist_act_motors2(family, kt, n_motors, n_it, epsilon, stat="probability"):
    """


    Parameters
    ----------

    Returns
    -------

    """
    # Dictionaries storing data (=values) per number of motors (=key)
    dict_active_motors = {}
    # Loop over range of motor team sizes
    for n in range(1, n_motors + 1):
        # Unpickle motor_0 object
        pickle_file_motor0 = open(f'Motor0_{family}_{epsilon}_{kt}kt_{n_it}it_{n}motors', 'rb')
        motor0 = pickle.load(pickle_file_motor0)
        pickle_file_motor0.close()

        # Append attribute 'bound motors': list of active motors per time step
        dict_active_motors[n] = motor0.antero_motors

    # Transform dictionary to Pandas dataframe with equal dimensions
    df_bound_motors = pd.DataFrame({key: pd.Series(value) for key, value in dict_active_motors.items()})
    # Melt columns (drop NaN rows)
    melted_bound_motors = pd.melt(df_bound_motors, value_vars=df_bound_motors.columns, var_name='N_motors').dropna()
    print(melted_bound_motors.head())

    # Plot distributions per size of motor team (N motors)
    g = sns.FacetGrid(melted_bound_motors, col="N_motors", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, discrete=True, common_norm=False)
    g.set_axis_labels("Active motors")
    g.set_titles("{col_name}")
    g.set(xticks=np.arange(0, 11, 1))
    plt.savefig(f'DistActiveMotors_{family}_{epsilon}_{kt}Kt.png')
    plt.show()

    return
