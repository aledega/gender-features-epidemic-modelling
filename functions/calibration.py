# Libraries

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from model import get_beta, run_model

# global variables
age_groups = ['20-29', '30-39', '40-49', '50-59', '60+']


def wmape(observed, predicted):
    """
    Computes the Weighted Mean Absolute Percentage Error (WMAPE).
    :param: observed (array-like): The observed values.
    :param: predicted (array-like): The modeled values.
    :return: (float) The WMAPE value.
    """
    return np.sum(np.abs(observed - predicted)) / np.sum(np.abs(observed))


def rmse(observed, predicted):
    """
    Computes the Root Mean Squared Error (RMSE).
    :param: observed (array-like): The observed values.
    :param: predicted (array-like): The modeled values.
    :return: (float) The WMAPE value.
    """
    return np.sqrt(np.mean((observed - predicted)**2))


def get_weekly_new_deaths(y, initial_date, t_max, t_step, aggregation='total', n_comp_deaths=5):
    """
    Compute weekly new deaths from a stochastic model with flexible aggregation.
    :param: y (np.ndarray): Array of shape (n_age, n_gen, n_comp, t_max).
    :param: initial_date (datetime): Start date of simulation.
    :param: t_max (int): Number of time steps.
    :param: t_step (int): Step size in days.
    :param: aggregation (str): Level of aggregation. Options: 'total' (default), 'gender', 'age', 'age_gender'.
    :param: n_comp_deaths (int): Index of death compartment (default: 5).
    :return: tuple: (dates, data_structure) Return list of dates and weekly number of deaths. The structure of the data depends on 'aggregation'.
    """

    # --- 1. Compute Weeks for Filtering ---
    # initial_week: 2 weeks after initial_date (allow model to adjust)
    start_comparison_date = initial_date + timedelta(days=14)
    initial_week = start_comparison_date.strftime("%Y-%W")
    if initial_week.endswith('-00'):
        initial_week = f"{int(initial_week[:4]) - 1}-52"

    # final_week: based on t_max
    final_comparison_date = initial_date + timedelta(days=int(t_max * t_step))
    final_week = final_comparison_date.strftime("%Y-%W")
    if final_week.endswith('-00'):
        final_week = f"{int(final_week[:4]) - 1}-52"

    # --- 2. Compute Daily Deaths ---
    new_deaths_timestep = np.diff(y[:, :, n_comp_deaths, :], axis=-1)  # Shape: (n_age, n_gen, t_max-1)

    # --- 3. Create Base DataFrame ---
    # Generate dates list matching diff array length
    dates = [initial_date + timedelta(days=t_step * i) for i in range(1, t_max)]
    
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    
    # Create columns for every Age-Gender pair
    for i, age in enumerate(age_groups):
        for gen in [0, 1]:  # 0=Male, 1=Female
            col_name = f"{age}_{gen}"
            df[col_name] = new_deaths_timestep[i, gen, :]

    # --- 4. Weekly Grouping Logic ---
    df['Year-Week'] = df['Date'].dt.strftime('%Y-%W')
    
    # Fix Week '00' edge case
    df['Year-Week'] = df['Year-Week'].apply(lambda x: f"{int(x[:4]) - 1}-52" if x.endswith('-00') else x)

    # Filter by date range
    mask = (df['Year-Week'] >= initial_week) & (df['Year-Week'] <= final_week)
    df_filtered = df.loc[mask].copy()
    
    # Group by week and sum ALL columns
    df_weekly = df_filtered.groupby('Year-Week').sum(numeric_only=True).reset_index()
    unique_dates = df_weekly['Year-Week'].tolist()

    # --- 5. Aggregation and Return ---
    
    if aggregation == 'total':
        # Sum all numeric columns (excluding Year-Week)
        total_data = df_weekly.iloc[:, 1:].sum(axis=1).tolist()
        return unique_dates, total_data

    elif aggregation == 'gender':
        # Identify columns for Male (0) and Female (1)
        male_cols = [c for c in df.columns if c.endswith('_0')]
        female_cols = [c for c in df.columns if c.endswith('_1')]
        
        male_data = df_weekly[male_cols].sum(axis=1).tolist()
        female_data = df_weekly[female_cols].sum(axis=1).tolist()
        
        return unique_dates, male_data, female_data

    elif aggregation == 'age':
        age_dict = {}
        for age in age_groups:
            # Sum columns starting with this age group (both genders)
            relevant_cols = [f"{age}_0", f"{age}_1"]
            age_dict[age] = df_weekly[relevant_cols].sum(axis=1).tolist()
            
        return unique_dates, age_dict

    elif aggregation == 'age_gender':
        male_dict = {}
        female_dict = {}
        
        for age in age_groups:
            male_dict[age] = df_weekly[f"{age}_0"].tolist()
            female_dict[age] = df_weekly[f"{age}_1"].tolist()
            
        return unique_dates, male_dict, female_dict

    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation}")



def calibrate_model(model_type, Nij, CM_dates_dict, initial_date, t_max, t_step, initial_cases, mu, epsilon, IFR, Delta, real_deaths, threshold, n_sim, save_interval, 
                    prior_dict = None, output_file="calibration_results.csv"):
    """
    Calibrates the model by running random simulations and checking error against real deaths. Adapts logic based on 'model_type' (e.g. '0', 'IB', 'CIB').
    :param model_type (str): Model version code ('0', 'B', 'CI', etc.)
    :param Nij (array): Population matrix.
    :param CM_dates_dict (dict): Dictionary of contact matrices by date.
    :param initial_date (datetime): Start date of simulation.
    :param t_max (int): Simulation duration in days.
    :param t_step (int): Time step in days.
    :param initial_cases (float): Estimate of initial cases.
    :param mu (float): Recovery rate.
    :param epsilon (float): Incubation rate.
    :param IFR (array/matrix): Infection Fatality Rate (vector or matrix depending on model_type).
    :param Delta (int): Delay in observed deaths
    :param real_deaths (list): Observed weekly deaths to calibrate against.
    :param threshold (float): WMAPE error threshold for acceptance.
    :param n_sim (int): Number of simulations to run.
    :param save_interval (int): How often to save results to disk.
    :param priors (dict): Dictionary of callable functions to generate parameters. Keys expected: 'i0', 'r0', 'R01', 'R02'. Optional key: 'r_beta' (if model type has 'B').
    :param output_file (str): Path to output CSV.
    :return pd.DataFrame: containing all accepted parameter sets and simulated deaths.
    """

    # --- 1. Default Priors (Fallback if they are not provided) ---
    if prior_dict is None:
        prior_dict = {}
    
    # Define defaults using lambda functions
    prior_defaults = {
        'i0':  lambda: np.random.randint(int(initial_cases), int(10 * initial_cases)) / Nij.sum(),
        'r0':  lambda: np.random.uniform(0.03, 0.10),
        'R01': lambda: np.random.uniform(1.0, 1.7),
        'R02': lambda: np.random.uniform(0.5, 1.7),
        'r_beta': lambda: np.random.uniform(1.0, 1.3)
    }
    # Update defaults with any user-provided priors. For example, if user provides 'R01', it overwrites the default; others stay default.
    prior_dict = {**prior_defaults, **prior_dict}


    # --- 2. Setup Logic Flags ---
    use_gender_beta = 'B' in model_type  # If 'B', we optimize r_beta

    # --- 3. Prepare Output File ---
    # Extract directory and ensure it exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define Header
    header = "R01,R02,i0,r0,r_beta,err,simulated_deaths\n"
    columns = ["R01", "R02", "i0", "r0", "r_beta", "err", "simulated_deaths"]

    # Write Header
    with open(output_file, "w") as f:
        f.write(header)

    # --- 4. Prepare Data for Loops ---
    # Find cutoff date CM (Nov 6 2020) for beta2 calculation
    min_key = min((key for key in CM_dates_dict if key >= datetime(2020, 11, 6)), default=None)
    CM_sixnov = CM_dates_dict[min_key] if min_key else list(CM_dates_dict.values())[0]
    
    CM_start = list(CM_dates_dict.values())[0]

    params = []
    counter_accepted = 0

    # --- 5. Calibration Loop ---
    for k in range(n_sim):
        if k % 1000 == 0:
            print(f"Running simulation {k}...")

        # -- Generate Random Parameters using the Dictionary of priors --
        R01 = prior_dict['R01']()
        R02 = prior_dict['R02']()
        i0 = prior_dict['i0']()
        r0 = prior_dict['r0']()

        if use_gender_beta:
            r_beta = prior_dict['r_beta']()
        else:
            r_beta = 1.0
         
        beta1 = get_beta(model_type, R01, mu, CM_start, r_beta=r_beta)
        beta2 = get_beta(model_type, R02, mu, CM_sixnov, r_beta=r_beta)


        # -- Run Simulation --
        y, dates = run_model(model_type, Nij, CM_dates_dict, initial_date, t_max, t_step, beta1, beta2, i0, r0, epsilon, mu, IFR, Delta, r_beta=r_beta)

        # -- Compute Weekly Deaths --
        _, simulated_deaths = get_weekly_new_deaths(y, initial_date, t_max, t_step, aggregation='total', n_comp_deaths=5)

        # -- Calculate Error --
        err = wmape(np.array(real_deaths), np.array(simulated_deaths))

        # -- Acceptance Check --
        if err <= threshold:
            counter_accepted += 1

        # -- Store Results --
        # Serialize deaths list to JSON string so it fits in one CSV cell
        sim_deaths_json = json.dumps(simulated_deaths)
        
        params.append([R01, R02, i0, r0, r_beta, err, sim_deaths_json])

        # -- Save to Disk periodically --
        if (k + 1) % save_interval == 0:
            print(f"Saving results at iteration {k + 1}...")
            df = pd.DataFrame(params, columns=columns)
            df.to_csv(output_file, mode="a", header=False, index=False)
            params = []  # Clear buffer

        # Print progress
        if k % 100 == 0:
            print(f"Iteration {k}, Accepted samples = {counter_accepted}")

    # --- 6. Final Save ---
    if params:
        print("Saving remaining results...")
        df = pd.DataFrame(params, columns=columns)
        df.to_csv(output_file, mode="a", header=False, index=False)

    return pd.read_csv(output_file)



def run_posterior_sampling(model_type, Nij, CM_dates_dict, initial_date, t_max, t_step, mu, epsilon, IFR, Delta, df_best_simulations, n_param_sets, n_sim_per_set,
                           output_file=None):
    """
    Samples parameter sets from the best simulations and re-runs the stochastic model 
    multiple times for each set to capture posterior uncertainty.
    
    :param model_type (str): Model version code.
    :param Nij (array): Population matrix.
    :param CM_dates_dict (dict): Dictionary of contact matrices.
    :param initial_date (datetime): Start date.
    :param t_max (int): Duration in steps.
    :param t_step (int): Time step size.
    :param mu (float): Recovery rate.
    :param epsilon (float): Incubation rate.
    :param IFR (array/matrix): Infection Fatality Rate (vector or matrix depending on model_type).
    :param Delta (float): Delay parameter (used for simulation but not saved).
    :param df_best_simulations (pd.DataFrame): Dataframe of accepted calibration runs.
    :param n_param_sets (int): Number of unique parameter sets to sample.
    :param n_sim_per_set (int): Number of stochastic runs per parameter set.
    :param output_file (str): Path to save the output pickle file (e.g. .pkl or .pkl.gz).
    :return: pd.DataFrame containing parameters and the heavy 'y' simulation arrays.
    """
    
    # --- 1. Setup Matrices ---
    min_key = min((key for key in CM_dates_dict if key >= datetime(2020, 11, 6)), default=None)
    CM_sixnov = CM_dates_dict[min_key] if min_key else list(CM_dates_dict.values())[0]
    CM_start = list(CM_dates_dict.values())[0]

    # --- 2. Sampling ---
    # Sample unique parameter combinations from the accepted pool
    safe_n = min(len(df_best_simulations), n_param_sets)
    df_sampling = df_best_simulations.sample(n=safe_n, replace=False, random_state=44).reset_index(drop=True)
    
    sampled_results = []
    
    print(f"Starting {n_sim_per_set} x {safe_n} simulations for model {model_type}...")

    # --- 3. Simulation Loop ---
    for i, row in df_sampling.iterrows():
        
        if i % 10 == 0:
            print(f"Processing parameter set {i}/{safe_n}")

        # Extract parameters from the calibration dataframe
        i0 = row["i0"]
        R01 = row["R01"]
        R02 = row["R02"]
        r0 = row["r0"]
        r_beta = row["r_beta"]
        
        # Calculate Betas
        beta1 = get_beta(model_type, R01, mu, CM_start, r_beta)
        beta2 = get_beta(model_type, R02, mu, CM_sixnov, r_beta)

        # Run multiple stochastic realizations for this fixed parameter set
        for sim in range(n_sim_per_set):
            # Run the model (Delta is passed here for the logic, but not saved in output)
            y, dates = run_model(model_type, Nij, CM_dates_dict, initial_date, t_max, t_step, beta1, beta2, i0, r0, epsilon, mu, IFR, Delta, r_beta=r_beta)
            
            sampled_results.append({
                "R01": R01,
                "R02": R02,
                "i0": i0,
                "r0": r0,
                "r_beta": r_beta,
                "y": y 
            })

    # --- 4. Create DataFrame and Save ---
    df_results = pd.DataFrame(sampled_results)

    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Save to pickle (compression inferred from .gz extension)
        df_results.to_pickle(output_file)
        print(f"Saved {len(df_results)} simulations to {output_file}")

    return pd.DataFrame(sampled_results)