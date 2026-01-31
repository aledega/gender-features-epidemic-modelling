import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy.linalg import eigvals



def update_matrices_polymod(date, df_reductions, CM_polymod):
    """
    This function computes the contact matrix for a specific date using Google Mobility reductions applied to the baseline POLYMOD matrices
        :param date (datetime): current simulation date
        :param df_reductions (DataFrame): dataframe containing weekly reduction parameters (indices must be 'YYYY-WW')
        :param CM_polymod (dict): dictionary containing baseline POLYMOD matrices (keys: 'home', 'work', 'leisure', 'other', 'school')
        :return: returns the contact matrix adjusted by mobility reductions for the given date
    """
    
    # extract week from date
    week = date.strftime("%Y-%W")
    
    # compute contacts' reductions parameter for each setting (Assuming reductions are stored as percentages, e.g., 50 for 50%)
    omega_work = (1 + df_reductions.loc[week, 'work_red']/100)**2
    omega_leisure = (1 + df_reductions.loc[week, 'leisure_red']/100)**2
    omega_other = (1 + df_reductions.loc[week, 'other_red']/100)**2
    omega_school = (3 - df_reductions.loc[week, 'school_red']) / 3

    # compute contact matrix by correcting the contacts in each setting by their corresponding parameter and summing them
    contact_matrix = (CM_polymod['home'] + 
                      omega_work * CM_polymod['work'] + 
                      omega_leisure * CM_polymod['leisure'] + 
                      omega_other * CM_polymod['other'] + 
                      omega_school * CM_polymod['school'])

    return contact_matrix


def update_matrices_comix(date, df_zones_byregion, df_population_byregion, CM_comix):
    """
    This function computes the contact matrix for a specific date by weighting CoMix matrices based on the population distribution across color zones
        :param date (datetime): current simulation date
        :param df_zones_byregion (DataFrame): dataframe where columns are dates and rows are regions, containing zone codes (1=Yellow, 2=Orange, 3=Red)
        :param df_population_byregion (DataFrame): dataframe containing population size per region (must have 'Population' column)
        :param CM_comix (dict): dictionary containing CoMix contact matrices for each zone (keys: 'Yellow', 'Orange', 'Red')
        :return: returns the weighted average contact matrix for the given date
    """

    # extract day from date
    date_str = date.strftime('%Y-%m-%d')

    # Get the zone classification for all regions on that day
    zone_data = df_zones_byregion[date_str]

    # compute populations in each zone
    pop_yellow = df_population_byregion.loc[zone_data[zone_data == 1].index, "Population"].sum()
    pop_orange = df_population_byregion.loc[zone_data[zone_data == 2].index, "Population"].sum()
    pop_red = df_population_byregion.loc[zone_data[zone_data == 3].index, "Population"].sum()
    pop_tot = df_population_byregion["Population"].sum()

    # compute the contact matrix for the current date weighting the population in each zone
    contact_matrix = (
        (pop_yellow / pop_tot) * CM_comix['Yellow'] +
        (pop_orange / pop_tot) * CM_comix['Orange'] +
        (pop_red / pop_tot) * CM_comix['Red']
    )

    return contact_matrix


def create_matrices_dict(initial_date, t_max, t_step, df_reductions, df_zones_byregion, df_population_byregion, CM_polymod, CM_comix):
    """
    This function generates a dictionary of contact matrices for the entire simulation period, switching methodology based on the date
        :param initial_date (datetime): start date of the simulation
        :param t_max (int): total duration of simulation in days
        :param t_step (int): time step in days
        :param df_reductions (DataFrame): dataframe with weekly reduction parameters
        :param df_zones_byregion (DataFrame): dataframe with zone classification per region
        :param df_population_byregion (DataFrame): dataframe with population size per region
        :param CM_polymod (dict): dictionary containing baseline POLYMOD contact matrices
        :param CM_comix (dict): dictionary containing CoMix contact matrices for each zone
        :return: returns a dictionary with dates as keys and contact matrices as values
    """

    # initialize the list of dates with the first one
    dates = [initial_date]  

    # add every date to the list      
    for i in np.arange(1, t_max, 1):
        dates.append(dates[-1] + timedelta(days=t_step))
    
    # initialize an empty dictionary 
    CM_dates_dict = {}

    # Cutoff date for switching methodologies (Italy Tier system introduction)
    cutoff_date = datetime(2020, 11, 6)

    # for each date add the corresponding contact matrix
    for date in dates:

        # if the date is before the enforcement of the tier-list measures use POLYMOD data adjusted by mobility
        if date < cutoff_date:
            CM_dates_dict[date] = update_matrices_polymod(date, df_reductions, CM_polymod)

        # if the date is after the enforcement of the tier-list measures use CoMix data considering the population in each zone
        else:
            CM_dates_dict[date] = update_matrices_comix(date, df_zones_byregion, df_population_byregion, CM_comix)

    return CM_dates_dict




def compute_spectral_radii(matrices_dict):
    """
    Computes the spectral radius (dominant real eigenvalue) for lists of matrices stored in a dictionary.
    :param matrices_dict (dict): A dictionary where keys are identifiers (e.g., 'Male-Male') and values are lists of numpy arrays (matrices).
    :return pd.DataFrame: A DataFrame where each column corresponds to a key in the input dictionary and contains the spectral radii of the matrices in that list.
    """
    # Use a dictionary comprehension to build the data structure in one step
    # Key -> Column Name, Value -> List of eigenvalues
    data = {
        key: [np.max(eigvals(m).real) for m in matrix_list]
        for key, matrix_list in matrices_dict.items()
    }
    
    return pd.DataFrame(data)