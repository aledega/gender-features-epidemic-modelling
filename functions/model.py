# Libraries
import numpy as np
from datetime import datetime, timedelta

# Global variables
n_age = 5
n_gen = 2
n_comp = 6



def get_R0(model_type, beta, mu, CM, r_beta=1.0):
    """
    Computes R0 for the compartmental model. The calculation method adapts based on the model version specified in 'model_type'.

    :param model_type (str): String code for the model version (e.g., '0', 'IB', 'CIB').
                             - 'C': Expects gender-stratified CM (10x10). If missing, expands age-only CM (5x5).
                             - 'B': Applies r_beta scalar to males. If missing, r_beta is effectively 1.0.
                             - 'I': Indicates IFR stratification (does not affect R0 calculation).
    :param beta (float): Attack rate (for females, or for everyone if model has no 'B').
    :param mu (float): Recovery rate.
    :param CM (np.ndarray): Contact matrix. Dimensions depend on 'model_type'.
    :param r_beta (float, optional): Multiplier for male susceptibility. Defaults to 1.0.
    :return: The reproduction number R0.
    """
    
    # --- 1. Handle Feature 'B' (Behaviors) ---
    # If 'B' is present, use r_beta. Otherwise, behaviors are uniform (multiplier = 1.0)
    current_r_beta = r_beta if 'B' in model_type else 1.0
    
    beta_f = beta
    beta_m = current_r_beta * beta

    # Build beta vector (length n_age * n_gen)
    # Males at even indices (0, 2, 4...), Females at odd indices (1, 3, 5...)
    beta_vec = np.zeros(n_age * n_gen)
    beta_vec[::2] = beta_m
    beta_vec[1::2] = beta_f

    # --- 2. Handle Feature 'C' (Contacts) ---
    # If 'C' is present, CM is already 10x10. If not, it's 5x5 and needs expansion.
    if 'C' in model_type:
        CM_final = CM
    else:
        # Kronecker product to expand age-matrix to age-gender matrix
        CM_final = np.kron(CM, np.ones((n_gen, n_gen)))

    # --- 3. Compute R0 ---
    # The Next Generation Matrix is CM_final multiplied by the diagonal beta vector
    # We take the maximum real eigenvalue
    next_gen_matrix = CM_final @ np.diag(beta_vec)
    eigenvalues = np.linalg.eigvals(next_gen_matrix)
    
    return 1 / mu * np.max(eigenvalues.real)




def get_beta(model_type, R0, mu, CM, r_beta=1.0):
    """
    This function computes the attack rate beta for the SIR model given a target R0.
    The calculation adapts based on the model version specified in 'model_type'.
        :param model_type (str): String code for the model version (e.g., '0', 'IB', 'CIB').
        :param R0 (float): target reproduction number
        :param mu (float): recovery rate
        :param CM (matrix): contacts matrix (dimensions depend on 'model_type')
        :param r_beta (float): relative susceptibility for males (only used if 'B' is in model_type). Defaults to 1.0.
        :return: returns the attack rate beta
    """
    
    # --- 1. Handle Feature 'B' (Behaviors) ---
    # If 'B' is in the string, use the provided r_beta. Otherwise, default to 1.0.
    current_r_beta = r_beta if 'B' in model_type else 1.0

    # Build the relative susceptibility vector (eta)
    # We initialize everything to 1.0 (females), then set even indices (males) to current_r_beta
    eta_vec = np.ones(n_age * n_gen)
    eta_vec[::2] = current_r_beta

    # --- 2. Handle Feature 'C' (Contacts) ---
    # If 'C' is present, CM is already 10x10. If not, expand the 5x5 age matrix.
    if 'C' in model_type:
        CM_final = CM
    else:
        CM_final = np.kron(CM, np.ones((n_gen, n_gen)))

    # --- 3. Compute Beta ---
    # The spectral radius (max eigenvalue) represents the 'amplification' factor of the contact structure
    # Formula: R0 = (beta / mu) * max_eigenvalue  =>  beta = (R0 * mu) / max_eigenvalue
    max_eigenval = np.max([eigen.real for eigen in np.linalg.eig(CM_final @ np.diag(eta_vec))[0]])

    return R0 * mu / max_eigenval




def initialize_model(Nij, i0, r0, epsilon, mu):
    """
    This function initializes the compartmental state vector y0 at t=0.
    It splits the initial infected fraction (i0) into Exposed and Infectious compartments proportional to the duration of those stages.
        :param Nij (array): Population matrix (n_age x n_gen)
        :param i0 (float): Initial fraction of the population that is infected (sum of Exposed + Infectious)
        :param r0 (float): Initial fraction of the population that is recovered
        :param epsilon (float): Inverse of the incubation/latent period (rate E -> I)
        :param mu (float): Recovery rate (rate I -> R)
        :return: returns y0, the initialized state vector of shape (n_age, n_gen, n_comp)
    """

    # initialise array of initial conditions
    y0 = np.zeros((n_age, n_gen, n_comp))

    # Calculate average durations for the Exposed (1/epsilon) and Infectious (1/mu) periods
    # These are used to distribute the initial i0 proportionally
    dur_E = epsilon**(-1)
    dur_I = mu**(-1)
    total_dur = dur_E + dur_I

    for age in range(n_age):
        for gen in range(n_gen):
            
            # Compartment 1 (Exposed/Latent): Proportion of i0 based on duration of E stage
            y0[age, gen, 1] = (dur_E / total_dur) * i0 * Nij[age, gen]
            
            # Compartment 2 (Infectious): Proportion of i0 based on duration of I stage
            y0[age, gen, 2] = (dur_I / total_dur) * i0 * Nij[age, gen]
            
            # Compartment 3 (Recovered): Direct fraction r0
            y0[age, gen, 3] = r0 * Nij[age, gen]
            
            # Compartment 0 (Susceptible): The remainder of the population
            # S = N - E - I - R
            y0[age, gen, 0] = Nij[age, gen] - y0[age, gen, 1] - y0[age, gen, 2] - y0[age, gen, 3]

    return y0




def compute_ifr_bygender(IFR, OR, Nij):
    """
    This function computes IFR stratified by gender given total IFR, an Odds Ratio (OR), and population distribution.
    It solves the system of equations derived from the definition of OR and the weighted average of IFR.
        :param IFR (array): Vector of total IFR for each age group (length n_age)
        :param OR (float): Odds ratio of mortality (Male vs Female). We use the value from the literature 1.39
        :param Nij (array): Matrix with population distribution [Males, Females] per age group
        :return: returns a matrix (n_age x 2) with [IFR_Male, IFR_Female] for each age group
    """
    IFR = np.array(IFR)
    Nij = np.array(Nij)
    
    # Calculate fractions of Males and Females per age group
    # f_M + f_F = 1 for each age row
    f_M = Nij[:, 0] / Nij.sum(axis=1)
    f_F = Nij[:, 1] / Nij.sum(axis=1)
    
    # Coefficients for the quadratic equation a*x^2 + b*x + c = 0 where x is IFR_Female. 
    # These are derived from substituting the OR formula into the total IFR formula.
    a = f_F * (OR - 1)
    b = f_M * OR + f_F - IFR * (OR - 1)
    c = -IFR
    
    # Solve for IFR_Female (using the positive root of the quadratic formula)
    IFR_F = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    # Calculate IFR_Male using the Odds Ratio relationship
    # Odds_Male = OR * Odds_Female  =>  p_m/(1-p_m) = OR * p_f/(1-p_f)
    IFR_M = (OR * IFR_F) / (1 + (OR - 1) * IFR_F)
    
    # Combine into a single matrix [Males, Females]
    return np.column_stack((IFR_M, IFR_F))



def stochastic_SLIRDDO(model_type, y, Nij, CM, t_step, beta, epsilon, mu, IFR, Delta, r_beta=1.0):
    """
    Simulates one time-step of the stochastic SLIRDDO model. The logic adapts based on the structure specified in 'model_type'.
    :param model_type (str): String code for the model version (e.g., '0', 'CI', 'B').
                             - 'C': Uses gender-stratified CM (10x10 indexing). Else 5x5 indexing.
                             - 'I': Uses gender-stratified IFR (matrix). Else age-only IFR (vector).
                             - 'B': Applies r_beta scalar to males. Else beta is uniform.
    :param y (array): State vector of shape (n_age, n_gen, n_comp).
    :param Nij (array): Population matrix (n_age x n_gen).
    :param CM (matrix): Contact matrix (dimensions depend on 'model_type').
    :param t_step (float): Time step duration.
    :param beta (float): Base transmission rate.
    :param epsilon (float): Inverse of incubation period.
    :param mu (float): Recovery rate.
    :param IFR (array/matrix): Infection Fatality Rate (dimensions depend on 'model_type').
    :param Delta (float): Duration of immunity/temporary recovery state D before D0.
    :param r_beta (float, optional): Multiplier for male susceptibility (used if 'B' in model_type). Defaults to 1.0.
    :return: Updated state vector y.
    """

    def transform_rate_to_probability(rate, delta_t):
        """Convert a rate to a probability given time step delta_t."""
        return 1 - np.exp(-rate * delta_t)

    new_y = y.copy()
    
    # Pre-calculate flags to avoid string checking inside the loop (optimization)
    use_stratified_contacts = 'C' in model_type
    use_stratified_ifr = 'I' in model_type
    use_gender_beta = 'B' in model_type

    for age1 in range(n_age):
        for gen1 in range(n_gen):

            delta_S_to_L = 0
            delta_L_to_I = 0
            delta_I_to_R = 0
            delta_I_to_D = 0
            delta_D_to_D0 = 0
            
            # --- 1. Compute Force of Infection (S -> L) ---
            S_to_L = 0

            for age2 in range(n_age):
                for gen2 in range(n_gen):
                    
                    # Logic C: If 'C' is present, index CM as (n_gen*age + gen). Otherwise, index CM as (age).
                    if use_stratified_contacts:
                        cm_val = CM[n_gen * age1 + gen1, n_gen * age2 + gen2]
                    else:
                        cm_val = CM[age1, age2]

                    # Add contribution to force of infection
                    S_to_L += cm_val * (y[age2, gen2, 2]) / Nij[age2, gen2]

            # Logic B: If 'B' is present and gender is Male (0), scale beta by r_beta. Otherwise, use standard beta.
            if use_gender_beta and gen1 == 0:
                beta_eff = beta * r_beta
            else:
                beta_eff = beta
            
            risk_S_to_L = transform_rate_to_probability(beta_eff * S_to_L, t_step)
            risk_L_to_I = transform_rate_to_probability(epsilon, t_step)

            # --- 2. Compute Outcomes (I -> R vs I -> D) ---
            # Logic I: If 'I' is present, IFR is a matrix [age, gen]. Otherwise, IFR is a vector [age].
            if use_stratified_ifr:
                current_ifr = IFR[age1, gen1]
            else:
                current_ifr = IFR[age1]

            risk_I_to_R = transform_rate_to_probability(mu * (1 - current_ifr), t_step)
            risk_I_to_D = transform_rate_to_probability(mu * current_ifr, t_step)
            
            # Probability of staying in I (complement of leaving)
            risk_I_stay = 1 - risk_I_to_R - risk_I_to_D
            risk_I = [risk_I_to_R, risk_I_to_D, risk_I_stay]

            risk_D_to_D0 = transform_rate_to_probability(1/Delta, t_step)

            # --- 3. Apply Stochastic Transitions ---
            if risk_S_to_L > 0:
                delta_S_to_L = np.random.binomial(y[age1, gen1, 0], risk_S_to_L)

            if y[age1, gen1, 1] != 0:
                delta_L_to_I = np.random.binomial(y[age1, gen1, 1], risk_L_to_I)

            if y[age1, gen1, 2] != 0:
                # Multinomial sample: returns shape (1, 3), we need [0][0] and [0][1]
                delta_I = np.random.multinomial(y[age1, gen1, 2], risk_I, size=1)
                delta_I_to_R = delta_I[0][0]
                delta_I_to_D = delta_I[0][1]

            if y[age1, gen1, 4] != 0:
                delta_D_to_D0 = np.random.binomial(y[age1, gen1, 4], risk_D_to_D0)

            # --- 4. Update Compartments ---
            
            # S (Susceptible)
            new_y[age1, gen1, 0] -= delta_S_to_L

            # L (Latent/Exposed)
            new_y[age1, gen1, 1] += delta_S_to_L
            new_y[age1, gen1, 1] -= delta_L_to_I

            # I (Infectious)
            new_y[age1, gen1, 2] += delta_L_to_I
            new_y[age1, gen1, 2] -= delta_I_to_R
            new_y[age1, gen1, 2] -= delta_I_to_D

            # R (Recovered)
            new_y[age1, gen1, 3] += delta_I_to_R

            # D (Dead)
            new_y[age1, gen1, 4] += delta_I_to_D
            new_y[age1, gen1, 4] -= delta_D_to_D0

            # DO (Dead Observed)
            new_y[age1, gen1, 5] += delta_D_to_D0

    return new_y





def run_model(model_type, Nij, CM_set, initial_date, t_max, t_step, beta1, beta2, i0, r0, epsilon, mu, IFR, Delta, r_beta=1.0):
    """
    Integrates the SLIRDDO system step-by-step over the simulation period. The function adapts based on the 'model_type' (e.g., '0', 'CI', 'B').
    :param model_type (str): String code for the model version.
    :param Nij (array): Population matrix (n_age x n_gen).
    :param CM_set (dict): Dictionary mapping dates to contact matrices.
    :param initial_date (datetime): Start date of the simulation.
    :param t_max (int): Total number of time steps.
    :param t_step (float, optional): Time step size.
    :param beta1 (float): Attack rate before the cutoff date (Nov 6, 2020).
    :param beta2 (float): Attack rate after the cutoff date.
    :param i0 (float): Initial infected fraction.
    :param r0 (float): Initial recovered fraction.
    :param epsilon (float): Inverse of incubation period.
    :param mu (float): Recovery rate.
    :param IFR (array/matrix): Infection Fatality Rate.
    :param Delta (float): Duration of temporary removed state D.
    :param r_beta (float, optional): Relative susceptibility for males (used if 'B' in model_type). Defaults to 1.0.
    :return: Tuple (solution array, list of dates).
    """

    # create solution array with shape: (Age Groups, Genders, Compartments, Time Steps)
    sol = np.zeros((n_age, n_gen, n_comp, t_max))

    # set initial conditions
    y0 = initialize_model(Nij, i0, r0, epsilon, mu)
    sol[:, :, :, 0] = y0
    
    t = 0                         # current time accumulator
    dates = [initial_date]        # list of dates corresponding to steps
    
    # Cutoff date for switching betas (Italy Tier system introduction)
    cutoff_date = datetime(2020, 11, 6)

    # integrate forward in time
    for i in np.arange(1, t_max, 1):
        
        # Update current date
        current_date = dates[-1] + timedelta(days=t_step)
        dates.append(current_date)

        # Retrieve the contact matrix for the current date
        # (Assumes CM_set has an entry for every date in the simulation)
        CM = CM_set[current_date]

        # Select the appropriate beta based on the date
        if current_date < cutoff_date:
            beta = beta1
        else:
            beta = beta2

        # Run one stochastic step. We pass 'model_type' and 'r_beta' so the inner function knows how to behave
        sol[:, :, :, i] = stochastic_SLIRDDO(
            model_type, 
            sol[:, :, :, i-1], 
            Nij, 
            CM, 
            t_step,
            beta, 
            epsilon, 
            mu, 
            IFR, 
            Delta,  
            r_beta=r_beta
        )

        # advance time accumulator
        t += t_step
        
    return sol, dates