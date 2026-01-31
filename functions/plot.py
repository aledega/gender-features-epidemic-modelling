# Libraries and functions
from calibration import rmse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import datetime

# Global variables 
age_groups = ['20-29', '30-39', '40-49', '50-59', '60+']                                   # list of strings with age groups 
n_age = len(age_groups)                                                                    # number of age groups
model_list = ['0', 'I', 'B', 'IB', 'C', 'CI', 'CB', 'CIB']                                 # list of strings with model codes
n_models = len(model_list)                                                                 # number of models



def plot_posterior_distributions(best_simulations_dict, model_list, Nij, initial_cases, save_path=None):
    """
    Plots the posterior distributions of calibrated parameters using the best simulations dictionary.
    
    :param best_simulations_dict: Dictionary {model_string: pd.DataFrame} of filtered results.
    :param model_list: List of model strings to enforce ordering.
    :param Nij: Population array (used for i0 scaling).
    :param initial_cases: Reference initial cases (used for i0 scaling).
    :param save_path: Full path to save the figure (e.g. '../outputs/Posterior.png'). If None, does not save.
    """

    # --- 1. Reshape Data for Plotting ---
    data_frames = []

    for model_string, df in best_simulations_dict.items():
        # Select columns to plot
        cols = ['R01', 'R02', 'i0', 'r0']
        
        # Only include r_beta if the model actually uses behavior ('B')
        if 'B' in model_string:
            cols.append('r_beta')
            
        # Subset and tag
        subset = df[cols].copy()
        subset['model'] = model_string
        data_frames.append(subset)

    # Combine
    full_df = pd.concat(data_frames, ignore_index=True)

    # Melt to long format (replacing the manual loop)
    plot_df = full_df.melt(id_vars='model', var_name='parameter', value_name='value')
    
    # Drop NaNs (which naturally happen because non-B models don't have r_beta)
    plot_df = plot_df.dropna().reset_index(drop=True)

    # --- 2. Plotting Configuration ---
    title_map = {
        'R01': r"$R_{0_1}$ (September 14, 2020)",
        'R02': r"$R_{0_2}$ (November 06, 2020)",
        'i0': r"Initial number of infected individuals $i_0$",
        'r0': r"Initial percentage of recovered individuals $r_0$",
        'r_beta': r"Increased transmission rate for males $r_{\beta}$",
    }

    palette = sns.color_palette("Set2", n_colors=len(model_list))
    model_palette = dict(zip(model_list, palette))

    # Layout configuration
    layout = {
        'R01': (0, slice(0, 2)), 
        'R02': (0, slice(2, 4)), 
        'i0':  (0, slice(4, 6)), 
        'r0': (1, slice(1, 3)), 
        'r_beta': (1, slice(3, 5))
    }

    # --- 3. Generate Plot ---
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)

    axes_by_row = {0: [], 1: []}

    for param, (row, col_slice) in layout.items():
        ax = fig.add_subplot(gs[row, col_slice])
        
        # Isolate specific parameter data
        subset = plot_df[plot_df["parameter"] == param].copy()

        # --- Specific Formatting Logic ---
        
        # i0 Scaling: Scale relative to initial_cases
        if param == 'i0':
            subset['value'] = subset['value'] * Nij.sum() / initial_cases
            
            # Create custom ticks at odd multiples (1x, 3x, 5x...)
            odd_multiples = [m for m in range(1, 11, 2)]
            ax.set_xticks(odd_multiples)
            ax.set_xticklabels([f"{m} $i_{{14\\,Sept}}$" if m > 1 else r"$i_{14\,Sept}$" for m in odd_multiples])
            ax.set_xlim(0, 11) # Keep visual range clean

        # r0 Formatting: Percentages
        elif param == 'r0':
            # Use FuncFormatter for robust % formatting
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        # --- Violin Plot ---
        sns.violinplot(
            data=subset, y="model", x="value", 
            order=model_list, hue="model", palette=model_palette, 
            density_norm="width", inner="quartile", linewidth=1, 
            ax=ax, legend=False
        )

        # Titles
        ax.set_title(title_map[param])
        ax.set_xlabel("Posterior value")

        # Y-Axis Logic: Show model names only on the leftmost plot of each row
        axes_by_row[row].append(ax)
        if len(axes_by_row[row]) == 1:
            ax.set_ylabel("Model")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # Aesthetics
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis="both")

    # --- 4. Save and Show ---
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()




def plot_deaths_aggregated(simulations_dict, observed_deaths, weeks_list, output_path=None):
    """
    Plots the aggregated weekly deaths for each model against observed data, annotating the RMSE on each subplot.
    :param simulations_dict (dict): Dictionary where keys are model strings and values are lists of weekly death arrays (n_sim x n_weeks).
    :param observed_deaths (array-like): Array of real observed weekly deaths matching 'weeks_list'.
    :param weeks_list (list): List of week strings (e.g., '2020-45') corresponding to the x-axis.
    :param output_path (str, optional): Path to save the figure.
    """

    # --- Helper: Date Formatting for Ticks ---
    def week_label(w): 
        # Converts "2020-45" -> "08 Nov" (approx) using the Monday of that week
        return datetime.strptime(f"{w}-1", "%Y-%W-%w").strftime("%d %b")

    # --- Setup Plotting Grid ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300, sharex=True, sharey=True)
    
    # Define Ticks (show ~4 ticks evenly spaced)
    n_ticks = 4
    tick_indices = np.linspace(2, len(weeks_list) - 3, n_ticks, dtype=int)
    tick_labels = [week_label(weeks_list[i]) for i in tick_indices]

    # Map model index to subplot (row, col)
    # Assumes exactly 8 models for a 2x4 grid
    model_to_position = {model: (i // 4, i % 4) for i, model in enumerate(model_list)}

    # Standard Font Sizes
    label_fontsize = 20
    title_fontsize = 22
    legend_fontsize = 20
    tick_fontsize = 16

    # --- Loop Over Models ---
    for model_key in model_list:
        if model_key not in simulations_dict:
            continue

        row_idx, col_idx = model_to_position[model_key]
        ax = axes[row_idx, col_idx]

        # 1. Prepare Simulation Stats
        sims = np.array(simulations_dict[model_key])
        mean_sim = np.mean(sims, axis=0)
        lower_ci = np.percentile(sims, 2.5, axis=0)
        upper_ci = np.percentile(sims, 97.5, axis=0)

        # 2. Calculate RMSE
        rmse_val = rmse(mean_sim, observed_deaths)

        # 3. Plot Simulations
        ax.plot(weeks_list, mean_sim, color='C0', label="Mean Predicted")
        ax.fill_between(weeks_list, lower_ci, upper_ci, color='C0', alpha=0.3)

        # 4. Plot Observed Data
        ax.scatter(weeks_list, observed_deaths, color='C0', label="Reported Total", zorder=5, s=20)

        # 5. Styling & Annotations
        ax.set_title(f"Model {model_key}", fontsize=title_fontsize)
        
        # Display RMSE (Simple text, no bolding/highlighting)
        ax.text(0.95, 0.95, f"RMSE: {rmse_val:.2f}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=18, color='C0')

        ax.tick_params(axis="both", which='major', labelsize=tick_fontsize)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Y-axis label (only left column)
        if col_idx == 0:
            ax.set_ylabel("Weekly Deaths", fontsize=label_fontsize)

        # X-axis ticks (only bottom row)
        if row_idx == 1:
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels)

    # --- Final Layout ---
    # Shared Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=legend_fontsize)

    fig.tight_layout(rect=[0.0, 0.05, 1, 1])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()



def plot_deaths_gender(simulations_male_dict, simulations_female_dict, observed_male, observed_female, weeks_list, output_path=None):
    """
    Plots the weekly deaths stratified by gender (Male=Blue, Female=Red) for each model, annotating the RMSE for both genders on each subplot.

    :param simulations_male_dict (dict): Dictionary where keys are model strings and values are lists of male weekly death arrays (n_sim x n_weeks).
    :param simulations_female_dict (dict): Dictionary where keys are model strings and values are lists of female weekly death arrays (n_sim x n_weeks).
    :param observed_male (array-like): Array of real observed male weekly deaths matching 'weeks_list'.
    :param observed_female (array-like): Array of real observed female weekly deaths matching 'weeks_list'.
    :param weeks_list (list): List of week strings (e.g., '2020-45') corresponding to the x-axis.
    :param output_path (str, optional): Path to save the figure.
    """
    
    # --- Helper: Date Formatting for Ticks ---
    def week_label(w): 
        return datetime.strptime(f"{w}-1", "%Y-%W-%w").strftime("%d %b")

    # --- Setup Plotting Grid ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300, sharex=True, sharey=True)
    
    # Define Ticks
    n_ticks = 4
    tick_indices = np.linspace(2, len(weeks_list) - 3, n_ticks, dtype=int)
    tick_labels = [week_label(weeks_list[i]) for i in tick_indices]

    # Map model to position
    model_to_position = {model: (i // 4, i % 4) for i, model in enumerate(model_list)}

    # Standard Font Sizes
    label_fontsize = 20
    title_fontsize = 22
    legend_fontsize = 20
    tick_fontsize = 16

    # --- Loop Over Models ---
    for model_key in model_list:
        if model_key not in simulations_male_dict:
            continue

        row_idx, col_idx = model_to_position[model_key]
        ax = axes[row_idx, col_idx]

        # 1. Prepare Simulation Stats (Male)
        sims_m = np.array(simulations_male_dict[model_key])
        mean_m = np.mean(sims_m, axis=0)
        lower_ci_m = np.percentile(sims_m, 2.5, axis=0)
        upper_ci_m = np.percentile(sims_m, 97.5, axis=0)

        # 2. Prepare Simulation Stats (Female)
        sims_f = np.array(simulations_female_dict[model_key])
        mean_f = np.mean(sims_f, axis=0)
        lower_ci_f = np.percentile(sims_f, 2.5, axis=0)
        upper_ci_f = np.percentile(sims_f, 97.5, axis=0)

        # 3. Calculate RMSE
        rmse_m = rmse(mean_m, observed_male)
        rmse_f = rmse(mean_f, observed_female)

        # 4. Plot Male (Blue)
        ax.plot(weeks_list, mean_m, color='blue', label="Mean Male")
        ax.fill_between(weeks_list, lower_ci_m, upper_ci_m, color='blue', alpha=0.3)
        ax.scatter(weeks_list, observed_male, color='blue', label="Reported Male", zorder=5, s=20)

        # 5. Plot Female (Red)
        ax.plot(weeks_list, mean_f, color='red', label="Mean Female")
        ax.fill_between(weeks_list, lower_ci_f, upper_ci_f, color='red', alpha=0.3)
        ax.scatter(weeks_list, observed_female, color='red', label="Reported Female", zorder=5, s=20)

        # 6. Styling & Annotations
        ax.set_title(f"Model {model_key}", fontsize=title_fontsize)
        
        # Annotation: Male RMSE
        ax.text(0.95, 0.95, f"RMSE M: {rmse_m:.2f}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=18, color='blue')

        # Annotation: Female RMSE (stacked below Male)
        ax.text(0.95, 0.85, f"RMSE F: {rmse_f:.2f}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=18, color='red')

        ax.tick_params(axis="both", which='major', labelsize=tick_fontsize)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col_idx == 0:
            ax.set_ylabel("Weekly Deaths", fontsize=label_fontsize)

        if row_idx == 1:
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels)

    # Adjust y-limits in order to leave enough space for the RMSE Annotations 
    # We only need to adjust the first ax because sharey=True links them all
    axes[0, 0].set_ylim(bottom=axes[0, 0].get_ylim()[0], top=axes[0, 0].get_ylim()[1] * 1.05)

    # --- Final Layout ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Add legend: 4 labels (Mean M, Reported M, Mean F, Reported F) displayed in 1 row.
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False, fontsize=legend_fontsize)

    fig.tight_layout(rect=[0.0, 0.05, 1, 1])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()


def plot_deaths_age(simulations_dict, observed_df, weeks_list, output_path=None):
    """
    Plots the weekly deaths stratified by age group (rows) and model (columns), annotating the RMSE on each subplot.
    
    :param simulations_dict (dict): Nested dict with weekly deaths by age group for each model {model_key: {age_group: [list of arrays]}}
    :param observed_df (pd.DataFrame): Dataframe of observed deaths with columns ['week', 'age_group', 'weekly_deaths']
    :param weeks_list (list): List of week strings (e.g., '2020-45') corresponding to the x-axis.
    :param output_path (str, optional): Path to save the figure.
    """

    # --- Helper: Date Formatting ---
    def week_label(w): 
        return datetime.strptime(f"{w}-1", "%Y-%W-%w").strftime("%d %b")

    # --- Setup Grid ---  
    # Create 5x8 grid (assuming 8 models)
    fig, axes = plt.subplots(n_age, n_models, figsize=(28, 20), sharex=True, sharey='row', dpi=300)

    # Ticks setup
    n_ticks = 4
    tick_indices = np.linspace(2, len(weeks_list)-3, n_ticks, dtype=int)
    tick_labels = [week_label(weeks_list[i]) for i in tick_indices]

    # Map model to column index
    # We sort keys to ensure consistent order, or rely on dictionary insertion order if using Python 3.7+
    model_keys = list(simulations_dict.keys())
    
    # --- Loop over Grid ---
    for col_idx, model_key in enumerate(model_keys):
        
        simulations_age_data = simulations_dict[model_key]

        for row_idx, age in enumerate(age_groups):
            ax = axes[row_idx, col_idx]

            # 1. Prepare Simulation Data
            if age not in simulations_age_data:
                continue
                
            sims = np.array(simulations_age_data[age])
            mean_sim = np.mean(sims, axis=0)
            lower_ci = np.percentile(sims, 2.5, axis=0)
            upper_ci = np.percentile(sims, 97.5, axis=0)

            # 2. Get Observed Data for this specific age group
            obs_data = observed_df[observed_df['age_group'] == age]
            # Ensure sorting matches weeks_list order if necessary, but assuming pre-sorted/filtered
            y_obs = obs_data['weekly_deaths'].values
            
            # 3. Calculate RMSE
            # Note: Ensure lengths match. If obs is shorter/longer, crop to min length
            n_points = min(len(mean_sim), len(y_obs))
            rmse_val = rmse(mean_sim[:n_points], y_obs[:n_points])

            # 4. Plot
            ax.plot(weeks_list, mean_sim, color='C0', label="Mean simulations")
            ax.fill_between(weeks_list, lower_ci, upper_ci, color='C0', alpha=0.3)
            
            # Plot Observed (Use the weeks from the dataframe to ensure alignment)
            ax.scatter(obs_data['week'].values, y_obs, color="C0", label="Reported", s=20, zorder=5)

            # 5. Styling & Annotation
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=18)


            # RMSE Annotation
            ax.text(0.95, 0.95, f"RMSE: {rmse_val:.2f}",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=20, color='C0')

            # Column Titles (Model Name) - Top Row Only
            if row_idx == 0:
                ax.set_title(f"Model {model_key}", fontsize=26, pad=10)

            # Row Labels (Age Group) - Left Column Only
            if col_idx == 0:
                ax.set_ylabel(f"{age}\nweekly deaths", fontsize=24)

            # X-Axis Labels - Bottom Row Only
            if row_idx == (n_age - 1):
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=18)

    # Adjust y-limits in order to leave enough space for the RMSE Annotations 
    for row_idx in range(n_age):
        # We only need to adjust the first axes in the row because sharey=True links them all
        ax_row = axes[row_idx, 0]
        
        # Get current autoscaled limits
        ymin, ymax = ax_row.get_ylim()
        
        # Increase upper limit by 10% or 15% to make room for text
        if row_idx >= n_age - 2:
            ax_row.set_ylim(bottom=ymin, top=ymax * 1.15)
        else:
            ax_row.set_ylim(bottom=ymin, top=ymax * 1.10)


    # --- Final Layout ---
    # Shared Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    fig.legend(unique_labels.values(), unique_labels.keys(), 
               loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=2, frameon=False, fontsize=28)

    # Align Y-labels for cleanliness
    fig.align_ylabels(axes[:, 0])

    fig.tight_layout(rect=[0.0, 0.05, 1, 1])
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()



def plot_deaths_age_gender(simulations_male_dict, simulations_female_dict, observed_df, weeks_list, output_path=None):
    """
    Plots the weekly deaths stratified by age group (rows), gender (color), and model (columns), annotating the RMSE for both genders on each subplot.

    :param simulations_male_dict (dict): Nested dict with male weekly deaths by age group for each model {model_key: {age_group: [list of arrays]}}
    :param simulations_female_dict (dict): Nested dict with female weekly deaths by age group for each model {model_key: {age_group: [list of arrays]}}
    :param observed_df (pd.DataFrame): Dataframe of observed deaths with columns ['week', 'gender', 'age_group', 'weekly_deaths']
    :param weeks_list (list): List of week strings (e.g., '2020-45') corresponding to the x-axis.
    :param output_path (str, optional): Path to save the figure.
    """

    # --- Helper: Date Formatting ---
    def week_label(w): 
        return datetime.strptime(f"{w}-1", "%Y-%W-%w").strftime("%d %b")

    # --- Setup Grid ---
    # Create 5x8 grid (assuming 8 models)
    fig, axes = plt.subplots(n_age, n_models, figsize=(28, 20), sharex=True, sharey='row', dpi=300)

    # Ticks setup
    n_ticks = 4
    tick_indices = np.linspace(2, len(weeks_list)-3, n_ticks, dtype=int)
    tick_labels = [week_label(weeks_list[i]) for i in tick_indices]

    model_keys = list(simulations_male_dict.keys())
    
    # --- Loop over Grid ---
    for col_idx, model_key in enumerate(model_keys):
        
        sim_data_male = simulations_male_dict[model_key]
        sim_data_female = simulations_female_dict[model_key]

        for row_idx, age in enumerate(age_groups):
            ax = axes[row_idx, col_idx]

            # --- 1. Male Data ---
            if age in sim_data_male:
                sims_m = np.array(sim_data_male[age])
                mean_m = np.mean(sims_m, axis=0)
                lower_ci_m = np.percentile(sims_m, 2.5, axis=0)
                upper_ci_m = np.percentile(sims_m, 97.5, axis=0)
                
                ax.plot(weeks_list, mean_m, color='blue', label="Mean Male")
                ax.fill_between(weeks_list, lower_ci_m, upper_ci_m, color='blue', alpha=0.3)
                
                # RMSE Male
                obs_male = observed_df[(observed_df['gender'] == 'M') & (observed_df['age_group'] == age)]['weekly_deaths'].values
                rmse_m = rmse(mean_m[:len(obs_male)], obs_male)
                
                ax.scatter(weeks_list, obs_male, color='blue', label="Reported Male", s=20, zorder=5)
                
                # Annotate Male
                ax.text(0.95, 0.95, f"RMSE M: {rmse_m:.1f}",
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=17, color='blue')

            # --- 2. Female Data ---
            if age in sim_data_female:
                sims_f = np.array(sim_data_female[age])
                mean_f = np.mean(sims_f, axis=0)
                lower_ci_f = np.percentile(sims_f, 2.5, axis=0)
                upper_ci_f = np.percentile(sims_f, 97.5, axis=0)
                
                ax.plot(weeks_list, mean_f, color='red', label="Mean Female")
                ax.fill_between(weeks_list, lower_ci_f, upper_ci_f, color='red', alpha=0.3)
                
                # RMSE Female
                obs_female = observed_df[(observed_df['gender'] == 'F') & (observed_df['age_group'] == age)]['weekly_deaths'].values
                rmse_f = rmse(mean_f[:len(obs_female)], obs_female)
                
                ax.scatter(weeks_list, obs_female, color='red', label="Reported Female", s=20, zorder=5)
                
                # Annotate Female (stacked below Male)
                ax.text(0.95, 0.85, f"RMSE F: {rmse_f:.1f}",
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=17, color='red')

            # --- 3. Styling ---
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=18)

            # Titles & Labels
            if row_idx == 0:
                ax.set_title(f"Model {model_key}", fontsize=26, pad=10)

            if col_idx == 0:
                ax.set_ylabel(f"{age}\nweekly deaths", fontsize=24)

            if row_idx == (n_age - 1):
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=18)

    # Adjust y-limits in order to leave enough space for the RMSE Annotations 
    for row_idx in range(n_age):
        # We only need to adjust the first axes in the row because sharey=True links them all
        ax_row = axes[row_idx, 0]
        
        # Get current autoscaled limits
        ymin, ymax = ax_row.get_ylim()
        
        # Increase upper limit by 10% or 15% to make room for text
        if row_idx >= n_age - 2:
            ax_row.set_ylim(bottom=ymin, top=ymax * 1.15)
        else:
            ax_row.set_ylim(bottom=ymin, top=ymax * 1.10)

    # --- Final Layout ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    fig.legend(unique_labels.values(), unique_labels.keys(), 
               loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=4, frameon=False, fontsize=28)

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=[0.0, 0.05, 1, 1])
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()