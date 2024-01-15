"""
After data supplement and cleaning, construct our database is undoubtly the basic of the whole model.
These code requires all the cleaned data include record from tokyo olypmic and other matches.
The first step in database construction is to restructure these records, 
combining them to form a JSON (dictionary) format. We organize the data by grouping it first by athlete and then by apparatus.

Next, we calculate the Kernel Density Estimation (KDE) Distribution and Gaussian Distribution for each athlete-apparatus pair. 
In addition, we develop and calculate various features, such as the average and variance of the Difficulty (D) score.

Once these constructions are complete, 
we train models to predict the parameters (Mean, Variance) of the Gaussian Distribution based on the features of an athlete in a specific apparatus. 
After identifying the best-performing model, we use it to calculate each athlete's Gaussian mean and variance in each apparatus, 
creating what we refer to as the MuStd_List.

Simultaneously, utilizing the same features, we construct a K-Nearest Neighbors (KNN) model. 
This model identifies the nearest neighbors for each athlete, allowing us to use the KDE Distribution of these neighboring athletes. 
The results are compiled into the KNNK_athletes_dict.

By successfully executing these steps, we accomplish the data preparation phase. 
Furthermore, we pre-calculated some essential data, which is crucial for ensuring the efficiency and speed of our simulations. 

-------------------------------------------
Required Files:
Cleaned_Data_2017_2021(With_Supplemental_Data).csv
Cleaned_Data_2022_2023.csv

-------------------------------------------
Output Files:
Necessary Data Components: {
    {Gender}_B_summary_data.txt
    {Gender}_KNNK_athletes_dict.txt
    {Gender}_MuStd_List.txt
    {Gender}_B_kde_dict_all_athlete.joblib
    {Gender}_best_model_mean.joblib
    {Gender}_best_model_std.joblib
    {Gender}_df_encoded.csv
    }
    
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import json
import random
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
    
      
#%%
def gaussian_distribution(x, mu, std):
     return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
 
#%%
def PlotDIstribution(mu, std, kde, data_for_kde, athlete, apparatus, len_df):#, mu_predicted, std_predicted):
# First Plot: Histogram with Fitted Gaussian
    plt.figure(figsize=(8, 6))

    distance = max(data_for_kde) - min(data_for_kde)
    if distance > 3.6:
        Bins = 20
    elif distance > 2:
        Bins = 12
    else:
        Bins = 6

    sns.distplot(data_for_kde, bins=Bins, kde=False, norm_hist=True, label='Histogram for historical records')

    x_vals = np.linspace(10, 16, 400)
    y_vals = gaussian_distribution(x_vals, mu, std)
    plt.plot(x_vals, y_vals, label="Fitted Gaussian", color='r', linewidth=2)

    plt.text(10.5, 0.9*max(y_vals), f'µ = {mu:.2f}\nσ = {std:.2f}', color='r', 
            bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.suptitle('Histogram with Fitted Gaussian')
    plt.title(f'Score Distribution for {athlete} in {apparatus} with {len_df} score recorded')
    plt.tight_layout()

    # Show the first plot
    plt.show()

    # Second Plot: KDE
    plt.figure(figsize=(8, 6))
    sns.distplot(data_for_kde, bins= Bins, kde=False, norm_hist=True, label='Histogram for historical records')

    x_grid = np.linspace(10, 16, 1000)
    y_vals_kde = kde.evaluate(x_grid)
    plt.plot(x_grid, y_vals_kde, label='KDE', color='g', linewidth=2)

    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title(f'Score Distribution for {athlete} in {apparatus} with {len_df} score recorded')
    plt.suptitle('Histogram with KDE')
    plt.tight_layout()

    # Show the second plot
    plt.show()

#%%
def load_combined_data():
    ori_data_2223 = pd.read_csv("Cleaned&ReOrganized_Data/Cleaned_Data_2022_2023.csv")
    ori_data_1721 = pd.read_csv("Cleaned&ReOrganized_Data/Cleaned_Data_2017_2021(With_Supplemental_Data).csv")
    return pd.concat([ori_data_1721, ori_data_2223], ignore_index=True)

def preprocess_data(df):
    df['New_LN_FN'] = df['New_LN_FN'].str.upper()
    df['Apparatus'] = df['Apparatus'].str.upper().replace(["VT1", "VT2"], "VT")
    df['Apparatus'] = df['Apparatus'].str.upper().replace(["UB", "UE"], "UE")
    return df

def organize_data_by_athlete(df):
    athlete_dict = {}
    for _, row in df.iterrows():
        athlete_name = row['New_LN_FN']
        apparatus = row['Apparatus']
        match_info = row.drop(['New_LN_FN', 'Apparatus']).to_dict()
        
        athlete_dict.setdefault(athlete_name, {}).setdefault(apparatus, []).append(match_info)
    return athlete_dict

def organize_data_by_apparatus(df):
    apparatus_dict = {}
    for _, row in df.iterrows():
        athlete_name = row['New_LN_FN']
        apparatus = row['Apparatus']
        match_info = row.drop(['New_LN_FN', 'Apparatus']).to_dict()

        apparatus_dict.setdefault(apparatus, {}).setdefault(athlete_name, []).append(match_info)
    return apparatus_dict

def create_summary(athlete_history, Gender):
    summary_data = []
    
    for athlete_name, apparatus_data in athlete_history.items():
        for apparatus, matches in apparatus_data.items():
            if matches[0]['Gender'] == Gender:
                data_entry = {
                    'Name': athlete_name,
                    'Country': matches[0]['Country'],
                    'Apparatus': apparatus,
                    'Matches Participated': len(matches),
                    'Average D_Score': np.mean([match['D_Score'] for match in matches]),
                    'D_Score Variance': np.var([match['D_Score'] for match in matches]),
                    'Average E_Score': np.mean([match['E_Score'] for match in matches]),
                    'E_Score Variance': np.var([match['E_Score'] for match in matches]),
                    'Average Penalty': np.mean([match['Penalty'] for match in matches]),
                    'Penalty Variance': np.var([match['Penalty'] for match in matches]),
                    'Matches Details': matches
                }
                summary_data.append(data_entry)
    
    summary_df = pd.DataFrame(summary_data).sort_values(by=['Name', 'Apparatus'])
    summary_df['Row Number'] = range(1, len(summary_df) + 1)
    return summary_df, summary_data

def reorder_summary_data(data):
    new_structure = {}

    for entry in data:
        athlete_name = entry['Name']
        apparatus = entry['Apparatus']

        # Extract and remove the 'Name', 'Country', and 'Apparatus' keys from the entry
        entry_data = {k: v for k, v in entry.items() if k not in ['Name', 'Country', 'Apparatus', 'Matches Details']}

        if athlete_name not in new_structure:
            new_structure[athlete_name] = {}
        new_structure[athlete_name][apparatus] = entry_data

    return new_structure

def reorder_summary_data_to_group(reorganized_dict):
    # Initialize an empty dictionary for the inverted data
    inverted_dict = {}

    # Iterate over each athlete and their nested dictionary
    for athlete_name, apparatus_data in reorganized_dict.items():
        for apparatus, data in apparatus_data.items():
            # Check if the apparatus is already added to the new dictionary
            if apparatus not in inverted_dict:
                inverted_dict[apparatus] = {}
            # Assign the athlete's data for the current apparatus
            inverted_dict[apparatus][athlete_name] = data

    return inverted_dict

#%%
# Function to prepare data for KDE
def prepare_data_for_kde(df):
    len_df = len(df)
    if len_df == 1: #Only have one score
        single_score = df['Score'].values[0]
        return np.array([single_score + 0.25, single_score - 0.25])
    elif df['Score'].nunique() == 1: #Have many scores but they are identical
        single_score = df['Score'].values[0]
        return np.tile([[single_score + 0.25], [single_score - 0.25]], (len_df, 1)).flatten()
    else: #Just many different score, great
        return df['Score'].values
        
def process_athlete_data(Athlete_history_dict, number_of_data_to_fit):
    # Initialize required dictionaries and counters
    A_kde_dict = {}
    A_Normal_Distribution_Parameter = {}
    A_kde_dict_all_athlete = {}
    number_data_fitted = 0
    number_data_fitted_gaussian = 0
    stop_processing = False
    
    print('Start to caculate kde and Gaussian\n')

    # Compute and display the total number of iterations
    total_iterations = sum(len(apparatus_data) for _, apparatus_data in Athlete_history_dict.items())
    if number_of_data_to_fit == -1:
        number_of_data_to_fit = total_iterations
    print(f"The loops have a total run of {total_iterations} times.")
    print(f"The loops will run a total of {number_of_data_to_fit} times.")
    
    print_it = np.floor(number_of_data_to_fit / 30)
    print('\nTO GO  :-------------------------------')
    print('CURRENT:-', end = '')
    # Iterate over each athlete and their associated data
    for athlete_name, apparatus_data in Athlete_history_dict.items():
        if stop_processing:
            break
        for apparatus, matches in apparatus_data.items():
            number_data_fitted += 1
            
            # Periodic logging
            if number_data_fitted % print_it == 0:
                print('-', end = '')

            # Ensure the apparatus exists in the dictionaries
            if apparatus not in A_kde_dict:
                A_kde_dict[apparatus] = {}
                A_Normal_Distribution_Parameter[apparatus] = {}
                A_kde_dict_all_athlete[apparatus] = {}

            all_data = [match.copy() for match in matches]
            df = pd.DataFrame(all_data)

            data_for_kde = prepare_data_for_kde(df)
            data = None if len(data_for_kde) != len(df) else df['Score'].values.reshape(-1, 1)
            
            # if df not being modified, fit Gaussian
            # If we have suitable data, fit a Gaussian Mixture Model
            if data is not None:
                number_data_fitted_gaussian += 1
                gmm = GaussianMixture(n_components=1)
                gmm.fit(data)
                mu, std = round(gmm.means_[0, 0], 6), round(np.sqrt(gmm.covariances_[0, 0, 0]), 6)
                A_Normal_Distribution_Parameter[apparatus][athlete_name] = {'mean': mu, 'std': std}


            kde = gaussian_kde(data_for_kde)
            A_kde_dict_all_athlete[apparatus][athlete_name] = kde

            A_kde_dict[apparatus][athlete_name] = kde
               
            if data is not None:
                # Periodic plotting
                if number_data_fitted_gaussian % 100 == 0:
                    len_df = len(df)
                    PlotDIstribution(mu, std, kde, data_for_kde, athlete_name, apparatus, len_df)

            # Break condition
            if number_data_fitted >= number_of_data_to_fit:
                print(f'\nSo far we have processed {number_data_fitted_gaussian} Gaussian data.')
                stop_processing = True
                break

    return A_kde_dict, A_Normal_Distribution_Parameter, A_kde_dict_all_athlete


def reorganize_dict(original_dict):
    # Initialize an empty dictionary for the reorganized data
    reorganized_dict = {}

    # Iterate over each apparatus and its nested dictionary
    for apparatus, athletes in original_dict.items():
        for athlete_name, athlete_data in athletes.items():
            # Check if the athlete is already added to the new dictionary
            if athlete_name not in reorganized_dict:
                reorganized_dict[athlete_name] = {}
            # Assign the athlete's data for the current apparatus
            reorganized_dict[athlete_name][apparatus] = athlete_data

    return reorganized_dict

#%%
def create_dataframe_from_summary(summary_data, distribution_parameters):
    """
    Create a dataframe from the given summary data and distribution parameters.
    
    Parameters:
    - summary_data: List of athlete summary data
    - distribution_parameters: Dictionary containing normal distribution parameters for each athlete
    
    Returns:
    - DataFrame with selected data from the summary and distribution parameters
    """
    refined_data = []

    # Iterate over the summary data
    for athlete_data in summary_data:
        athlete_name = athlete_data['Name']
        athlete_apparatus = athlete_data['Apparatus']

        # Check if the athlete's data exists in the distribution parameters
        if athlete_name in distribution_parameters.get(athlete_apparatus, {}):
            mean = distribution_parameters[athlete_apparatus][athlete_name].get('mean')
            std = distribution_parameters[athlete_apparatus][athlete_name].get('std')

            if mean is not None:
                current_entry = athlete_data.copy()
                del current_entry['Matches Details']
                current_entry['Mean'] = mean
                current_entry['Std'] = std
                refined_data.append(current_entry)

    # Convert the list to a DataFrame and drop rows with missing values
    return pd.DataFrame(refined_data).dropna()

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
        "Linear Regression": LinearRegression()
    }

    r2_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2_scores[name] = r2_score(y_test, predictions)
        
    return models, r2_scores

def plot_r2_scores(r2_scores_mean, r2_scores_std):
    sns.set_style("whitegrid")
    palette = sns.color_palette("coolwarm", n_colors=3)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    if Gender == 'm':
        Gen = 'Men'
    else:
        Gen = 'Women'

    # Plot R^2 scores for 'Mean' predictions
    sns.barplot(x=list(r2_scores_mean.keys()), y=list(r2_scores_mean.values()), palette=palette, ax=ax[0])
    ax[0].set_title(f"R^2 Scores for Predicting Mean on {Gen}", pad=20)
    ax[0].set_ylim([0, 1])
    for index, value in enumerate(r2_scores_mean.values()):
        ax[0].text(index, value + 0.02, f"{value:.3f}", ha='center', va='center', fontweight='bold')
    
    # Plot R^2 scores for 'Std' predictions
    sns.barplot(x=list(r2_scores_std.keys()), y=list(r2_scores_std.values()), palette=palette, ax=ax[1])
    ax[1].set_title(f"R^2 Scores for Predicting Std on {Gen}", pad=20)
    ax[1].set_ylim([0, 1])
    for index, value in enumerate(r2_scores_std.values()):
        ax[1].text(index, value + 0.02, f"{value:.3f}", ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

#%%
def train_models_for_mean_std(df_encoded):
    # Prepare features and targets
    X = df_encoded.drop(['Name', 'Country', 'Mean', 'Std'], axis=1)
    print('-----------------------------------')
    print(f"Number of columns: {len(X.columns)}")
    print("Column names:")
    print(X.columns)
    print('-----------------------------------')
    y_mean = df_encoded['Mean']
    y_std = df_encoded['Std']

    # Split data
    X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X, y_mean, test_size=0.2, random_state=42)
    X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y_std, test_size=0.2, random_state=42)
    
    # Train and evaluate models for 'Mean' and 'Std'
    models_mean, r2_scores_mean = train_and_evaluate_models(X_train_mean, y_train_mean, X_test_mean, y_test_mean)
    models_std, r2_scores_std = train_and_evaluate_models(X_train_std, y_train_std, X_test_std, y_test_std)

    # Plot R^2 scores
    plot_r2_scores(r2_scores_mean, r2_scores_std)

    # Identify and print best models
    best_model_mean = max(r2_scores_mean, key=r2_scores_mean.get)
    best_model_std = max(r2_scores_std, key=r2_scores_std.get)

    # Identify best models
    best_model_mean_name = max(r2_scores_mean, key=r2_scores_mean.get)
    best_model_std_name = max(r2_scores_std, key=r2_scores_std.get)
    best_model_mean = models_mean[best_model_mean_name]
    best_model_std = models_std[best_model_std_name]

    print(f"Best model for predicting 'Mean': {best_model_mean_name} with R^2 score of {r2_scores_mean[best_model_mean_name]:.3f}")
    print(f"Best model for predicting 'Std': {best_model_std_name} with R^2 score of {r2_scores_std[best_model_std_name]:.3f}")


    # Print coefficients for linear regression models
    print('\nCoefficients for Linear Regression (Mean):')
    print(models_mean["Linear Regression"].coef_)
    print('Coefficients for Linear Regression (Std):')
    print(models_std["Linear Regression"].coef_)
    return best_model_mean, best_model_std
    


#%%
def PlotKDEQuartic(athlete, apparatus, Gender, B_kde_dict_all_athlete, KNNK_athletes_dict,\
                   best_model_mean, best_model_std, df_encoded, MuStd_List):
    
    kde = B_kde_dict_all_athlete[athlete][apparatus]
    plt.figure(figsize=(8, 6))

    # Adjust legend to include the mean and std lines
    x_grid = np.linspace(10, 16, 1000)
    y_vals = kde.evaluate(x_grid)
    plt.plot(x_grid, y_vals, label=f'KDE - {athlete}', color = 'g', linewidth=3)

    NearAthletes = KNNK_athletes_dict[athlete][apparatus]
    
    shallow_green = (0.4, 0.8, 0.4)
    for other_athlete in NearAthletes:
        kde = B_kde_dict_all_athlete[other_athlete][apparatus]
        y_vals = kde.evaluate(x_grid)
        plt.plot(x_grid, y_vals, label=f'KDE - {other_athlete}', color = shallow_green, linewidth=2.5)
            
    #Now For normal distribution
    X = df_encoded
    
    global summary_data
    Gaussian_data_valid = 1
    filtered_df = X[X['Name'] == athlete]

    apparatus_columns_m = {
        'FX': 'Apparatus_FX',
        'HB': 'Apparatus_HB',
        'PB': 'Apparatus_PB',
        'PH': 'Apparatus_PH',
        'SR': 'Apparatus_SR',
        'VT': 'Apparatus_VT'
    }

    apparatus_columns_w = {
        'FX': 'Apparatus_FX',
        'UE': 'Apparatus_UE',
        'BB': 'Apparatus_BB',
        'VT': 'Apparatus_VT'
    }

    if Gender == 'm':
        if apparatus in apparatus_columns_m:
            column_name = apparatus_columns_m[apparatus]
            summary_data = filtered_df[filtered_df[column_name] == 1]
        else:
            print("Invalid apparatus code?")
            Gaussian_data_valid = 0

        if len(summary_data) == 0:
            Gaussian_data_valid = 0
    else:
        if apparatus in apparatus_columns_w:
            column_name = apparatus_columns_w[apparatus]
            summary_data = filtered_df[filtered_df[column_name] == 1]
        else:
            print("Invalid apparatus code?")
            Gaussian_data_valid = 0

        if len(summary_data) == 0:
            Gaussian_data_valid = 0
   
        
    if Gaussian_data_valid == 1:
        summary_data = summary_data.drop(['Name', 'Country', 'Mean', 'Std'], axis=1)
        summary_data = summary_data.values.tolist()
        summary_data = summary_data[0]

        mu = best_model_mean.predict([summary_data])
        std = best_model_std.predict([summary_data])
        
        x_vals = np.linspace(10, 16, 400)
        y_vals = gaussian_distribution(x_vals, mu, std)
        plt.plot(x_vals, y_vals, label="Fitted Gaussian", color = 'r', linewidth=3)
    
    else:
        #Now for predicted
        mu, std = MuStd_List[athlete][apparatus]
        x_vals = np.linspace(10, 16, 400)
        y_vals = gaussian_distribution(x_vals, mu, std)
        plt.plot(x_vals, y_vals, label="Fitted Gaussian(Only one Score)", color = 'r', linewidth=3, linestyle = ':')
        
        
    plt.title(f'Score Distribution for {athlete} in {apparatus} with its KNN neighbours')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.xlim(10, 16)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #%% And then plot these scores
    Individual_Scores = GetManyScore(athlete, apparatus, B_kde_dict_all_athlete,  KNNK_athletes_dict, MuStd_List, 400)

    # Calculate the average score and variance
    average_score = np.mean(Individual_Scores)
    variance = np.var(Individual_Scores)

    plt.style.use('seaborn-whitegrid')  # Using seaborn style for a more elegant look

    # Plotting the scores over attempts with improved aesthetics
    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(Individual_Scores)), Individual_Scores, 
                color='steelblue', label='Total Score')
    
    # Highlight the average line and improve the label
    plt.axhline(y=average_score, color='darkorange', linestyle='-', linewidth=2, label=f'Average Score: {average_score:.2f}')
    
    plt.ylim(0, 16)
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adding grid
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Adding title and labels with a more formal font
    plt.title('Total Scores Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Attempt Number', fontsize=12)
    plt.ylabel('Total Score', fontsize=12)

    # Improve the legend
    plt.legend(frameon=True, loc='upper right', fontsize=10)

    # Adding variance in a more subtle location with a smaller font size
    plt.text(len(Individual_Scores) * 0.5, average_score + average_score * 0.1, f'Variance: {variance:.3f}',
            horizontalalignment='center', color='darkorange', fontsize=10)

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    

#%%
def GetManyScore(athlete, apparatus, B_kde_dict_all_athlete,  KNNK_athletes_dict, MuStd_List, Num_Of_Score_To_Get):
 
    kde = B_kde_dict_all_athlete[athlete][apparatus]
    Score1s = kde.resample(Num_Of_Score_To_Get)#.astype(np.float32)

    NearAthletes = KNNK_athletes_dict[athlete][apparatus]

    for other_athlete in NearAthletes:
        kde = B_kde_dict_all_athlete[other_athlete][apparatus]
        Score2a = (kde.resample(Num_Of_Score_To_Get))#.astype(np.float32)
        Score2b = (kde.resample(Num_Of_Score_To_Get))#.astype(np.float32)
        Score2c = (kde.resample(Num_Of_Score_To_Get))#.astype(np.float32)
    

    Score2s = list(map(lambda x, y, z: x + y + z, Score2a, Score2b, Score2c))
    Score2s = [x / 3 for x in Score2s]

    #Now For normal distribution
    [mu, std] = MuStd_List[athlete][apparatus]
    if std < 0:
        std = - std
    Score3s = np.random.normal(mu, std, Num_Of_Score_To_Get)#.astype(np.float32)
    
    Individual_Scores = list(map(lambda x, y, z: x + y + z, Score1s, Score2s, Score3s))
    Individual_Scores = [x / 3 for x in Individual_Scores]
    
    Individual_Scores = Individual_Scores[0].tolist()  
    #print(athlete, 'in', apparatus, 'gets', Score)
   
    return Individual_Scores
    
    
#%%
def filter_single_matches(data):
    for apparatus in data:
        athletes_to_remove = [athlete for athlete, stats in data[apparatus].items() if stats['Matches Participated'] == 1]
        for athlete in athletes_to_remove:
            del data[apparatus][athlete]
    return data


# 2. Normalize the B_summary_data
def normalize_data(data):
    flattened_data = {}
    for apparatus, athletes in data.items():
        for athlete, stats in athletes.items():
            flattened_key = f"{athlete}|||{apparatus}"  # Modified separator
            flattened_data[flattened_key] = stats
            
    df = pd.DataFrame(flattened_data).T
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    # Reshape the normalized data back to the original structure
    normalized_data = {}
    for index, row in normalized_df.iterrows():
        apparatus, athlete = index.split("|||")  # Modified separator
        if athlete not in normalized_data:
            normalized_data[athlete] = {}
        normalized_data[athlete][apparatus] = row.to_dict()
        
    return normalized_data


#%%
def euclidean_distance(behavior1, behavior2):
    """Calculate the Euclidean distance between two behaviors."""
    common_keys = set(behavior1.keys()).intersection(set(behavior2.keys()))
    sum_squared_difference = sum((behavior1[key] - behavior2[key]) ** 2 for key in common_keys)
    return sum_squared_difference


def find_closest_k_for_apparatus(athlete_behavior, apparatus_data, k=3):
    """Find the k closest athletes for a specific apparatus."""
    
    distances = []
    for athlete, behavior in apparatus_data.items():
        distance = euclidean_distance(athlete_behavior, behavior)
        distances.append((athlete, distance))
    distances.sort(key=lambda x: x[1])  # Sort by distance

    return [athlete for athlete, _ in distances[:k]]

def knn_for_apparatus(B_summary_data, A_summary_data):
    """Find the k closest athletes for each athlete in B and each apparatus in A."""
    closest_athletes_dict = {}
    
    for athlete, behavior in B_summary_data.items():
        closest_athletes_dict[athlete] = {}
        for apparatus, behavior_target in B_summary_data[athlete].items():
            apparatus_data = A_summary_data[apparatus]
            
            # if athlete in apparatus_data:
            #     del apparatus_data[athlete]
            closest_athletes_dict[athlete][apparatus] = find_closest_k_for_apparatus(behavior_target, apparatus_data)
    
    return closest_athletes_dict


def plot_athlete_comparison(B_summary_data, closest_athletes_dict, A_summary_data, athlete_name, apparatus):
    """
    Plots a comparison of the given athlete's stats with their matched athletes for a specific apparatus.

    Parameters:
    - B_summary_data: Dictionary containing the summary data for group B athletes.
    - closest_athletes_dict: Dictionary containing the closest matches for each athlete.
    - A_summary_data: Dictionary containing the summary data for group A athletes.
    - athlete_name: Name of the athlete in group B to be compared.
    - apparatus: The apparatus for which the comparison is made.
    """
    # Fetch the athlete's stats from B_summary_data
    athlete_stats = B_summary_data[athlete_name][apparatus]

    # Fetch the matched athletes for the apparatus
    matched_athletes = closest_athletes_dict[athlete_name][apparatus]

    all_athlete_names = [athlete_name] + matched_athletes
    all_data = [athlete_stats]

    for name in matched_athletes:
        match_stats = A_summary_data[apparatus][name]
        all_data.append(match_stats)

    # Plotting
    metrics = list(athlete_stats.keys())

    colors = [
    (0.8, 0.4, 0.4),
    (0.8, 0.8, 0.4),
    (0.4, 0.8, 0.4),
    (0.4, 0.8, 0.8),
    (0.4, 0.4, 0.8),
    (0.8, 0.4, 0.8)
    ]
   
    bar_width = 0.15
    index = np.arange(len(all_athlete_names))

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        values = [data[metric] for data in all_data]
        # Use modulo to cycle through colors
        ax.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Athlete Name')
    ax.set_ylabel('Value')
    ax.set_title(f'Comparison of {apparatus} stats for {athlete_name} and their matches')
    ax.set_xticks(index + bar_width * len(metrics) / 2)
    ax.set_xticklabels(all_athlete_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    
    
#%% GetMuSTD_Data
def GetMuStd(best_model_mean, best_model_std, df_encoded, B_summary_data, Gender):
    print("\nGetting Mu Std Data", end = '')
    df_encoded_Full = Get_df_encoded_Full(B_summary_data)
    
    MuStd_List = {}
    
    if Gender == 'm':
        column_to_apparatus = {'Apparatus_FX': 'FX',
                               'Apparatus_HB': 'HB',
                               'Apparatus_PB': 'PB',
                               'Apparatus_PH': 'PH',
                               'Apparatus_SR': 'SR',
                               'Apparatus_VT': 'VT'}
    else:
        column_to_apparatus = {'Apparatus_FX': 'FX',
                               'Apparatus_BB': 'BB',
                               'Apparatus_UE': 'UE',
                               'Apparatus_VT': 'VT'}
        
    #for athlete, behavior in B_summary_data.items():
        
    loop = len(df_encoded_Full)
    div = round(loop/5)
    for i in range(loop):
        if i%div == 0:
            print('.', end = '')
        global row
        row = df_encoded_Full.iloc[i]
        name = df_encoded_Full.iloc[i]['Athlete']
        
        apparatus = None
        for column, apparatus_code in column_to_apparatus.items():
            if row[column] == 1:
                apparatus = apparatus_code
                break
            
        summary_data = row
        summary_data = summary_data.drop(['Athlete'], axis=0)
        summary_data = summary_data.values.tolist()
        
        mu = best_model_mean.predict([summary_data])
        std = best_model_std.predict([summary_data])
        
        
        # Check if athlete already exists
        if name not in MuStd_List:
            # Athlete doesn't exist, create entry
            MuStd_List[name] = {} 
        
        # Check if apparatus exists for athlete
        if apparatus not in MuStd_List[name]:
            # Apparatus doesn't exist, create it
            MuStd_List[name][apparatus] = [float(mu), float(std)]
      
    print("\nMu Std Created")
    return MuStd_List

def Get_df_encoded_Full(B_summary_data):
    df = pd.DataFrame(columns=['Athlete', 'Apparatus', 'Matches Participated', 'Average D_Score', 
                           'D_Score Variance', 'Average E_Score', 'E_Score Variance',
                           'Average Penalty', 'Penalty Variance']) 
    for athlete, v1 in B_summary_data.items():
        for apparatus, v2 in v1.items():
            row = {
                'Athlete': athlete,
                'Apparatus': apparatus
            }
            row.update(v2)

            df = df.append(row, ignore_index=True)
    df_encoded_Full = pd.get_dummies(df, columns=['Apparatus'])
    return df_encoded_Full
  
    
  
    
  
#%%
if __name__ == '__main__':
    
    Gender = 'm' # 'm' / 'w'
    OverWrite_Data = 1 # Whether save the new data
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Preprocess data and organize it
    ori_data = preprocess_data(load_combined_data())
    Athlete_history_dict = organize_data_by_athlete(ori_data)
    apparatus_history_dict = organize_data_by_apparatus(ori_data)
    
    # Create summary data
    summary_df, summary_data = create_summary(Athlete_history_dict, Gender)
    B_summary_data = reorder_summary_data(summary_data)
    A_summary_data = reorder_summary_data_to_group(B_summary_data)
    
    #%% Output Summary data
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path for the output file
    output_path_B = os.path.join(current_directory, f"Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_B_summary_data.txt")
    
    if OverWrite_Data == 1:
        with open(output_path_B, "w") as fb:
            fb.write(json.dumps(B_summary_data, indent=4))
    
    #%% Now Training Model
    # Process athlete data
    number_of_data_to_fit = -1 # -1 means fit every data
    A_kde_dict, A_Normal_Distribution_Parameter, A_kde_dict_all_athlete =\
        process_athlete_data(Athlete_history_dict, number_of_data_to_fit)
   
    # Rearrange the data
    B_kde_dict = reorganize_dict(A_kde_dict)
    B_Normal_Distribution_Parameter = reorganize_dict(A_Normal_Distribution_Parameter)

    B_kde_dict_all_athlete = reorganize_dict(A_kde_dict_all_athlete)
    
    if OverWrite_Data == 1:
        dump(B_kde_dict_all_athlete, f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_B_kde_dict_all_athlete.joblib')
    
    # Train models if the flag is set

    print('\nTraining Start:')
    
    # Create a dataframe from the summary data
    df = create_dataframe_from_summary(summary_data, A_Normal_Distribution_Parameter)
    df_encoded = pd.get_dummies(df, columns=['Apparatus'])
    
    # Execute the main function
    best_model_mean, best_model_std = train_models_for_mean_std(df_encoded)
    if OverWrite_Data == 1:
        dump(best_model_mean, f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_best_model_mean.joblib')
        dump(best_model_std, f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_best_model_std.joblib')
        df_encoded.to_csv(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_df_encoded.csv', index=False)
        
    # For test
    # TX = [2, 5, 0, 8, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # prediction = rf_model_std.predict(np.array(TX).reshape(1, -1))
    # print(prediction)
    
    #%% KNN-K
    A_summary_data = filter_single_matches(A_summary_data)

    with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_B_summary_data.txt', 'r') as file:
        B_summary_data = eval(file.read())
        
        
    normalized_B_data = normalize_data(B_summary_data)
    normalized_A_data = normalize_data(A_summary_data)
    
    closest_athletes_dict = knn_for_apparatus(normalized_B_data, normalized_A_data)
    # closest_athletes_dict = knn_for_apparatus(B_summary_data, A_summary_data)

    plot_number = 0
    max_plot_number = 6
    Normaliezd_visual = 1
    
    while plot_number < max_plot_number:
        plot_number += 1
        random_int = random.randint(0, len(B_summary_data) - 1)
        athlete_name = list(B_summary_data.keys())[random_int]
        apparatus = list(B_summary_data[athlete_name].keys())[0]
   
        if Normaliezd_visual == 0:
            plot_athlete_comparison(B_summary_data, closest_athletes_dict, A_summary_data, athlete_name, apparatus)
        else:
            plot_athlete_comparison(normalized_B_data, closest_athletes_dict, normalized_A_data, athlete_name, apparatus)
    
    KNNK_athletes_dict = closest_athletes_dict
    if OverWrite_Data == 1:
        with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_KNNK_athletes_dict.txt', "w") as fb:
            fb.write(json.dumps(KNNK_athletes_dict, indent=4))
 
    MuStd_List = GetMuStd(best_model_mean, best_model_std, df_encoded, B_summary_data, Gender)
    if OverWrite_Data == 1:
        with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_MuStd_List.txt', "w") as fb:
            fb.write(json.dumps(MuStd_List, indent=4))
        
    #%% Now Plot
    for i in range(0,10):
        # Randomly select a key from the dictionary
        random_athlete = random.choice(list(KNNK_athletes_dict.keys()))
        random_apparatus = random.choice(list(KNNK_athletes_dict[random_athlete].keys()))
        PlotKDEQuartic(random_athlete, random_apparatus, Gender, B_kde_dict_all_athlete, KNNK_athletes_dict,\
                           best_model_mean, best_model_std, df_encoded, MuStd_List)