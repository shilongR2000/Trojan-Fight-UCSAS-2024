'''
Following the Genetic Algorithm process, we arrive at a selection of potential best teams. 
To determine the absolute best team, we conduct additional simulations on each of these candidates. 
This step involves generating a more definitive estimation for each team, allowing us to make an informed final selection.

When initiating the simulation, users simply need to select the most suitable training history CSV files. 
The code is designed to automatically extract the top three teams from the final generation of each file. 
It then removes any duplicate teams and runs the unique ones through approximately 2048 simulation cycles.

Upon completion of these simulations, we obtain detailed estimations for each team. 
What a great journey.

-------------------------------------------
Required Files:
Necessary Data Components: {
    {Gender}_B_summary_data.txt
    {Gender}_KNNK_athletes_dict.txt
    {Gender}_MuStd_List.txt
    {Gender}_B_kde_dict_all_athlete.joblib
    {Gender}_best_model_mean.joblib
    {Gender}_best_model_std.joblib
    {Gender}_df_encoded.csv
    }

{Gender}_TrainingHistory_{AvgScore}_Gen{Num}.csv
'''

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import warnings
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilenames

from joblib import load

import ast
import re

from SimulateRaceV17 import get_data, get_athlete_list_for_match, Simulate_match_1

def fitness(candidate):
    """caculate eatimated score, use function 'team_score' to achieve it"""
    
    df_US_ath = filtered_df[filtered_df['New_LN_FN'].isin(candidate)]
    
    print('-->', df_US_ath["New_LN_FN"].unique().tolist())
    
    average_score, variance_score, total_score_all, score_max = \
        Simulate_match_1(df_US_ath, necessary_data, match_times, df_tokyo, Other_Teams)
        
    return average_score, variance_score

def select_ByLowerBound(population, to_keep, Gender, match_times):
    """Keep the best team"""
    global population_with_fitness
    
    # Create a list of (individual, fitness) tuples
    population_with_fitness = []
    team_var = []
    for individual in population:
        fit, var = fitness(individual)  # Calculate the fitness for the individual
        
        sample_size = match_times
        lower_bound = fit - 1.96 * (var / np.sqrt(sample_size)) #95%Lower Bound
        
        population_with_fitness.append((lower_bound, individual, fit, var))


    
    # Sort the list of tuples by the fitness value in descending order
    population_with_fitness.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the sorted individuals
    
    sorted_LowerBound = [LowerBound for LowerBound, _, _, _ in population_with_fitness]
    
    sorted_population = [individual for _, individual, _, _ in population_with_fitness]
 
    # Calculate Avg_Scores after sorting
    Avg_Scores = [fit for _, _, fit, _ in population_with_fitness]
    sorted_team_var = [var for _, _, _, var in population_with_fitness]

    # Assign label（Team 1, Team 2, ...）
    team_labels = [f"Team {i+1}" for i in range(len(sorted_population))]
    
    
    #%% Start plot
    plt.figure(figsize=(16, 10))
    
    x_shifted = [i - 0.2 for i in range(len(sorted_population))]  # Shift x position for the second bar
    bars_LB = plt.bar(x_shifted, sorted_LowerBound, color='green', width=0.2, label='Team 95% Lower Bound')
    
    # Plot Avg_Scores
    bars_avg = plt.bar(team_labels, Avg_Scores, color='skyblue', width=0.2, label='Avg Score')

    # Plot team_var
    x_shifted = [i + 0.2 for i in range(len(sorted_population))]  # Shift x position for the second bar
    bars_var = plt.bar(x_shifted, sorted_team_var, color='orange', width=0.2, label='Team Variance')

    # Create a legend with team members
    team_member_info = ["\n".join(team) for team in sorted_population]
    for bar, team_info in zip(bars_avg, team_member_info):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), team_info, 
                ha='center', va='bottom', fontsize=8)

    plt.xlabel('Teams')
    plt.ylabel('Scores and Variance')
    plt.title('Outcome of Final Selection')
    if Gender == 'w':
        plt.ylim(0, 25)
    else:
        plt.ylim(0, 15)
    plt.xticks(range(len(sorted_population)), team_labels, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.show()

    return Avg_Scores, sorted_team_var, sorted_population[:to_keep]

#%% Main
if __name__ == '__main__': 
    #%% Read from past historical data
    print('Now popped selection window')
    # Set up the tkinter file dialog without the root window appearing
    tk.Tk().withdraw()

    # Open the file dialog to choose multiple files
    file_paths = askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

    match = re.search(r'^(.*?)_Training', file_paths[0])

    # Extract the matched string
    if match:
        Gender = match.group(1)[-1]
        print("Gender:", Gender)
    else:
        print("Pattern not found")
        
    
    global match_times
    match_times = 2048
    
    Do_Simulation = 0

    Plot_Flag = 0 # Whether Plot Distribution and Score Num_Of_Score_To_Get = 100

    
    population_size = 12 #10
    group_size = 5
    Parent_Keep_Number = 4
    
#%% Load the data
    with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_B_summary_data.txt', 'r') as file:
        B_summary_data = eval(file.read())

    with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_KNNK_athletes_dict.txt', 'r') as file:
        KNNK_athletes_dict = eval(file.read())
       
    with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_MuStd_List.txt', 'r') as file:
        MuStd_List = eval(file.read())
       
    B_kde_dict_all_athlete = load(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_B_kde_dict_all_athlete.joblib')
   
    best_model_mean = load(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_best_model_mean.joblib')
    best_model_std = load(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_best_model_std.joblib')
    df_encoded = pd.read_csv(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_df_encoded.csv')
   
    necessary_data = {
        'B_kde_dict_all_athlete': B_kde_dict_all_athlete,
        'KNNK_athletes_dict': KNNK_athletes_dict,
        'B_summary_data': B_summary_data,
        'best_model_mean': best_model_mean,
        'best_model_std': best_model_std,
        'df_encoded': df_encoded,
        'MuStd_List': MuStd_List
    }
        
    # df_usa_w, tokyo_team, df_all, team_all_name, df_tokyo = get_data(Gender)
    # team_all_name = get_athlete_list_for_match(df_usa_w, tokyo_team, df_all, team_all_name, df_tokyo)

    US_Team, Opponent_Country_List, df_all, df_tokyo = get_data(Gender)
    Other_Teams = get_athlete_list_for_match(Opponent_Country_List, df_all, df_tokyo, Gender)
   
    if Other_Teams['Gender'].iloc[0] == 'w':
        Apparatus = ['BB', 'FX', 'UE', 'VT']
    elif Other_Teams['Gender'].iloc[0] == 'm':
        Apparatus = ['PH', 'FX', 'HB', 'PB', 'SR', 'VT']
    

    with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_Filtered_Athlete.csv', 'r') as file:
        final_athletes = [line.strip() for line in file]
     
    athlete_names = final_athletes[1:]
    # Filter the DataFrame
    filtered_df = US_Team[US_Team['New_LN_FN'].isin(athlete_names)]

#%%
    # Check if any file was selected
    if not file_paths:
        print("No files selected.")
    else:
        # Read each CSV file into a DataFrame and store them in a list
        dfs = [pd.read_csv(file_path) for file_path in file_paths]

        combined_top_teams = pd.DataFrame()  # Initialize an empty DataFrame to store top teams

        for i in range(len(dfs)):
            last_gen = max(dfs[i]['Gen'])
            df_last = dfs[i][dfs[i]['Gen'] == last_gen]

            top_teams_df = df_last.groupby('Gen', group_keys=False).apply(lambda x: x.nlargest(3, 'avg_score'))

            # Convert the string representation of the list to actual lists
            top_teams_df['team_list'] = top_teams_df['team_list'].apply(ast.literal_eval)

            # Append to the combined DataFrame
            combined_top_teams = pd.concat([combined_top_teams, top_teams_df])

        # Sort the DataFrame by 'avg_score' in descending order
        combined_top_teams = combined_top_teams.sort_values(by='avg_score', ascending=False)

        # Remove duplicate teams based on 'team_list'
        combined_top_teams = combined_top_teams.drop_duplicates(subset='team_list')

        # Reset index if needed
        combined_top_teams.reset_index(drop=True, inplace=True)

    print(combined_top_teams)

    if Do_Simulation == 1:
        #%% Simulation and make final choice
        population = combined_top_teams['team_list']
        population = population.tolist()
        all_population = population
        Avg_Scores, sorted_team_var, best_team = select_ByLowerBound(population, 1, Gender, match_times)
    
        max_avg_score = max(Avg_Scores)
        max_avg_score = [max_avg_score] * len(Avg_Scores)
        gen = (i + 1)
        gen = [gen] * len(Avg_Scores)
        current = pd.DataFrame({
        "team_list": all_population,
        "avg_score": Avg_Scores,
        "variance": sorted_team_var,
        'Max_Avg_Score': max_avg_score,
        'Order': range(1,  len(Avg_Scores) + 1),
        'Gen': gen
        })
    
        print(best_team)
         
     