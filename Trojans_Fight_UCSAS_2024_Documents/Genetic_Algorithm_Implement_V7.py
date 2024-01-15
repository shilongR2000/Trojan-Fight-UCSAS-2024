'''
The Core Code of our model is designed to output the final candidate teams using a Genetic Algorithm. 
The process begins by selecting athletes from a filtered group. In the first generation, we randomly assemble 12 different teams, 
each comprising five athletes, ensuring that each team meets the basic criteria to participate in all competitions.

To obtain a stable and reliable performance estimate for each team, we run simulations 1024 times. 
This extensive simulation process allows us to derive a reliable grade for each team's performance. 
Based on these grades, the best four teams are automatically advanced to the next generation. Meanwhile, the six teams with the lowest grades are eliminated. 
The remaining top six teams undergo a process of crossing and mutation to create eight new teams, 
resulting in a total of 12 optimized teams for the next generation.

This process is repeated for 12 generations, continuously refining and optimizing the team compositions. 
Of course, all these parameters can be easily modified.
The progress and outcomes of each generation, including team compositions and their respective grades, are meticulously recorded and exported to a CSV file. 
This file serves as a comprehensive record of the algorithm's outputs, documenting the evolution and optimization of team selections across generations.

Incorporating the feature to train from historical data, controlled by setting the 'Use_History' variable to 1, 
offers significant convenience. This functionality enables users to pick up where previous training sessions left off, 
using an existing training history CSV file. This approach is particularly advantageous for iterative model development, 
as it eliminates the need to start the training process from scratch each time. 

Through numerous optimizations, the program has been designed to operate with sufficient running speed. 

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

{Gender}_Filtered_Athlete.csv

-------------------------------------------
Output Files:
{Gender}_TrainingHistory_{AvgScore}_Gen{Num}.csv

'''

import random
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from spyder_kernels.utils.iofuncs import load_dictionary
from random import randint

import time
from joblib import load

import ast
import tkinter as tk
from tkinter.filedialog import askopenfilename

import re

from SimulateRaceV17 import get_data, get_athlete_list_for_match, GetUSSample, Simulate_match_1


#%% Basic Gene functions
def initialize_population(pop_size, group_size, athletes_list, Apparatus):
    """Initialize the population and randomly select athletes from the athlete list to form a team"""
    population = []
    for _ in range(pop_size):
        sampled_names, _ = GetUSSample(athletes_list, Apparatus)
        population.append(sampled_names.tolist())
    return population

def fitness(candidate):
    """caculate eatimated score, use function 'team_score' to achieve it"""
    
    df_US_ath = filtered_df[filtered_df['New_LN_FN'].isin(candidate)]
    
    print('-->', df_US_ath["New_LN_FN"].unique().tolist())
    
    average_score, variance_score, total_score_all, score_max = \
        Simulate_match_1(df_US_ath, necessary_data, match_times, df_tokyo, Other_Teams)
        
    return average_score, variance_score

def select(population, to_keep):
    """Keep the best team"""
    global population_with_fitness
    
    # Create a list of (individual, fitness) tuples
    population_with_fitness = []
    team_var = []
    for individual in population:
        fit, var = fitness(individual)  # Calculate the fitness for the individual
        population_with_fitness.append((individual, fit))
        team_var.append((var, fit))
    
    # Sort the list of tuples by the fitness value in descending order
    population_with_fitness.sort(key=lambda x: x[1], reverse=True)
    team_var.sort(key=lambda x: x[1], reverse=True)

    # Extract the sorted individuals
    sorted_population = [individual for individual, _ in population_with_fitness]
    sorted_team_var = [var for var, _ in team_var]
    
    # Calculate Avg_Scores after sorting
    Avg_Scores = [fit for _, fit in population_with_fitness]
    
    return Avg_Scores, sorted_team_var, sorted_population[:to_keep]

def crossover(parent1, parent2, group_size):
    """Crossover operation ensuring no duplicate members and a fixed group size."""
    # Combine and remove duplicates
 
    combined_list = parent1 + parent2
    child = []
    
    while len(child) < group_size:
        # Randomly select a name
        chosen_name = random.choice(combined_list)
    
        # Add the chosen name to the selected list
        child.append(chosen_name)
    
        # Remove all instances of the chosen name to avoid duplicates
        combined_list = [name for name in combined_list if name != chosen_name]
    return child

def mutate(candidate, filtered_df, mutation_rate = 0.2):
    """Mutation operation ensuring no duplicate members."""
    if random.random() < mutation_rate:
        mutate_point = random.randint(0, len(candidate) - 1)
        # print('Now Mutate!!!')
        # Ensure the new candidate is not already in the team
        unique_names = filtered_df['New_LN_FN'].drop_duplicates()
        available_candidates = unique_names[~unique_names.isin(child)]
        new_member = available_candidates.sample(n=1)
        new_member = new_member.tolist()[0]
        candidate[mutate_point] = str(new_member)

    return candidate


#%% Check US Team
def CkeckUSSample(US_Team):
    
    sampled_names = US_Team
    try:
        df_US_ath = US_Team[US_Team.isin(sampled_names)]
    except:
        US_Team = pd.Series(US_Team)
        df_US_ath = US_Team[US_Team.isin(sampled_names)]
 
 #Ensure each apparatus have at least 3 scores that such team can join 'team all-around'
    try:
        if not all(df_US_ath.groupby("Apparatus").size().reindex(Apparatus, fill_value=0) >= 3):
            Pass_status = 0
        else:
            Pass_status = 1
    except: 
        Pass_status = 1 #Only when using history that we encount such problem
    return Pass_status


def CkeckUSTeam(child):
    
    child = pd.Series(child)
    df_US_ath = US_Team[US_Team['New_LN_FN'].isin(child)]
 
    #Ensure each apparatus have at least 3 scores that such team can join 'team all-around'
    if not all(df_US_ath.groupby("Apparatus").size().reindex(Apparatus, fill_value=0) >= 3):
        Pass_status = 0
    else:
        Pass_status = 1
         
    return Pass_status

#%% Plot part
def PlotAGeneration(generations, gen_avg, gen_var):
    plt.figure(figsize=(10, 6))
    generations_list = list(range(1, generations + 1))

    # Plotting gen_avg with a line
    plt.plot(generations_list, gen_avg, color='royalblue', label='Average', linewidth=2, marker='o')
    
    # Plotting gen_var with a line
    plt.plot(generations_list, gen_var, color='coral', label='Variance', linewidth=2, marker='x')
    
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Adding title and labels
    plt.title('Generation Average and Variance Over Time', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    
    # Adding a legend
    plt.legend()
    
    # Display the plot
    plt.show()
    
#%% Main
if __name__ == '__main__':
    #Data Preparation
    Gender = 'w' # 'w'
    global match_times
    match_times = 1024

    Plot_Flag = 0 # Whether Plot Distribution and Score Num_Of_Score_To_Get = 100

    Use_History = 1

    generations = 4

    
    if Use_History == 0:
    # parameter
        population_size = 12 #10
        group_size = 5
        Parent_Keep_Number = 4
        Gender = Gender
    else:
        # Set up the tkinter file dialog without the root window appearing
        tk.Tk().withdraw()

        # Open the file dialog to choose the file
        file_path = askopenfilename(title="Select the CSV file", filetypes=[("CSV files", "*.csv")])

        
        if not file_path:
            print("No file selected.")
        else:
            # Read the CSV file into a DataFrame
            history = pd.read_csv(file_path)
        for i in range(len(history)):
            history['team_list'][i] = ast.literal_eval(history['team_list'][i])
            
        match = re.search(r'^(.*?)_Training', file_path)

        # Extract the matched string
        if match:
            Gender = match.group(1)[-1]
            print("Gender:", Gender)
        else:
            print("Pattern not found")
            
 
        trained_generations = np.max(history['Gen'])
        population_size = int(len(history) / trained_generations)
        group_size = 5
        Parent_Keep_Number = 4
    
   
    # Load the data
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
     
     
    # Initialize the population
    gen_avg = []
    gen_var = []
    # Start Iteration
    if Use_History == 0:
        Start_Gen = 0
        population = initialize_population(population_size, group_size, filtered_df, Apparatus)
    else:
        Start_Gen = trained_generations 
        for i in range(trained_generations):
            gen_avg.append(np.mean(history[history['Gen'] == i + 1]['avg_score']))
            gen_var.append(np.mean(history[history['Gen'] == i + 1]['variance']))
        generations = generations + Start_Gen
        population = history[history['Gen'] == Start_Gen]['team_list']
        population = population.tolist()
        # Again modifiy
        offspring = []
        for i in range(Parent_Keep_Number):#Keep best 2 parent
            best_parent = population[i]
            if best_parent not in offspring:
                offspring.append(best_parent)
        
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2, group_size)
            child = mutate(child, filtered_df)
            child = sorted(child)
            Pass_status = CkeckUSSample(child)
            # Check if the generated child is not already in the offspring list
            if Pass_status == 1:
                if child not in offspring:
                    offspring.append(child)

        population = offspring
        
    #%% Start Gene Alg
    for i in range(Start_Gen, generations):
       # Selection
       print('--------------------------')
       print('[The', i + 1, 'th generation]')
       start_time = time.time()
       
       # Core, to simulate the race 
       all_population = population
       Avg_Scores, sorted_team_var, population = select(population, population_size // 2)
       
       print('In this gen we used:', time.time() - start_time, 'seconds')
       
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
       
       try:
           history = pd.concat([history, current], axis=0, ignore_index=True)
       except:
           history = current
       
       # Automatically generate labels based on the length of Avg_Scores
       labels = [f'Score {i+1}' for i in range(len(Avg_Scores))]
       # Creating the bar chart
       plt.figure(figsize=(10, 6))
       plt.bar(labels, Avg_Scores, color='lightblue')
        
       avg_score = np.mean(Avg_Scores)
       variance = np.var(Avg_Scores)
        
       gen_avg.append(avg_score)
       gen_var.append(variance)

       # Add a horizontal line for the average score
       plt.axhline(y=avg_score, color='orange', linestyle='-', label=f'Average Score: {avg_score:.3f}')
       plt.plot([], [], ' ', label=f'Variance: {variance:.3f}')
        
       # Adding title and labels
       plt.title(f'Average Scores On Gen [{i + 1}]')
       plt.xlabel('Labels')
       plt.ylabel('Average Scores')
       plt.ylim(0, 25)
       # Display the chart
       plt.show()
        
       if i < generations - 1:#If not in last gen, we do generate new gen after test last gen
            # Cross and Mutate
            print('Now start to make next gen:')
            offspring = []
            for i in range(Parent_Keep_Number):#Keep best 2 parent
                best_parent = population[i]
                print(best_parent)
                if best_parent not in offspring:
                    offspring.append(best_parent)
            
            print('- - - - -Those above are the best parent that we choose to keep - - - - -')
            while len(offspring) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child = crossover(parent1, parent2, group_size)
                child = mutate(child, filtered_df)
                child = sorted(child)
                Pass_Status = CkeckUSTeam(child)
                # Check if the generated child is not already in the offspring list
                if Pass_Status == 1:
                    if child not in offspring:
                        offspring.append(child)
                        print(child)
               
            population = offspring
            
    #%% Plot it after each gen
    PlotAGeneration(generations, gen_avg, gen_var)
    
    #%% Give the best reault
    print('--------------------------')
    print('[Genetic Algorithm with', generations, 'step finished]')
    print("Now run the best team again")
    Final_Scores, sorted_team_var, best_solution = select(population[0:1], 1)#only input best 2 team to have another final try
    # print("Best Team:", sorted(best_solution[0]))
    # print("Team Score:", Final_Scores)
    
    Final_Scores = round(Final_Scores[0], 2)
    history.to_csv(f'Training_History/Training_History_{Gender}/{Gender}_TrainingHistory_{Final_Scores}_Gen{generations}.csv', index=False)