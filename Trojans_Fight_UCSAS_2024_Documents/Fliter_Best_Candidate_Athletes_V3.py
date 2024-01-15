'''
This code is specifically designed to identify the best candidates from the extensive pool of talented US athletes. 
Given the sheer number of skilled individuals, finding the optimal five-person team for our model could be quite challenging. 

To streamline this process, we employ simulation code for each athlete across different apparatuses. 
In each apparatus, we determine the top five athletes based on their performance metrics. 
These top performers are then compiled into a candidate list. 
This list represents our most promising athletes and serves as an ideal starting point for our subsequent genetic algorithm.

By focusing on these selected candidates, we significantly enhance the efficiency of the optimization process. 
This targeted approach accelerates the convergence speed of the algorithm, 
ensuring that we can effectively identify the best five-person team without getting overwhelmed by the vast number of potential combinations.

The outcome of this process is systematically documented. The code outputs a CSV file named {Gender}_Filtered_Athlete.csv.

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

-------------------------------------------
Output Files:
{Gender}_TrainingHistory_{AvgScore}_Gen{Num}.csv
'''

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from spyder_kernels.utils.iofuncs import load_dictionary
from random import randint

import time
from joblib import load

import math
import csv

#%%
def GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get):
    step_times = []
    
    start_time = time.time()#1
    B_kde_dict_all_athlete = necessary_data['B_kde_dict_all_athlete']
    KNNK_athletes_dict = necessary_data['KNNK_athletes_dict']
 
    kde = B_kde_dict_all_athlete[athlete][apparatus]
    Score1s = kde.resample(Num_Of_Score_To_Get)

    NearAthletes = KNNK_athletes_dict[athlete][apparatus]
    step_times.append(time.time() - start_time)#1
    
    
    start_time = time.time()#2
    
    for other_athlete in NearAthletes:
        kde = B_kde_dict_all_athlete[other_athlete][apparatus]
        Score2a = (kde.resample(Num_Of_Score_To_Get))
        Score2b = (kde.resample(Num_Of_Score_To_Get))
        Score2c = (kde.resample(Num_Of_Score_To_Get))
    

    Score2s = list(map(lambda x, y, z: x + y + z, Score2a, Score2b, Score2c))
    Score2s = [x / 3 for x in Score2s]
    step_times.append(time.time() - start_time)#2
    
    start_time = time.time()#3
    #Now For normal distribution

    MuStd_List = necessary_data['MuStd_List'] 
    
    [mu, std] = MuStd_List[athlete][apparatus]
    if std < 0:
        std = -std
    Score3s = np.random.normal(mu, std, Num_Of_Score_To_Get)
    
    step_times.append(time.time() - start_time)#3
    
    
    Scores = list(map(lambda x, y, z: x + y + z, Score1s, Score2s, Score3s))
    Scores = [x / 3 for x in Scores]
    
    Scores = Scores[0].tolist()  
    #print(athlete, 'in', apparatus, 'gets', Score)
   
    return Scores


#%%
def gaussian_distribution(x, mu, std):
     return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
   
def PlotKDEQuartic_Modified2(athlete, apparatus, necessary_data):
    B_kde_dict_all_athlete = necessary_data['B_kde_dict_all_athlete']
    KNNK_athletes_dict = necessary_data['KNNK_athletes_dict']

    kde = B_kde_dict_all_athlete[athlete][apparatus]
    
    plt.figure(figsize=(8, 6))
    # sns.distplot(data_for_kde, bins=20, kde=False, norm_hist=True)

    # Adjust legend to include the mean and std lines
    x_grid = np.linspace(4, 16, 1000)
    y_vals = kde.evaluate(x_grid)
    plt.plot(x_grid, y_vals, label=f'KDE - {athlete}', color = 'g', linewidth=3)

    
    NearAthletes = KNNK_athletes_dict[athlete][apparatus]
    
    for other_athlete in NearAthletes:
        kde = B_kde_dict_all_athlete[other_athlete][apparatus]
        y_vals = kde.evaluate(x_grid)
        plt.plot(x_grid, y_vals, label=f'KDE - {other_athlete}', color = 'y', linewidth=3)
            

    [mu, std] = MuStd_List[athlete][apparatus]
    x_vals = np.linspace(4, 16, 400)
    y_vals = gaussian_distribution(x_vals, mu, std)
    plt.plot(x_vals, y_vals, label="Fitted Gaussian", color = 'r', linewidth=3)
    
    plt.title(f'Score Distribution for {athlete} in {apparatus} with its KNN neighbours')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.xlim(4, 16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    


#%%

def get_data(Gender):
    df1 = pd.read_csv('Cleaned&ReOrganized_Data/Cleaned_Data_2017_2021(With_Supplemental_Data).csv')
    df2 = pd.read_csv('Cleaned&ReOrganized_Data/Cleaned_Data_2022_2023.csv')
    df3 = pd.concat([df1, df2], axis=0)
    
    df3["Apparatus"] = df3["Apparatus"].replace(["VT1","VT2"],"VT")
    df3["Apparatus"] = df3["Apparatus"].replace("hb","HB")
    df3["Apparatus"] = df3["Apparatus"].replace("UB","UE")
    df3['New_LN_FN'] = df3['New_LN_FN'].str.upper()
    
    df1["Apparatus"] = df1["Apparatus"].replace(["VT1","VT2"],"VT")
    df1["Apparatus"] = df1["Apparatus"].replace("hb","HB")
    df1["Apparatus"] = df1["Apparatus"].replace("UB","UE")
    df1['New_LN_FN'] = df1['New_LN_FN'].str.upper()
    
    df_tokyo = df1.groupby(["New_LN_FN","Gender","Country","Apparatus"])["Score"].mean().reset_index()
    df_tokyo = df_tokyo[df_tokyo["Country"] != "USA"]
    
    df_US = df3.groupby(["New_LN_FN","Gender","Country","Apparatus"])["Score"].mean().reset_index()
    df_US = df_US[df_US["Country"] == "USA"]
    US_Team = df_US[df_US["Gender"] == Gender]
    
    #Get the list of participants from other 11 countries. 
    
    tokyo_team = ["ROC","CHN","FRA","BEL","GBR","ITA","JPN","GER","CAN","NED","ESP"]
   
    df_all = df3.groupby(["New_LN_FN","Gender","Country","Apparatus"])["Score"].mean().reset_index()
    df_all = df_all[df_all["Gender"] == Gender]
    team_all_name = pd.DataFrame()
    return US_Team, tokyo_team, df_all, team_all_name, df_tokyo 



#%%
def ReOrg_to_apparatus(ALL_XXX):   # Reorganizing the dictionary
    reorganized_dict = {}
    for athlete, apparatus_scores in ALL_XXX.items():
        for apparatus, scores in apparatus_scores.items():
            if apparatus not in reorganized_dict:
                reorganized_dict[apparatus] = {}
            reorganized_dict[apparatus][athlete] = scores
    
    return reorganized_dict



#%%
if __name__ == '__main__':
   
    start_time0 = time.time()#00
    
    Gender = 'w' # 'w'
    Plot_Flag = 0 #Whether Plot Distribution and Score
    Num_Of_Score_To_Get = 2048
    FilterTopN = 5
    
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
    
    US_Team, tokyo_team, df_all, team_all_name, df_tokyo = get_data(Gender)
    
    if Gender == 'w':
        Apparatus = ['BB', 'FX', 'UE', 'VT']
    elif Gender == 'm':
        Apparatus = ['PH', 'FX', 'HB', 'PB', 'SR', 'VT']
    

    ALL_Scores = {}
    
    start_time = time.time()#0
    for i in range(len(US_Team['New_LN_FN'])):
        athlete = US_Team['New_LN_FN'].iloc[i]
        apparatus = US_Team['Apparatus'].iloc[i]
        Scores = GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get)
        if athlete not in ALL_Scores:
            ALL_Scores[athlete] = {}
        ALL_Scores[athlete][apparatus] = Scores
    
    
    ALL_MeanVar = {}
    for athlete in ALL_Scores:
        for apparatus in ALL_Scores[athlete]:
            lst = ALL_Scores[athlete][apparatus]
            mean = sum(lst) / len(lst)
            variance = sum((x - mean) ** 2 for x in lst) / len(lst)
            if athlete not in ALL_MeanVar:
                ALL_MeanVar[athlete] = {}
            ALL_MeanVar[athlete][apparatus] = [mean, variance] 
    
    # Z-score for 95% confidence interval (change this value for different confidence levels)
    z_score = 1.96
    z_score = 1.645 #90%
    
    ALL_Range = {}
    
    for athlete, apparatus_data in ALL_MeanVar.items():
        for apparatus, mean_var in apparatus_data.items():
            mean, variance = mean_var
            std_dev = math.sqrt(variance)
    
            # Calculating the confidence range
            lower_bound = mean - z_score * std_dev
            upper_bound = mean + z_score * std_dev
    
            if athlete not in ALL_Range:
                ALL_Range[athlete] = {}
            ALL_Range[athlete][apparatus] = (lower_bound, upper_bound)


    A_ALL_Range = ReOrg_to_apparatus(ALL_Range)
    
    # with open(f'{Gender}_ALL_Score_Test.txt', "w") as fb:
    #     fb.write(json.dumps(A_ALL_Range, indent=4))
        
    A_ALL_MeanVar = ReOrg_to_apparatus(ALL_MeanVar)
    
    
    # Sorting athletes in each apparatus based on mean score
    sorted_data = {}
    for apparatus, athletes in A_ALL_MeanVar.items():
        sorted_data[apparatus] = dict(sorted(athletes.items(), key=lambda item: item[1][0], reverse=True))
    
    # Plotting the data for each apparatus
    palette = "Blues_d"
    # Plotting the data for each apparatus with the new color scheme
    if Gender == 'w':
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    elif Gender == 'm':
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Athlete Performance by Apparatus', fontsize=16)
    
    apparatuses = list(sorted_data.keys())
    for i, apparatus in enumerate(apparatuses):
        ax = axes.flatten()[i]
        sns.barplot(
            x=list(sorted_data[apparatus].keys()), 
            y=[scores[0] for scores in sorted_data[apparatus].values()], 
            ax=ax, 
            palette=palette
        )
        ax.set_title(apparatus, fontsize=14)
        ax.set_ylabel('Mean Score', fontsize=12)
        ax.set_xlabel('Athlete', fontsize=12)
        ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()
    

    def sort_and_top_five(apparatus_data, FilterTopN=5):
        # Calculate mean of the confidence interval and sort
        sorted_athletes = sorted(apparatus_data.items(), key=lambda x: sum(x[1]) / 2, reverse=True)
        # Keep only top 5
        return sorted_athletes[:FilterTopN]
    
    # Initialize a merged list to store the top athletes from each apparatus
    merged_list = []
    
    # Iterate over each apparatus and process the data
    for apparatus in Apparatus:
        apparatus_data = A_ALL_Range[apparatus]  # Retrieve data for each apparatus
        top_athletes = sort_and_top_five(apparatus_data)
        merged_list.extend(top_athletes)  # Add the top athletes to the merged list

    
    # Remove duplicates and get final list of athletes
    final_athletes = list({athlete[0]: athlete for athlete in merged_list}.values())
    
    athlete_names = [athlete[0] for athlete in final_athletes]



    # Convert the list to a format suitable for CSV (list of lists)
    athlete_names_csv = [[name] for name in athlete_names]
    # Write to CSV file
    with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_Filtered_Athlete.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Flitered Athlete List'])  # Writing header
        writer.writerows(athlete_names_csv)
    print(f'Now the Filtered {Gender} Athlete list for top {FilterTopN} in each apparatus have been created')
    
    
        
    # Assuming 'final_athletes' contains the names of the top 5 athletes across all apparatus
    top_athletes = [athlete[0] for athlete in final_athletes]
    Num = len(top_athletes)
    print(f'And the list contain {Num} athletes.')
    
    # Plotting the data with confidence intervals
    if Gender == 'w':
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    elif Gender == 'm':
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    fig.suptitle('Athlete Performance with Confidence Intervals by Apparatus', fontsize=16)
    
    for i, apparatus in enumerate(apparatuses):
        ax = axes.flatten()[i]
        # Extracting names, means, and confidence intervals
        athlete_names = list(sorted_data[apparatus].keys())
        means = [sorted_data[apparatus][name][0] for name in athlete_names]
        conf_intervals = [ALL_Range[name][apparatus] for name in athlete_names]
        lower_bounds = [mean - conf_interval[0] for mean, conf_interval in zip(means, conf_intervals)]
        upper_bounds = [conf_interval[1] - mean for mean, conf_interval in zip(means, conf_intervals)]
    
        # Looping over each athlete to plot
        for athlete_name, mean, lower_bound, upper_bound in zip(athlete_names, means, lower_bounds, upper_bounds):
            color = 'orange' if athlete_name in top_athletes else 'lightblue'
            ax.errorbar(athlete_name, mean, yerr=[[lower_bound], [upper_bound]], fmt='o', ecolor=color, elinewidth=5, capsize=4, color=color)
    
        ax.set_title(apparatus, fontsize=14)
        ax.set_ylabel('Mean Score', fontsize=12)
        ax.set_xlabel('Athlete', fontsize=12)
        ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()