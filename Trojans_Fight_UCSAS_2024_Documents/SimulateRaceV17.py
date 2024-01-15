'''
While running indenpendtly, these code enables the random generation of a five-member team from the dataset located in 
'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_Filtered_Athlete.csv', and do simulation multiple times.

Additionally, the code supports the 'Gene' component of our system. By importing data from this file, 
the genetic algorithm can effectively calculate the fitness score for each generated team. 
This score is crucial for evaluating team performance and is a key factor in the algorithm's selection and optimization processes.

While we have estimated score distributions for each athlete in each apparatus, and the US team is generated randomly, 
a major unresolved issue is determining the teams from other countries and their participating athletes. 

Due to limited information about the upcoming Paris Olympics, we rely on data from the last Olympics in Tokyo, 
assuming a relatively stable competitor landscape. Noting rule differences between the Tokyo and Paris Olympics, 
we adapt our approach for teams with more than five athletes, 
selecting the best team composition based on their ability to participate in all events and optimizing for the best overall scores. 
Similarly, for countries with fewer than five athletes, we explore potential additions to optimize the team. 
This process results in fixed and optimized five-person teams from 11 countries.

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

'''

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import warnings
import matplotlib.pyplot as plt
import os
from spyder_kernels.utils.iofuncs import load_dictionary
from random import randint

import time
from joblib import load

import statistics

from concurrent.futures import ProcessPoolExecutor

#%%
def GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get):

    B_kde_dict_all_athlete = necessary_data['B_kde_dict_all_athlete']
    KNNK_athletes_dict = necessary_data['KNNK_athletes_dict']
 
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

    MuStd_List = necessary_data['MuStd_List'] 
    
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
def GetScore(athlete, apparatus, Score_Stack, necessary_data, Num_Of_Score_To_Get):
    
    try:
        Score = Score_Stack[athlete][apparatus].pop()
    except:
        Individual_Scores = GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get)
        Score_Stack[athlete][apparatus] = Individual_Scores
        Score = Score_Stack[athlete][apparatus].pop()
        #print('More Simulate for', athlete, 'in', apparatus)
    
    return Score, Score_Stack


#%%
def gaussian_distribution(x, mu, std):
     return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
   
def PlotKDEQuartic_Modified(athlete, apparatus, necessary_data, Score1, Score2, Score3, Score):
    B_kde_dict_all_athlete = necessary_data['B_kde_dict_all_athlete']
    KNNK_athletes_dict = necessary_data['KNNK_athletes_dict']

    kde = B_kde_dict_all_athlete[athlete][apparatus]
    
    plt.figure(figsize=(8, 6))
    #sns.distplot(data_for_kde, bins=20, kde=False, norm_hist=True)

    # Adjust legend to include the mean and std lines
    x_grid = np.linspace(10, 16, 1000)
    y_vals = kde.evaluate(x_grid)
    plt.plot(x_grid, y_vals, label=f'KDE - {athlete}', color = 'g', linewidth=3)

    
    NearAthletes = KNNK_athletes_dict[athlete][apparatus]
    
    for other_athlete in NearAthletes:
        kde = B_kde_dict_all_athlete[other_athlete][apparatus]
        y_vals = kde.evaluate(x_grid)
        plt.plot(x_grid, y_vals, label=f'KDE - {other_athlete}', color = 'y', linewidth=3)
            
        
    [mu, std] = MuStd_List[athlete][apparatus]
    if std < 0:
        std = - std
    x_vals = np.linspace(10, 16, 400)
    y_vals = gaussian_distribution(x_vals, mu, std)
    plt.plot(x_vals, y_vals, label="Fitted Gaussian", color = 'r', linewidth=3)
    
    # Add vertical lines
    plt.axvline(x=Score1, color='g', linestyle='--', linewidth=1.5) 
    plt.axvline(x=Score2, color='y', linestyle='--', linewidth=1.5)
    plt.axvline(x=Score3, color='r', linestyle='--', linewidth=1.5)
    
    plt.axvline(x=Score, color='b', linestyle='--', linewidth=2)

    plt.title(f'Score Distribution for {athlete} in {apparatus} with its KNN neighbours')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.xlim(10, 16)
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
    df_tokyo = df_tokyo[df_tokyo["Gender"] == Gender]
    
    df_US = df3.groupby(["New_LN_FN","Gender","Country","Apparatus"])["Score"].mean().reset_index()
    df_US = df_US[df_US["Country"] == "USA"]
    US_Team = df_US[df_US["Gender"] == Gender]
    
    Opponent_Country_List = ["ROC","CHN","FRA","BEL","GBR","ITA","JPN","GER","CAN","NED","ESP"]
   
    df_all = df3.groupby(["New_LN_FN","Gender","Country","Apparatus"])["Score"].mean().reset_index()
    df_all = df_all[df_all["Country"] != "USA"]
    df_all = df_all[df_all["Gender"] == Gender]
    
    return US_Team, Opponent_Country_List, df_all, df_tokyo 

#%%
def get_athlete_list_for_match(Opponent_Country_List, All_Team_Info, df_tokyo, Gender):
    Other_Teams = pd.DataFrame()
    if Gender == 'w':
        apparatus_required = ['BB', 'FX', 'UE', 'VT']
    else:
        apparatus_required = ['PH', 'FX', 'HB', 'PB', 'SR', 'VT']
        
    
    #%%
    for a in Opponent_Country_List:
        break_flag = 0
        team_all_max = 0
        team_name = pd.DataFrame()
        # Filtering athletes by country
        df_country = All_Team_Info[All_Team_Info["Country"].isin([a])]
        Athlete_in_TokyoOlympic = df_tokyo[df_tokyo["Country"].isin([a])]

        # Getting unique names
        unique_names = df_country['New_LN_FN'].drop_duplicates()

        # Getting athletes who participated in Tokyo Olympic
        Athlete_List_in_TokyoOlympic = Athlete_in_TokyoOlympic['New_LN_FN'].drop_duplicates()
        
        # print(len(Athlete_List_in_TokyoOlympic))
        if len(Athlete_List_in_TokyoOlympic) == 5:
            sampled_names = Athlete_List_in_TokyoOlympic 
            selected_athletes = df_country[df_country['New_LN_FN'].isin(sampled_names)]
            team_name = selected_athletes 
        else:
            #Now Generate a 5 people team for a specific counrty if we have more or less athlete participated in Tokyo Olympic
            for j in range(100):
                if break_flag == 1:
                    break
                
                # Adjust the Athlete List to have exactly 5 athletes
                if len(Athlete_List_in_TokyoOlympic) == 4:
                    # If less than 5, randomly add athletes to make it 5
                    additional_athletes_candidate = unique_names[~unique_names.isin(Athlete_List_in_TokyoOlympic)]
                    if j < len(additional_athletes_candidate):
                        additional_athletes = additional_athletes_candidate.iloc[j]
                        sampled_names = pd.concat([Athlete_List_in_TokyoOlympic, pd.Series([additional_athletes])])
                    else:
                        break_flag = 1
                        
                elif len(Athlete_List_in_TokyoOlympic) == 6:
                    # If more than 5, randomly remove athletes to make it 5
                    if j < 6:
                        sampled_names = Athlete_List_in_TokyoOlympic.drop(Athlete_List_in_TokyoOlympic.index[j])
                    else:
                        break_flag = 1
                        
                elif len(Athlete_List_in_TokyoOlympic) > 6:
                    sampled_names = Athlete_List_in_TokyoOlympic.sample(n=5, replace=False)
                    
                elif len(Athlete_List_in_TokyoOlympic) == 0:
                    sampled_names = unique_names.sample(n=5, replace=False)
                    
                elif len(Athlete_List_in_TokyoOlympic) < 4:
                    additional_athletes_candidate = unique_names[~unique_names.isin(Athlete_List_in_TokyoOlympic)]
                    additional_athletes = unique_names.sample(n= (5 - len(Athlete_List_in_TokyoOlympic)), replace=False)
                    sampled_names = pd.concat([Athlete_List_in_TokyoOlympic, pd.Series([additional_athletes])])
                    
                else:
                    print('The athlete in tokyo oylmipc for this country is wired')
   

                selected_athletes = df_country[df_country['New_LN_FN'].isin(sampled_names)]
            
                #Ensure that there will be at least 3 scores on each apparatus  
                if not all(selected_athletes.groupby("Apparatus").size().reindex(apparatus_required, fill_value=0) >= 3):
                    pass
                else:
                #The real competition would be 5 to 4, picking the top 3 high scores, but in the simulation we could simply pick the top 3
                    sorted_athletes = selected_athletes.sort_values(by=["Apparatus", "Score"], ascending=[True, False])
                    top_3_scores_sum = sorted_athletes.groupby("Apparatus").head(3).groupby("Apparatus")["Score"].sum()
                    team_all_scores = top_3_scores_sum.sum()
                
                    #Update best team if needed
                    if team_all_scores > team_all_max:
                        team_all_max = team_all_scores
                        team_name = selected_athletes.copy()
                    
        Other_Teams = pd.concat([Other_Teams,team_name],axis = 0)
        # print(a)
        # print(len(Athlete_List_in_TokyoOlympic), end = '')
        # print('--', len(unique_names))

    #Get the list of participants from other 11 countries base on tokyo olympic list and basic optimization.
    return Other_Teams

#%% Simulate match
def Simulate_match_1(df_US_ath, necessary_data, match_times, df_tokyo, Other_Teams, parallel_computing = 1):
    #Ranking by individual competitions
    #Ranking by different apparatus
    #Randomly select 5 US athletes to participate in different apparatus, and output the rankings of these five athletes in each competition.
    #constraints： the top eight can not be more than two people from the same country
    
    #team all around Logical organization
    #Team all around Logic organizer: take 4 people from the given 12 countries, then the fifth person takes the one with the highest average score among the remaining candidates (tentative).
    #The formation of five people follows a process (the selection of countries other than the US should be fixed)
    
    
    #The current process is to randomly select five people
    #Conditions: five people must have at least four scores in different programs
    #The first three scores are selected and summed up, and the scores of the four events are added together to find the total.
    #In the final round, the first three scores are taken (fixed) and summed up.
    
    if parallel_computing == 1:
        total_score_all = []
        if match_times <= 100:
            Pool_Num = 4
        elif match_times <= 400:
            Pool_Num = 8
        elif match_times <= 800:
            Pool_Num = 12
        else:
            Pool_Num = 16
        
        part_match_times = match_times // Pool_Num
        
        Num_Of_Score_To_Get = part_match_times * 5
        # ProcessPoolExecutor 
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Create a task list
            futures = [executor.submit(SumilateCore, part_match_times, df_tokyo, Other_Teams, \
                                    Num_Of_Score_To_Get, necessary_data, df_US_ath) \
                    for _ in range(Pool_Num)]
            # Get results
            data_all = []
            for future in futures:
                data_all.extend(future.result())
            
            total_score_all = [data_all[i] for i in range(0, len(data_all), 2)]
            total_score_all = [item for sublist in total_score_all for item in sublist]
            
            toal_match_details_all = [data_all[i] for i in range(1, len(data_all), 2)]
    else:
        Num_Of_Score_To_Get = round(match_times / 1.75)
        total_score_all, total_match_details_all = SumilateCore(match_times, df_tokyo, Other_Teams, \
                                    Num_Of_Score_To_Get, necessary_data, df_US_ath)
        
    score_max = max(total_score_all)


    total_score_all_float = [float(score) for score in total_score_all]
    average_score = statistics.mean(total_score_all_float)
    variance_score = statistics.variance(total_score_all_float)
    
    # Print result for a team
    print("\n-->Average Score:[", round(average_score, 2), "] | Variance:[", round(variance_score, 2), ']\n')
    #print("Variance:", variance_score, '\n')

    return average_score, variance_score, total_score_all, score_max



def SumilateCore(part_match_times, df_tokyo, Other_Teams, Num_Of_Score_To_Get, necessary_data, df_US_ath):
    total_score_part = []
    total_match_detail_part = []
    
    if Other_Teams['Gender'].iloc[0] == 'w':
        Apparatus = ['BB', 'FX', 'UE', 'VT']
    elif Other_Teams['Gender'].iloc[0] == 'm':
        Apparatus = ['PH', 'FX', 'HB', 'PB', 'SR', 'VT']
    else:
        print('Something wrong with gender')
    
    
    #Construct Score_Stack to improve running efficiency
    Score_Stack = {}
    for i in range(len(Other_Teams['New_LN_FN'])):
        athlete = Other_Teams['New_LN_FN'].iloc[i]
        apparatus = Other_Teams['Apparatus'].iloc[i]
        Individual_Scores = GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get)
        if athlete not in Score_Stack:
            Score_Stack[athlete] = {}
        Score_Stack[athlete][apparatus] = Individual_Scores
       
    for i in range(len( df_US_ath['New_LN_FN'])):
        athlete = df_US_ath['New_LN_FN'].iloc[i]
        apparatus = df_US_ath['Apparatus'].iloc[i]
        Individual_Scores = GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get)
        if athlete not in Score_Stack:
            Score_Stack[athlete] = {}
        Score_Stack[athlete][apparatus] = Individual_Scores
   
    for i in range(len( df_tokyo['New_LN_FN'])):
        athlete = df_tokyo['New_LN_FN'].iloc[i]
        apparatus = df_tokyo['Apparatus'].iloc[i]
        Individual_Scores = GetManyScore(athlete, apparatus, necessary_data, Num_Of_Score_To_Get)
        if athlete not in Score_Stack:
            Score_Stack[athlete] = {}
        Score_Stack[athlete][apparatus] = Individual_Scores
       
        
    for RaceRound in range(part_match_times):
   
        index = Apparatus + ["individual all_around", "Team all around"]
        data = {
                "gold": [0] * len(index),
                "silver": [0] * len(index),
                "bronze": [0] * len(index)
            }
        df_medals = pd.DataFrame(data, index=index)

        for apparatu in Apparatus:
        #individual qualifying round 
            df_US = df_US_ath[df_US_ath["Apparatus"] == apparatu]
            df_OtherCountry = df_tokyo[df_tokyo["Apparatus"] == apparatu]   
            df_qualifying_round = pd.concat([df_US, df_OtherCountry], axis=0)  

            df_qualifying_round, Score_Stack = UpdateScore(Num_Of_Score_To_Get, necessary_data, Score_Stack, df_qualifying_round)
                
            df_qualifying_round = df_qualifying_round.sort_values(by = "Score",ascending = False)
        
            #If there are more than two people from a country in the top eight, then the third place athlete cannot join the finals
            top_8 = pd.DataFrame()
            skipped_indices = set()
            country_counts = {}
    
            for index, row in df_qualifying_round.iterrows():
                if index in skipped_indices:
                    continue
            
                country = row['Country']
                if country_counts.get(country, 0) < 2:
                    top_8 = top_8.append(row)
                    country_counts[country] = country_counts.get(country, 0) + 1
                    skipped_indices.add(index)
            
                if len(top_8) == 8:
                    break              
        #individual final round 
            top_8_names = top_8["New_LN_FN"].tolist()

            matching_rows = df_qualifying_round[df_qualifying_round["New_LN_FN"].isin(top_8_names)]
            
            result_df = pd.DataFrame(matching_rows)
            result_df, Score_Stack = UpdateScore(Num_Of_Score_To_Get, necessary_data, Score_Stack, result_df)
                  
            result_df = result_df.sort_values(by = "Score", ascending = False).head(3)
            
            df_medals = result_to_medal(result_df, apparatu, df_medals)
    
        #individual all-round qualification
        df_athletes = pd.concat([df_US_ath, df_tokyo], axis=0)

        #Filtering for the athlete who can participate in every apparatus.
        qualified_athletes = df_athletes.groupby('New_LN_FN').filter(lambda x: len(x['Apparatus'].unique()) == len(Apparatus)) 
        qualified_athletes = qualified_athletes[qualified_athletes['Apparatus'].isin(Apparatus)]
        
        qualified_athletes, Score_Stack = UpdateScore(Num_Of_Score_To_Get, necessary_data, Score_Stack, qualified_athletes)
        athlete_scores = qualified_athletes.groupby(['New_LN_FN', 'Country'])['Score'].sum().reset_index()
        athlete_scores = athlete_scores.rename(columns={'Score': 'Total_Score'})
        athlete_scores = athlete_scores.sort_values(by = "Total_Score",ascending = False)
        
        #No more than 2 finalists from one country.
        top_24 = pd.DataFrame()
        skipped_indices = set()
        country_counts = {}
        
        for index, row in athlete_scores.iterrows():
            if index in skipped_indices:
                continue
            
            country = row['Country']
            if country_counts.get(country, 0) < 2:
                top_24 = top_24.append(row)
                country_counts[country] = country_counts.get(country, 0) + 1
                skipped_indices.add(index)
            
            if len(top_24) == 24:
                break
    
        #individual all-round final
        top_24_names = top_24["New_LN_FN"].tolist()

        matching_rows = athlete_scores[athlete_scores["New_LN_FN"].isin(top_24_names)]
        result_df = pd.DataFrame(matching_rows)
        
        for i in range(len(result_df)):
            athlete = result_df['New_LN_FN'].iloc[i]
            apparatus = apparatu
            Full_Score = 0
            for apparatus in Apparatus:
                Score, Score_Stack = GetScore(athlete, apparatus, Score_Stack, necessary_data, Num_Of_Score_To_Get)
                Full_Score += Score
            result_df['Total_Score'].iloc[i] = Full_Score
           
        result_df = result_df.sort_values(by = "Total_Score", ascending = False).head(3)
    
        df_medals = result_to_medal(result_df, "individual all_around", df_medals)

        #Group all-round qualifications
        Other_teams = ["ROC","USA","CHN","FRA","BEL","GBR","ITA","JPN","GER","CAN","NED","ESP"]
        
        team_all = pd.concat([Other_Teams,df_US_ath], axis = 0)
        df_scores = pd.DataFrame(Other_teams, columns=['Country'])
        df_scores['Team all around score'] = 0

        for country in Other_teams:
            df_country = team_all[team_all["Country"] == country]

            df_country, Score_Stack = UpdateScore(Num_Of_Score_To_Get, necessary_data, Score_Stack, df_country)
                
            df_sorted = df_country.sort_values(by=['Apparatus', 'Score'], ascending=[True, False])
            top_3_scores = df_sorted.groupby('Apparatus').head(3) #Get top 3 scores from each apparatus
            total_score = top_3_scores['Score'].sum() #Sum up

            df_scores.loc[df_scores['Country'] == country, 'Team all around score'] = total_score

        df_scores = df_scores.sort_values("Team all around score", ascending = False).head(8)
    
        #Group all-round qualifications final
        tokyo_team_final = list(df_scores["Country"])
        df_scores = pd.DataFrame(tokyo_team_final, columns=['Country'])
        df_scores['Team all around score'] = 0


        for country in tokyo_team_final:
            df_country = team_all[team_all["Country"] == country]

            df_country, Score_Stack = UpdateScore(Num_Of_Score_To_Get, necessary_data, Score_Stack, df_country)
                
            df_sorted = df_country.sort_values(by=['Apparatus', 'Score'], ascending=[True, False])
            top_3_scores = df_sorted.groupby('Apparatus').head(3) #Get top 3 scores from each apparatus
            total_score = top_3_scores['Score'].sum() #Sum up

            df_scores.loc[df_scores['Country'] == country, 'Team all around score'] = total_score
 
        df_scores = df_scores.sort_values("Team all around score", ascending = False).head(3)
    
        df_scores["Rank"] = range(1,4)
       
        for index,row in df_scores.iterrows():
            if row["Country"] == "USA":
                rank = row["Rank"]
                if rank == 1:
                    df_medals.at["Team all around", "gold"] = 1
                elif rank == 2:
                    df_medals.at["Team all around", "silver"] = 1
                elif rank == 3:
                    df_medals.at["Team all around", "bronze"] = 1
        
        gold_count = (df_medals['gold'] != 0).sum()
        silver_count = (df_medals['silver'] != 0).sum()
        bronze_count = (df_medals['bronze'] != 0).sum()

        total_score = gold_count * 3 + silver_count * 2 + bronze_count

        total_score_part.append(total_score)
        total_match_detail_part.append(df_medals)
        
    return total_score_part, total_match_detail_part


def result_to_medal(result_df, row_name, df_medals):

    # Assign gold medal (first position)
    if result_df.iloc[0]["Country"] == "USA":
        df_medals.at[row_name, "gold"] = result_df.iloc[0]['New_LN_FN']

    # Assign silver medal (second position)
    if result_df.iloc[1]["Country"] == "USA":
        df_medals.at[row_name, "silver"] = result_df.iloc[1]['New_LN_FN']

    # Assign bronze medal (third position)
    if result_df.iloc[2]["Country"] == "USA":
        df_medals.at[row_name, "bronze"] = result_df.iloc[2]['New_LN_FN']

    return df_medals

def UpdateScore(Num_Of_Score_To_Get, necessary_data, Score_Stack, result_df):
    updated_scores = []
    
    for i in range(len(result_df)):
        athlete = result_df['New_LN_FN'].iloc[i]
        apparatus = result_df['Apparatus'].iloc[i]
        
        score, Score_Stack = GetScore(athlete, apparatus, Score_Stack, necessary_data, Num_Of_Score_To_Get)
        updated_scores.append(score)

    result_df['Score'] = updated_scores
    return result_df, Score_Stack
    
    #%% Print Outcome
def PrintOutCome(total_score_all):
    plt.bar(range(len(total_score_all)), total_score_all)
    plt.title('Total Scores Over Time')  # Add a title to the plot
    plt.xlabel('Attempt Number')  # Label for the x-axis
    plt.ylabel('Total Score')  # Label for the y-axis
    plt.ylim(0, 25)
    
    # Calculate the average score and variance
    avg_score = np.mean(total_score_all)
    variance = np.var(total_score_all)
    # Add a horizontal line for the average score
    plt.axhline(y=avg_score, color='orange', linestyle='-', label=f'Average Score: {avg_score:.2f}')
    # Display the variance as text on the plot
    plt.plot([], [], ' ', label=f'Variance: {variance:.3f}')

    plt.legend()
    plt.show()
    return 0
    

def GetUSSample(US_Team, Apparatus):
     unique_names = US_Team['New_LN_FN'].drop_duplicates()
     sampled_names = unique_names.sample(n=5) 
     df_US_ath = US_Team[US_Team['New_LN_FN'].isin(sampled_names)]
 
 #Ensure that there will be at least 3 scores on each apparatus to participate in 'team all-around'
     while not all(df_US_ath.groupby("Apparatus").size().reindex(Apparatus, fill_value=0) >= 3):
        sampled_names = unique_names.sample(n=5, replace=False)
        df_US_ath = US_Team[US_Team['New_LN_FN'].isin(sampled_names)]
        
     sampled_names.tolist()
     return sampled_names, df_US_ath
 

#%% Main Test Code
if __name__ == '__main__':

    start_time0 = time.time()#00
    
    Gender = 'm' # 'w'
    Use_Specific_Sample = 0
    Plot_Flag = 0 # Whether Plot Distribution and Score
    Use_Filtered_Athlete = 1
    match_times = 1024

    parallel_computing = 1 #If 0, do not use pool
    
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
    
    if Gender == 'w':
        Apparatus = ['BB', 'FX', 'UE', 'VT']
    else:
        Apparatus = ['PH', 'FX', 'HB', 'PB', 'SR', 'VT']
        
    if Use_Specific_Sample == 1:
        df_US_ath = pd.read_csv(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_AthleteSample_df_US_ath.csv')
        sampled_names = np.array(df_US_ath['New_LN_FN'].unique())
    else:
        if Use_Filtered_Athlete == 1:
            # Simpler version to read names from a CSV file
            with open(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_Filtered_Athlete.csv', 'r') as file:
                final_athletes = [line.strip() for line in file]
            
            athlete_names = final_athletes[1:]
            # Filter the DataFrame
            filtered_df = US_Team[US_Team['New_LN_FN'].isin(athlete_names)]
            # Check and report athletes not in US_Team
            athletes_not_in_df = [name for name in athlete_names if name not in filtered_df['New_LN_FN'].unique()]
            # Reporting
            if athletes_not_in_df:
                print("Athletes not found in US_Team:", athletes_not_in_df, '\n')
            else:
                print("All final athletes are present in US_Team.\n")
            sampled_names, df_US_ath = GetUSSample(filtered_df, Apparatus)
        else:
            sampled_names, df_US_ath = GetUSSample(US_Team)
        #df_US_ath.to_csv(f'Cleaned&ReOrganized_Data/ReOrganized_Data_{Gender}/{Gender}_AthleteSample_df_US_ath.csv', index=False)
    print(sampled_names)
    print('--------------------------------------')
  
    
    average_score, variance_score, total_score_all, score_max = \
        Simulate_match_1(df_US_ath, necessary_data, match_times, df_tokyo, Other_Teams, parallel_computing)
    
    print('--------------------------------------')
    
    print('All time used:', time.time() - start_time0)
    
    print('--------------------------------------')
    PrintOutCome(total_score_all)