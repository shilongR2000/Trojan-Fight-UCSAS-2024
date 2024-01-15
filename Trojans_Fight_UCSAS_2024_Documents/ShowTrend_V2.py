import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import re


# Set up the tkinter file dialog without the root window appearing
tk.Tk().withdraw()

# Open the file dialog to choose the file
file_path = askopenfilename(title="Select the CSV file", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected.")
else:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

# Regular expression to match everything up to "Training:"
match = re.search(r'^(.*?)_Training', file_path)

# Extract the matched string
if match:
    Gender = match.group(1)[-1]
    print("Gender:", Gender)
else:
    print("Pattern not found")
    
if Gender == 'm':
    Gender = 'Men'
else:
    Gender = 'Women'


population = max(df['Order'])
top_teams_df = df.groupby('Gen', group_keys=False).apply(lambda x: x.nlargest(int(population/6), 'avg_score'))

df = top_teams_df
# Convert the string representation of the list to actual lists
df['team_list'] = df['team_list'].apply(ast.literal_eval)

# Create a counter for each generation to count occurrences of each athlete
generations = df['Gen'].unique()
generation_frequencies = {gen: Counter() for gen in generations}

# Fill the counters with the athletes' occurrences
for gen in generations:
    gen_athletes = df[df['Gen'] == gen]['team_list'].tolist()
    generation_frequencies[gen].update(sum(gen_athletes, []))

# Convert the counters into a DataFrame for plotting per generation frequencies
gen_frequency_df = pd.DataFrame(generation_frequencies).fillna(0).T

# Count the total occurrences of each athlete across all generations
total_frequencies = Counter(sum(df['team_list'].tolist(), []))

# Convert the total frequencies into a DataFrame for plotting
total_frequency_df = pd.DataFrame(total_frequencies.items(), columns=['Athlete', 'Frequency'])

# Sort the DataFrame by frequency for the total count
total_frequency_df_sorted = total_frequency_df.sort_values(by='Frequency', ascending=True)

fig = plt.figure(figsize=(9, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

# First subplot (larger)
ax1 = fig.add_subplot(gs[0])
gen_frequency_df.plot(kind='bar', ax=ax1, stacked=True)
ax1.set_title(f'Frequency of Each Athlete by Generation for {Gender}')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Frequency')
# Reverse the order of the legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1[::-1], labels1[::-1], title='Athlete', bbox_to_anchor=(1.05, 1), loc='upper left')

# Second subplot (smaller)
ax2 = fig.add_subplot(gs[1])
total_frequency_df_sorted.plot(kind='barh', x='Athlete', y='Frequency', ax=ax2, color='skyblue')
ax2.set_title(f'Total Frequency of Each Athlete Across All Generations  for {Gender}')
ax2.set_xlabel('Total Frequency')
ax2.set_ylabel('Athlete')

plt.tight_layout()
plt.show()


#%%
top_teams_df = df.groupby('Gen', group_keys=False).apply(lambda x: x.nlargest(int(population/2), 'avg_score'))

# Calculate average, variance, and maximum for each generation
generations = range(max(top_teams_df['Gen']) + 1)
avg_scores = []
var_scores = []
max_scores = []

for i in generations:
    ith_teams = top_teams_df[top_teams_df['Gen'] == i]
    ith_avg = np.mean(ith_teams['avg_score']) if not ith_teams.empty else 0
    ith_var = np.mean(ith_teams['variance']) if not ith_teams.empty else 0
    ith_max = max(ith_teams['avg_score']) if not ith_teams.empty else 0
    avg_scores.append(ith_avg)
    var_scores.append(ith_var)
    max_scores.append(ith_max)

# Plotting the results
plt.figure(figsize=(9, 6))
plt.plot(generations, avg_scores, label='Average Score', linewidth=2)
plt.plot(generations, var_scores, label='Variance in Score', linewidth=2)
plt.plot(generations, max_scores, label='Maximum Score', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('Score')
plt.title(f'Performance Metrics by Generation for {Gender}')
plt.tight_layout()
plt.legend()
plt.show()


    
    

