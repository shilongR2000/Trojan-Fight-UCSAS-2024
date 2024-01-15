'''
This code handling the vital task of data preparation. Its input is the historical performance data of various athletes, provided by UCSAS. 
This data, sourced directly from PDF files, often contains inconsistencies like duplicate rows, varying athlete name formats, and occasional zero scores. 
Our code's primary objective is to cleanse this data thoroughly, rectifying these issues to produce a cleaned and reliable version. 
This refined dataset will serve as an accurate basis for further model construction and analysis.
(From cleaning_V2_1006)
-------------------------------------------
Required Files:
data_2022_2023.csv
data_2017_2021.csv

AllRound_Data.csv
SingleItem_Data.csv

-------------------------------------------
Output Files:
Cleaned_Data_2022_2023.csv
Cleaned_Data_2017_2021(With_Supplemental_Data)


'''

import pandas as pd
import os
import warnings

def extract_before_second_space(s):
    parts = s.split(" ", 2)
    return ' '.join(parts[:2]) if len(parts) > 2 else s

def get_longest_string(series):
    return max(series, key=len)

#%% main
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    # Load data

    for step in range(4):
        if step == 0:
            ori_data = pd.read_csv("Original_Data/data_2022_2023.csv")
        elif step == 1:
            ori_data = pd.read_csv("Original_Data/data_2017_2021.csv")
        elif step == 2:
            ori_data = pd.read_csv("Supplemental_Data(File&Crawling_Code)/AllRound_Data.csv")
        elif step == 3:
            ori_data = pd.read_csv("Supplemental_Data(File&Crawling_Code)/SingleItem_Data.csv")

        # Define the conditions, Dele two unusual row
        condition = (ori_data['LastName'] == 'MARTINEZ') & (ori_data['FirstName'] == 'Victor') & (ori_data['Apparatus'] == 'VT1')
        
        # Filter the data
        ori_data = ori_data[~condition]

        ori_data['LastName'] = ori_data['LastName'].replace('ÖNDER', 'ONDER')
        ori_data['LastName'] = ori_data['LastName'].replace('PR脺GEL', 'PRUGEL')
        ori_data['LastName'] = ori_data['LastName'].replace('P脛IV脛NEN', 'PAIVANEN')
        ori_data['LastName'] = ori_data['LastName'].replace('P脡REZ FERN脕NDEZ', 'PEREZ')
        
        # Reset the index
        ori_data.reset_index(drop=True, inplace=True)

        ori_data.dropna(subset = ['Score'], inplace = True)

        missing_indices = ori_data[(ori_data['D_Score'] == 0) & (ori_data['E_Score'] == 0) & (ori_data['Score'] == 0)].index
        ori_data.drop(missing_indices, inplace = True)

        ori_data.reset_index(drop = True, inplace = True)

        # Find rows where D_Score is missing and E_Score is present
        missing_dscore_indices = ori_data[(ori_data['D_Score'].isnull()) & (ori_data['E_Score'].notnull())].index
        ori_data.loc[missing_dscore_indices, 'D_Score'] = ori_data['Score'] - ori_data['E_Score']
        
        # Find rows where E_Score is missing and D_Score is present
        missing_escore_indices = ori_data[(ori_data['E_Score'].isnull()) & (ori_data['D_Score'].notnull())].index
        ori_data.loc[missing_escore_indices, 'E_Score'] = ori_data['Score'] - ori_data['D_Score']

        # Fill missing Score using the relationship: score = escore + dscore
        missing_score_indices = ori_data[(ori_data['Score'].isnull()) & (ori_data['E_Score'].notnull()) & (ori_data['D_Score'].notnull())].index
        ori_data.loc[missing_score_indices, 'Score'] = ori_data['E_Score'] + ori_data['D_Score']
            
        # Preprocess data
        ori_data['Date'] = pd.to_datetime(ori_data['Date'], errors='coerce').dt.date
        ori_data[['FirstName', 'LastName']] = ori_data[['FirstName', 'LastName']].fillna('Missing_Data')
        ori_data['LastName'] = ori_data['LastName'].str.replace("'", " ")
        ori_data['LastName'] = ori_data['LastName'].str.replace(" ", "-")
        ori_data['LastName'] = ori_data['LastName'].str.replace("ABDUL-HADI", "ABDUL HADI")


        compare_name_data = pd.DataFrame()
        compare_name_data['New_LN_FN'] = ori_data['LastName'] + ' ' + ori_data['FirstName']
        compare_name_data['LN_FN_Extract'] =  compare_name_data['New_LN_FN'].apply(extract_before_second_space)
        compare_name_data['LN_Extract'] = ori_data['LastName'].apply(extract_before_second_space)

        compare_name_data.reset_index(drop=True, inplace=True)
        
        rows_to_delete = ori_data[(ori_data['D_Score'].isnull()) & (ori_data['Score'].isnull())].index
        # Print the rows to be deleted
        print("Rows to be deleted:", len(rows_to_delete))
        # Drop these rows
        ori_data = ori_data.drop(rows_to_delete)

        #%% Method2
        matching_groups = compare_name_data.groupby(['LN_FN_Extract', 'LN_Extract']).filter(lambda x: len(x) > 1)

        for _, group in matching_groups.groupby(['LN_FN_Extract', 'LN_Extract']):
            longest_ln_fn = get_longest_string(group['New_LN_FN'])
            compare_name_data.loc[group.index, 'New_LN_FN'] = longest_ln_fn

        ori_data['New_LN_FN'] = compare_name_data['New_LN_FN']

        #Modify some names
        ori_data.loc[ori_data['New_LN_FN'] == 'CARRERES Jan', 'New_LN_FN'] = 'CARRERES-MACIA Jan'
        ori_data.loc[ori_data['New_LN_FN'] == 'CALVO-MORENO-JO Missing_Data', 'New_LN_FN'] = 'CALVO-MORENO Jossimar Orlando'
        ori_data.loc[ori_data['New_LN_FN'] == 'CEMLYN-JONES Joe', 'New_LN_FN'] = 'CEMLYN-JONES Joseph'
        ori_data.loc[ori_data['New_LN_FN'] == 'ELPITIYA-BADALG-D Missing_Data', 'New_LN_FN'] = 'ELPITIYA-BADALGE-DONA Milka Gehani'
        ori_data.loc[ori_data['New_LN_FN'] == 'GUIMARÃES Yuri', 'New_LN_FN'] = 'GUIMARAES Yuri'
        ori_data.loc[ori_data['New_LN_FN'] == 'KUMARASINGHEGE-HG Missing_Data', 'New_LN_FN'] = 'KUMARASINGHEGE Hansha'
        ori_data.loc[ori_data['New_LN_FN'] == 'MASONSTEPHENS Clay', 'New_LN_FN'] = 'MASON-STEPHENS Clay'

        ori_data['New_LN_FN'] = ori_data['New_LN_FN'].str.replace("-", " ")
        ori_data['New_LN_FN'] = ori_data['New_LN_FN'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

    #%% Delete repeated score
        ori_data['Penalty'].fillna(0, inplace=True)
        to_drop = []

        print('Still Running', end = '')
        for i in range(len(ori_data)):
            if i%1000 == 0:
                print('.', end = '')
            current_row = ori_data.iloc[i, 3:].values
            for j in range(1, 20):
                try:
                    subset_data = ori_data.iloc[i + j, 3:].values  # Consider data from the next 20 row onwards
                    if (current_row == subset_data).all():
                        to_drop.append(i)
                        #print('Row', i ,'AS', i+j)
                        break
                except:
                    break
        
        # Drop duplicates
        ori_data.drop(index=to_drop, inplace=True)
    
        # List the columns with New_LN_FN first
        cols = ['New_LN_FN'] + [col for col in ori_data if col != 'New_LN_FN']
        
        # Reindex the DataFrame with the new column order
        ori_data = ori_data[cols]

        ori_data['New_LN_FN'] = ori_data['New_LN_FN'].str.upper()

        # Get the directory of the currently executing script
        current_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Create the full path for the CSV file
        try:
            if step == 0:
                csv_path = os.path.join(current_directory, 'Cleaned&ReOrganized_Data/Cleaned_Data_2022_2023.csv')
                # Save the DataFrame to the CSV
                ori_data.to_csv(csv_path, index=False)
                print('File is created')
            else:
                if step == 1:
                    ori_data_p1 = ori_data
                elif step == 2:
                    ori_data_p2 = ori_data
                else:
                    ori_data_p3 = ori_data
                    combined_data = pd.concat([ori_data_p1, ori_data_p2, ori_data_p3], ignore_index=True)

                    csv_path = os.path.join(current_directory, 'Cleaned&ReOrganized_Data/Cleaned_Data_2017_2021(With_Supplemental_Data).csv')
                    # Save the DataFrame to the CSV
                    combined_data.to_csv(csv_path, index=False)
                    print('File is created')
        except:
            print('File can not be modified, please close it at first')