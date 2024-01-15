'''
The code is designed to extract data from the Tokyo Olympics All-Around competition's record PDFs, 
covering both qualification and final rounds. It then compiles and formats this data into a CSV file, 
aligning with the standard format provided by UCSAS. This ensures consistency and usability of the data for further analysis and modeling.

-------------------------------------------
Required Files:
mag_final_aa_results.pdf
mag_quals_aa.pdf
(all from https://gymnasticsresults.com/results/2021/olympics/)
(Should be put in a single folder)

-------------------------------------------
Output Files:
AllRound_Data.csv
'''
import PyPDF2
import os
import csv

import tkinter as tk
from tkinter import filedialog
#%%
Choose_Mode = 1

#%%
# Create a root window but keep it hidden
root = tk.Tk()
root.withdraw()

script_dir = os.path.dirname(os.path.realpath(__file__))

if Choose_Mode == 1:
    # Set up options for the directory dialog
    dir_options = {
        'title': 'Choose a Directory',
        'initialdir': script_dir  # Set the initial directory to the script's directory
    }
    # Open the directory dialog and store the selected folder path
    pdf_folder = filedialog.askdirectory(**dir_options)
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print('We have', len(pdf_files),'pdf files in the folder')
else:
    # Set up options for the file dialog
    file_options = {
        'title': 'Choose a PDF file',
        'filetypes': [('PDF files', '*.pdf')],
        'initialdir': script_dir  
    }
    
    # Open the file dialog and store the selected file path
    pdf_files = filedialog.askopenfilename(**file_options)
    pdf_files = [pdf_files]


pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
print(len(pdf_files))

#%% Start crawl

list_of_dicts = []
for file in pdf_files:
    pdf_path = os.path.join(pdf_folder, file)
    with open(pdf_path, 'rb') as pdfFileObj:

        pdfReader = PyPDF2.PdfReader(pdfFileObj)

        #print(len(pdfReader.pages))
        
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]

            
            entire_text = pageObj.extract_text()
            #print(entire_text)
            lines = entire_text.splitlines()

            splitted = []
            temp_list = []
            count = 0
            
            for line in lines:
                count += 1
                
                #print(line)
                if not((line[0].isdigit() and line[2].isalpha()) or (line[0].isdigit() and line[1].isdigit() and line[3].isalpha())):
                    temp_list.append(line)
                else:
                    #print(temp_list)
                    splitted.append(temp_list)
                    #print(splitted)
                    temp_list = []
                    temp_list.append(line)

                if len(lines) == count:
                    splitted.append(temp_list)


            #print(splitted)

            header = splitted[0]
            date = header[3]
            
            #hardcoded competition and location variable
            competition = "Olympic Games"
            location = "Tokyo, Japan"
            gender_full = header[5].split("'")[0].strip()
            if gender_full == "Women":
                gender = "w"
            else:
                gender = "m"
            if "Qualification" in header[7].strip():
                round = "qual"
            else:
                round = "final"

            #print(splitted[24])
            for i in range(1, len(splitted)):
                #print(splitted[i][0])
                name_line = splitted[i][0].split()
                last_name = name_line[1]
                #print(last_name)
                first_name = name_line[2]
                country = name_line[-3]
                
                fx_score = splitted[i][1].split()[1]
                fx_d_score = splitted[i][2]
                fx_e_score = splitted[i][3]
                adder = 0
                if "(" not in splitted[i][4]:
                    fx_penalty = splitted[i][4]
                    fx_rank = splitted[i][5 + adder]
                    adder += 1
                else:
                    fx_penalty = 0
                    fx_rank = splitted[i][4 + adder]
                
                ph_score = splitted[i][5 + adder]
                ph_d_score = splitted[i][6 + adder]
                ph_e_score = splitted[i][7 + adder]
                if "(" not in splitted[i][8 + adder]:
                    ph_penalty = splitted[i][8 + adder]
                    ph_rank = splitted[i][9 + adder]
                    adder += 1
                else:
                    ph_penalty = 0
                    ph_rank = splitted[i][8 + adder]
                
                sr_score = splitted[i][9 + adder]
                sr_d_score = splitted[i][10 + adder]
                sr_e_score = splitted[i][11 + adder]
                if "(" not in splitted[i][12 + adder]:
                    sr_penalty = splitted[i][12 + adder]
                    sr_rank = splitted[i][13 + adder]
                    adder += 1
                else:
                    sr_penalty = 0
                    sr_rank = splitted[i][12 + adder]
                
                vt_score = splitted[i][13 + adder]
                vt_d_score = splitted[i][14 + adder]
                vt_e_score = splitted[i][15 + adder]
                if "(" not in splitted[i][16 + adder]:
                    vt_penalty = splitted[i][16 + adder]
                    vt_rank = splitted[i][17 + adder]
                    adder += 1
                else:
                    vt_penalty = 0
                    vt_rank = splitted[i][16 + adder]
                
                pb_score = splitted[i][17 + adder]
                pb_d_score = splitted[i][18 + adder]
                pb_e_score = splitted[i][19 + adder]
                if "(" not in splitted[i][20 + adder]:
                    pb_penalty = splitted[i][20 + adder]
                    pb_rank = splitted[i][21 + adder]
                    adder += 1
                else:
                    pb_penalty = 0
                    pb_rank = splitted[i][20 + adder]
                
                hb_score = splitted[i][21 + adder]
                hb_d_score = splitted[i][22 + adder]
                hb_e_score = splitted[i][23 + adder]
                if "(" not in splitted[i][24 + adder]:
                    hb_penalty = splitted[i][24 + adder]
                    hb_rank = splitted[i][25 + adder]
                    adder += 1
                else:
                    hb_penalty = 0
                    hb_rank = splitted[i][24 + adder]
                
                
                
                fx_dict = {
                    "LastName": last_name, 
                    "FirstName": first_name, 
                    "Gender": gender, 
                    "Country": country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": "FX",
                    "Rank": fx_rank,
                    "D_Score": fx_d_score,
                    "E_Score": fx_e_score,
                    "Penalty": fx_penalty,
                    "Score": fx_score
                    }
                
                ph_dict = { 
                    "LastName": last_name, 
                    "FirstName": first_name, 
                    "Gender": gender, 
                    "Country": country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": "PH",
                    "Rank": ph_rank,
                    "D_Score": ph_d_score,
                    "E_Score": ph_e_score,
                    "Penalty": ph_penalty,
                    "Score": ph_score
                    }
                
                sr_dict = {
                    "LastName": last_name, 
                    "FirstName": first_name, 
                    "Gender": gender, 
                    "Country": country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": "SR",
                    "Rank": sr_rank,
                    "D_Score": sr_d_score,
                    "E_Score": sr_e_score,
                    "Penalty": sr_penalty,
                    "Score": sr_score
                    }
                
                vt_dict = {
                    "LastName": last_name, 
                    "FirstName": first_name, 
                    "Gender": gender,   
                    "Country": country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": "VT",
                    "Rank": vt_rank,
                    "D_Score": vt_d_score,
                    "E_Score": vt_e_score,
                    "Penalty": vt_penalty,
                    "Score": vt_score
                    }

                pb_dict = {
                    "LastName": last_name, 
                    "FirstName": first_name, 
                    "Gender": gender, 
                    "Country": country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": "PB",
                    "Rank": pb_rank,
                    "D_Score": pb_d_score,
                    "E_Score": pb_e_score,
                    "Penalty": pb_penalty,
                    "Score": pb_score
                    }
                
                hb_dict = {
                    "LastName": last_name, 
                    "FirstName": first_name, 
                    "Gender": gender, 
                    "Country": country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": "HB",
                    "Rank": hb_rank,
                    "D_Score": hb_d_score,
                    "E_Score": hb_e_score,
                    "Penalty": hb_penalty,
                    "Score": hb_score
                    }
                
            
                list_of_dicts.append(fx_dict)
                list_of_dicts.append(ph_dict)
                list_of_dicts.append(sr_dict)
                list_of_dicts.append(vt_dict)
                list_of_dicts.append(pb_dict)
                list_of_dicts.append(hb_dict)


#%% 
for i in range(len(list_of_dicts)):
    rank_str = list_of_dicts[i]['Rank']
    # Removing non-numeric characters
    rank_str = rank_str.strip("()")
    # Converting to an integer
    rank_int = int(rank_str)
    list_of_dicts[i]['Rank'] = rank_int

with open('AllRound_Data.csv', 'w', newline='') as csvfile:
    # Get the keys in the first dictionary to use as the field names
    fieldnames = list_of_dicts[0].keys()

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # write the header

    # Write the dictionaries as rows
    for data in list_of_dicts:
        writer.writerow(data)
