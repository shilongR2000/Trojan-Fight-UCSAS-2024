'''
The code is designed to extract data from the Tokyo Olympics Single-Item competition's record PDFs, 
covering both qualification and final rounds. It then compiles and formats this data into a CSV file, 
aligning with the standard format provided by UCSAS. This ensures consistency and usability of the data for further analysis and modeling.

-------------------------------------------
Required Files:
mag_final_fx_results.pdf
mag_final_hb_results.pdf
mag_final_pb_results.pdf
mag_final_ph_results.pdf
mag_final_sr_results.pdf
mag_final_vt_results.pdf

mag_quals_fx.pdf
mag_quals_hb.pdf
mag_quals_pb.pdf
mag_quals_ph.pdf
mag_quals_sr.pdf
mag_quals_vt.pdf
(all from https://gymnasticsresults.com/results/2021/olympics/)
(Should be put in a single folder)

-------------------------------------------
Output Files:
SingleItem_Data.csv
'''
import PyPDF2
import os
import csv
import pandas as pd

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


#%%   
list_of_dicts = []
Data = []

for file in pdf_files:
    if Choose_Mode == 1:
        pdf_path = os.path.join(pdf_folder, file)
    else:
        pdf_path = file
    with open(pdf_path, 'rb') as pdfFileObj:

        pdfReader = PyPDF2.PdfReader(pdfFileObj)

        #print(len(pdfReader.pages))
        
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]

            
            entire_text = pageObj.extract_text()
            #print(entire_text)
            lines = entire_text.splitlines()
            
            splitted = []
            
            for line in lines:     
                #print(line)
                if (line[:4].isdigit() and line[4] == ' ' and line[5:6].isalpha()) or\
                    (line[:2].isdigit() and line[2] == ' ' and line[3:6].isdigit()\
                     and line[6] == ' ' and line[7:8].isalpha()):
                    splitted.append(line)

            #print(splitted)

            header = lines
            date = header[3]
            
            #hardcoded competition and location variable
            competition = "Olympic Games"
            location = "Tokyo, Japan"
            Gender = header[5].split("'")[0].strip()
            apparatus_full = header[5].split("'s ")[1].strip()
            
            if apparatus_full == 'Horizontal Bar':
                apparatus = 'HB'
            elif  apparatus_full == 'Pommel Horse':
                apparatus = 'PH'
            elif  apparatus_full == 'Floor Exercise':
                apparatus = 'FX'
            elif  apparatus_full == 'Rings':
                apparatus = 'SR'
            elif  apparatus_full == 'Parallel Bars':
                apparatus = 'PB'
            elif  apparatus_full == 'Vault':
                apparatus = 'VT1'
                
                splitted = [] #Inlcude two lines of data
                Keep_record = 0
                line_cache = []
                for line in lines:     
                    #print(line)
                    if Keep_record == 1:
                        splitted.append(line_cache + line)
                        Keep_record = 0
                    if (line[:4].isdigit() and line[4] == ' ' and line[5:6].isalpha()) or\
                        (line[:2].isdigit() and line[2] == ' ' and line[3:6].isdigit()\
                         and line[6] == ' ' and line[7:8].isalpha()):
                        line_cache = line
                        Keep_record = 1
            else:
                print('Undefined apparatus?')
            
            if Gender == "Women":
                Gender = "w"
            else:
                Gender = "m"
            if "Qualification" in header[7].strip():
                round = "qual"
            else:
                round = "final"
            
            split_data = [line.split() for line in splitted]
            
            print('------------')
            for i in range(len(split_data)):
                print(len(split_data[i]))
                
            for i in range(len(split_data)):
                skip_step = 0
                
                try:
                    int(split_data[i][1]) #If the second element is a number, then rank should just be first element
                    Rank = split_data[i][0]
                    del split_data[i][1]
                    print('Did it')
                except:
                    Rank = split_data[i][0][:-3] #Take only the rank number
                
              
                try: 
                    float(split_data[i][4 + skip_step]) #If can not being convert to num, then the name must be more than 2 space
                    Last_Name = split_data[i][1]
                    First_Name = split_data[i][2]
                except:
                    # Combine the 2nd and 3rd elements for the ith row
                    Last_Name = split_data[i][1] + ' ' + split_data[i][2]
                    First_Name = split_data[i][3]
                    skip_step += 1
                    
                Country = split_data[i][3 + skip_step]
                    
                if apparatus == 'VT1':
                    skip_step += 1
                    
                Score = split_data[i][4 + skip_step]
                
                
                if apparatus == 'VT1':
                    if float(split_data[i][5 + skip_step]) < 0:
                        Penalty = split_data[i][5 + skip_step]
                        skip_step += 1
                    else:
                        Penalty = 0
                else:
                    try:
                        if float(split_data[i][7 + skip_step]) < 0:
                            Penalty = split_data[i][7 + skip_step]
                        else:
                            Penalty = 0
                    except:
                        pass
                    
                    try: #For Qulification file, different format
                        if float(split_data[i][-1]) < 0:
                            Penalty = split_data[i][-1]
                        else:
                            Penalty = 0
                    except:
                        pass

                  
                try: #Adjust for potential Q
                    float(split_data[i][5 + skip_step])
                except:
                    skip_step += 1

                D_Score = split_data[i][5 + skip_step]
                
                if apparatus == 'VT1':
                    skip_step += 1
                    
                E_Score = split_data[i][6 + skip_step]
                
                #%% Now finished data crawl
                Dict = {
                    "LastName": Last_Name, 
                    "FirstName": First_Name,
                    "Gender": Gender,   
                    "Country": Country,
                    "Date": date,
                    "Competition": competition,
                    "Round": round,
                    "Location": location,
                    "Apparatus": apparatus,
                    "Rank": Rank,
                    "D_Score": D_Score,
                    "E_Score": E_Score,
                    "Penalty": Penalty,
                    "Score": Score
                    }
                
                
                df = pd.DataFrame([Dict])
                Data.append(df)
                
                if apparatus == 'VT1': #If VT, continues keep track of VT2
                    
                    try: #The VT in qulification have different format
                        split_data[i][12 + skip_step]
                    except:
                        skip_step -= 2
                        
                    try:
                        float(split_data[i][9 + skip_step])
                    except:
                        skip_step += 1
                        
                    if float(split_data[i][9 + skip_step]) > 0:
                        Score = split_data[i][9 + skip_step]
                        
                        if float(split_data[i][10 + skip_step]) < 0:
                            Penalty = split_data[i][10 + skip_step]
                            skip_step += 1
                        else:
                            Penalty = 0
                            
                    else:
                        Score = split_data[i][8 + skip_step]
                        Penalty = split_data[i][9 + skip_step]
                    
                    
                        
                    D_Score = split_data[i][10 + skip_step]
                    
                    E_Score = split_data[i][12 + skip_step]
                    
                    #%% Now finished data crawl
                    Dict = {
                        "LastName": Last_Name, 
                        "FirstName": First_Name,
                        "Gender": Gender,   
                        "Country": Country,
                        "Date": date,
                        "Competition": competition,
                        "Round": round,
                        "Location": location,
                        "Apparatus": 'VT2',
                        "Rank": Rank,
                        "D_Score": D_Score,
                        "E_Score": E_Score,
                        "Penalty": Penalty,
                        "Score": Score
                        }
                    
                    df = pd.DataFrame([Dict])
                    Data.append(df)
            
            print('----------')

            Final_Data = pd.concat(Data, ignore_index=True)
            print(Final_Data)
            
            A = Final_Data
            
            Final_Data.to_csv('SingleItem_Data.csv', index=False)
        