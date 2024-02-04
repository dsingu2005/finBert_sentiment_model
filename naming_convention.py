import os
import re

def get_quarter(month):
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4

def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file)
                if match:
                    year, month, day = map(int, match.groups())
                    quarter = get_quarter(month)
                    company_code = root.split('/')[-1]
                    new_filename = f"CC_{company_code}_Q{quarter}{year}_{day}_{month}_{year}.xlsx"
                    old_file_path = os.path.join(root, file)
                    
                    new_directory = os.path.join(directory, "New 29 Companies", company_code)
                    if not os.path.exists(new_directory):
                        os.makedirs(new_directory)
                    
                    new_file_path = os.path.join(new_directory, new_filename)
                    os.rename(old_file_path, new_file_path)

# rename_files_in_directory("29 Companies")

def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file)
                if match:
                    year, month, day = map(int, match.groups())
                    quarter = get_quarter(month)
                    if quarter == 1:
                        print("Q1", os.path.join(root, file))
                    else:
                        print(os.path.join(root, file))

# list_files("29 Companies")
                        

def rename_files_in_directory(directory):
    new_directory = os.path.join(directory, "New 29 Companies")
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file)
                if match:
                    year, month, day = map(int, match.groups())
                    quarter = get_quarter(month)
                    company_code = root.split('/')[-1]
                    new_filename = f"CC_{company_code}_Q{quarter}{year}_{day}_{month}_{year}.xlsx"
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(new_directory, new_filename)
                    os.rename(old_file_path, new_file_path)
            elif file.endswith(".png") and "_output" in file:
                company_code = root.split('/')[-1]
                new_filename = f"{company_code}_RAWSENTIMENT.png"
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(new_directory, new_filename)
                os.rename(old_file_path, new_file_path)

# rename_files_in_directory("29 Companies")
                

def flip_day_month_in_filenames(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                match = re.search(r'CC_(.*)_Q(\d{1})(\d{4})_(\d{1,2})_(\d{1,2})_(\d{4})', file)
                if match:
                    company_code, quarter, year, day, month, year2 = match.groups()
                    new_filename = f"CC_{company_code}_Q{quarter}{year}_{month}_{day}_{year2}.xlsx"
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, new_filename)
                    os.rename(old_file_path, new_file_path)

flip_day_month_in_filenames("29 Companies")