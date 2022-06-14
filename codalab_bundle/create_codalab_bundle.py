import os
import zipfile
import shutil

current_folder = os.getcwd()
bundle_file_name = 'bundle.zip'

#Copy code from the working directory to the scoring_program
folder_out = 'scoring_program'
folder_in = '../data_challenge_preparation'
list_of_files = os.listdir(folder_in)
list_of_file_to_copy = [i for i in list_of_files if i.endswith(('.py','.toml'))]
for file_name in list_of_file_to_copy :
    src = os.path.join(folder_in,file_name)
    dst = os.path.join(folder_out,file_name)
    shutil.copyfile(src, dst)


# zip folders
folder_to_zip = ['reference_data','scoring_program']
for folder in folder_to_zip :
    path_folder = os.path.join(current_folder,folder)
    path_folder_zip = os.path.join(current_folder,"{}.zip".format(folder))
    if os.path.exists(path_folder_zip) :
        os.remove(path_folder_zip)
    shutil.make_archive(path_folder, 'zip', path_folder)

# zip file
list_of_files = os.listdir(current_folder)
list_of_files_to_zip = [i for i in list_of_files if i.endswith(('.html','.zip','.yaml','.jpg'))]

# Delete all bundle
bundle_path = os.path.join(current_folder,bundle_file_name)
if os.path.exists(bundle_path):
    os.remove(bundle_path)

# Create new bundle
with zipfile.ZipFile(bundle_file_name, 'w') as zipF:
    for file in list_of_files_to_zip:
        zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)














