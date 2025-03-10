import os
import shutil

def copy_code(input_folder,G_drive):
    files = os.listdir(input_folder)
    files = [f for f in files if f.endswith(('.py','.csv')) ]
    code_outdir = G_drive + '\\ code'
    if not os.path.exists(code_outdir):
        os.makedirs(code_outdir)    
    for f in files:
        shutil.copy(input_folder+'\\'+f,code_outdir)

