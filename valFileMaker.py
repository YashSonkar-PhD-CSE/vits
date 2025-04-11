import os

valFiles = []

with open('./filelists/wtimit_val.txt.cleaned', 'r+') as readFile:
    valFiles = readFile.readlines()

valFiles = [v.split('|')[0] for v in valFiles]

os.makedirs('./valFiles/', exist_ok=True)
for file in valFiles:
    # /ssd_scratch/cvit/yash/converted_wavs/normal/sIII/wavs/fileId.wav
    #                                               -3   -2         -1
    speaker = file.split('/')[-3]
    print(speaker)
    os.makedirs(f"./valFiles/{speaker}", exist_ok=True)
    os.system(f'cp {file} ./valFiles/{speaker}/')
