import os

NORMAL_DIR = "/ssd_scratch/cvit/yash/converted_wavs/normal"
FILELIST_PATH = "/ssd_scratch/cvit/yash/vits/filelists/wtimit_normal_list.txt"
speakers = os.listdir(NORMAL_DIR)
speakers = [int(s[1:]) for s in speakers]

files = {}
for speaker in speakers:
    files[speaker] = os.listdir(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "wavs"))

speakerIndex = 0
with open(FILELIST_PATH, "w") as filelist:
    for speaker in speakers:
        fileIndex = 0
        textFilePath = files[speaker]
        for file in textFilePath:
            print(speakerIndex, "/", fileIndex, end="\r")
            fileIndex += 1
            with open(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "txt", file.replace(".wav", ".txt")), "r+") as txtFile:
                wavPath = os.path.join(NORMAL_DIR, f"s{speaker:03d}", "wavs", file)
                filelist.write(f"{wavPath}|{speaker}|{txtFile.read().strip()}\n")
        speakerIndex += 1
