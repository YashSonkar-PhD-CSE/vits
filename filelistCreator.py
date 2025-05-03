import os

NORMAL_DIR = "/mnt/Shared-Storage/yash/wTIMIT/converted_wavs/whisper"
FILELIST_PATH = "/mnt/Shared-Storage/yash/vits/wtimit_normal_list1.txt"
speakers = os.listdir(NORMAL_DIR)
speakers = [int(s[1:]) for s in speakers]

files = {}
for speaker in speakers:
    files[speaker] = os.listdir(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "newWav"))

speakerIndex = 0
with open(FILELIST_PATH, "a") as filelist:
    for speaker in speakers:
        fileIndex = 0
        textFilePath = files[speaker]
        for file in textFilePath:
            print(speakerIndex, "/", fileIndex, end="\r")
            fileIndex += 1
            with open(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "txt", file.replace(".wav", ".txt")), "r+") as txtFile:
                wavPath = os.path.join(NORMAL_DIR, f"s{speaker:03d}", "newWav", file)
                filelist.write(f"{wavPath}|{speaker}|{txtFile.read().strip()}\n")
        speakerIndex += 1

speakerIndices = set()
with open("./filelists/wtimit_normal_list.txt", "r+") as readFile:
    for line in readFile.readlines():
        speaker = int(line.strip().split("|")[1])
        speakerIndices.add(speaker)
speakerIndices = sorted(list(speakerIndices))
speakerMap = {}
for i in range(len(speakerIndices)):
    speakerMap[speakerIndices[i]] = i

# trainLines, valLines = [], []
# with open("./filelists/wtimit_train.txt.cleaned", "r+") as trainFile:
#     for line in trainFile.readlines():
#         parts = line.strip().split("|")
#         path, index, transcript = parts[0], int(parts[1]), parts[2]
#         trainLines.append("|".join([path, str(speakerMap[index]), transcript]) + "\n")
# with open("./filelists/wtimit_val.txt.cleaned", "r+") as valFile:
#     for line in valFile.readlines():
#         parts = line.strip().split("|")
#         path, index, transcript = parts[0], int(parts[1]), parts[2]
#         valLines.append("|".join([path, str(speakerMap[index]), transcript]) + "\n")
# with open("./filelists/wtimit_train1.txt.cleaned", "w") as trainFile:
#     trainFile.writelines(trainLines)
# with open("./filelists/wtimit_val1.txt.cleaned", "w") as valFiles:
#     valFiles.writelines(valLines)