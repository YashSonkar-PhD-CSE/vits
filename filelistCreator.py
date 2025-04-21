import copy
import os
import librosa
import numpy as np
# NORMAL_DIR = "/ssd_scratch/cvit/yash/converted_wavs/normal"
# FILELIST_PATH = "/ssd_scratch/cvit/yash/vits/filelists/wtimit_normal_list.txt"
# speakers = os.listdir(NORMAL_DIR)
# speakers = [int(s[1:]) for s in speakers]

# files = {}
# for speaker in speakers:
#     files[speaker] = os.listdir(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "wavs"))

# speakerIndex = 0
# with open(FILELIST_PATH, "w") as filelist:
#     for speaker in speakers:
#         fileIndex = 0
#         textFilePath = files[speaker]
#         for file in textFilePath:
#             print(speakerIndex, "/", fileIndex, end="\r")
#             fileIndex += 1
#             with open(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "txt", file.replace(".wav", ".txt")), "r+") as txtFile:
#                 wavPath = os.path.join(NORMAL_DIR, f"s{speaker:03d}", "wavs", file)
#                 filelist.write(f"{wavPath}|{speaker}|{txtFile.read().strip()}\n")
#         speakerIndex += 1

# speakerIndices = set()
# with open('./filelists/wtimit_val2.txt.cleaned', "r+") as readFile:
#     lines = readFile.readlines()
# lines = [l.strip() for l in lines]

# speakerIndices = set([l.split('|')[1] for l in lines])
# speakerMap = {}
# for i, speakerIndex in enumerate(speakerIndices):
#     speakerMap[speakerIndex] = i
# print(len(speakerIndices))
# maxWavValue = 0
# with open('./filelists/wtimit_val3.txt.cleaned', "r+") as writeFile:
#     for line in writeFile.readlines():
#         parts = line.split('|')
#         path = parts[0]
#         audio, sr = librosa.load(path, sr=24_000, dtype=np.float32)
#         audioMax = audio.max()
#         maxWavValue = maxWavValue if maxWavValue > audioMax else audioMax
# print(maxWavValue)


def search(filePath, sid, cleanedText):
    wavId = filePath.split("/")[-1]
    # print(wavId, sid)
    seen = []
    for fP, si, ct in cleanedText:
        # if fP == wavId and si != sid:
        #     # print(fP, end=",")
        # if si == sid and fP != wavId:
            # print(si, end=",")
        if wavId == fP and si == sid:
            # print(fP, wavId, sid, si)
            return ct
    raise ValueError

filePaths, sids = [], []
with open('./wtimit_train2.txt.cleaned', 'r+') as cF:
    cleanedText = cF.readlines()
trainFilePaths = [cT.split("|")[0].split("/")[-1] for cT in cleanedText]
trainSids = [int(cT.split("|")[1]) for cT in cleanedText]
cTs = [cT.split("|")[2] for cT in cleanedText]
cleanedText = list(zip(trainFilePaths, trainSids, cTs))
for speaker in os.listdir('./vFiles'):
    for file in os.listdir(f'./vFiles/{speaker}'):
        # if file.endswith('393.wav'):
            # print(speaker, "_", int(speaker))
            # print("HERE")
        filePaths.append(f'./vFiles/{speaker}/{file}')
with open('test.txt.cleaned', 'w') as newFile:
    for filePath in filePaths:
        print(filePath.split('/'))
        sid = int(filePath.split('/')[-2])
        cT = search(filePath, sid, copy.copy(cleanedText))
        # if filePath.endswith('393.wav') and sid == 51:
        #     print("Found it", sid)
        newFile.write(f'{filePath}|{sid}|{cT.strip()}\n')