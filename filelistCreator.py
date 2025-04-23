# import copy
# import os
# import librosa
# import numpy as np
# # NORMAL_DIR = "/ssd_scratch/cvit/yash/converted_wavs/normal"
# # FILELIST_PATH = "/ssd_scratch/cvit/yash/vits/filelists/wtimit_normal_list.txt"
# # speakers = os.listdir(NORMAL_DIR)
# # speakers = [int(s[1:]) for s in speakers]

# # files = {}
# # for speaker in speakers:
# #     files[speaker] = os.listdir(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "wavs"))

# # speakerIndex = 0
# # with open(FILELIST_PATH, "w") as filelist:
# #     for speaker in speakers:
# #         fileIndex = 0
# #         textFilePath = files[speaker]
# #         for file in textFilePath:
# #             print(speakerIndex, "/", fileIndex, end="\r")
# #             fileIndex += 1
# #             with open(os.path.join(NORMAL_DIR, f"s{speaker:03d}", "txt", file.replace(".wav", ".txt")), "r+") as txtFile:
# #                 wavPath = os.path.join(NORMAL_DIR, f"s{speaker:03d}", "wavs", file)
# #                 filelist.write(f"{wavPath}|{speaker}|{txtFile.read().strip()}\n")
# #         speakerIndex += 1

# # speakerIndices = set()
# # with open('./filelists/wtimit_val2.txt.cleaned', "r+") as readFile:
# #     lines = readFile.readlines()
# # lines = [l.strip() for l in lines]

# # speakerIndices = set([l.split('|')[1] for l in lines])
# # speakerMap = {}
# # for i, speakerIndex in enumerate(speakerIndices):
# #     speakerMap[speakerIndex] = i
# # print(len(speakerIndices))
# # maxWavValue = 0
# # with open('./filelists/wtimit_val3.txt.cleaned', "r+") as writeFile:
# #     for line in writeFile.readlines():
# #         parts = line.split('|')
# #         path = parts[0]
# #         audio, sr = librosa.load(path, sr=24_000, dtype=np.float32)
# #         audioMax = audio.max()
# #         maxWavValue = maxWavValue if maxWavValue > audioMax else audioMax
# # print(maxWavValue)


# def search(filePath, sid, cleanedText):
#     wavId = filePath.split("/")[-1]
#     # print(wavId, sid)
#     seen = []
#     for fP, si, ct in cleanedText:
#         # if fP == wavId and si != sid:
#         #     # print(fP, end=",")
#         # if si == sid and fP != wavId:
#             # print(si, end=",")
#         if wavId == fP and si == sid:
#             # print(fP, wavId, sid, si)
#             return ct
#     raise ValueError

# filePaths, sids = [], []
# with open('./wtimit_train2.txt.cleaned', 'r+') as cF:
#     cleanedText = cF.readlines()
# trainFilePaths = [cT.split("|")[0].split("/")[-1] for cT in cleanedText]
# trainSids = [int(cT.split("|")[1]) for cT in cleanedText]
# cTs = [cT.split("|")[2] for cT in cleanedText]
# cleanedText = list(zip(trainFilePaths, trainSids, cTs))
# for speaker in os.listdir('./vFiles'):
#     for file in os.listdir(f'./vFiles/{speaker}'):
#         # if file.endswith('393.wav'):
#             # print(speaker, "_", int(speaker))
#             # print("HERE")
#         filePaths.append(f'./vFiles/{speaker}/{file}')
# with open('test.txt.cleaned', 'w') as newFile:
#     for filePath in filePaths:
#         print(filePath.split('/'))
#         sid = int(filePath.split('/')[-2])
#         cT = search(filePath, sid, copy.copy(cleanedText))
#         # if filePath.endswith('393.wav') and sid == 51:
#         #     print("Found it", sid)
#         newFile.write(f'{filePath}|{sid}|{cT.strip()}\n')
# speakers = []
# lines = []
# with open('wtimit_train2.txt.cleaned', 'r+') as readFile:
#     for line in readFile.readlines():
#         # sp = int(line.strip().split("|")[1])
#         path = line.strip().split("|")[0]
#         sp = int(path.split("/")[-3][1:])
#         if sp not in speakers:
#             speakers.append(sp)
#         lines.append(line.strip())  
# speakers = sorted(speakers)
# speakerMap = {}
# speakerMap = {
#     0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 
#     13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 
#     24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 
#     35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 
#     46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 
#     57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 
#     68: 68, 69: 69, 70: 70, 71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 101: 76, 102: 77, 103: 78, 
#     104: 79, 105: 80, 108: 81, 109: 82, 110: 83, 111: 84, 112: 85, 116: 86, 117: 87, 118: 88, 
#     119: 89, 121: 90, 122: 91, 123: 92, 124: 93, 125: 94, 126: 95, 127: 96, 128: 97, 129: 98, 
#     130: 99, 131: 100
# }
# for i, speaker in enumerate(speakers):
#     speakerMap[speaker] = i
# print(speakerMap)
# print(len(speakerMap))
# print(len(set(speakerMap.keys())))
import random

lines = []
with open('wtimit_train2_1.txt.cleaned', 'r+') as readFile:
    lines = readFile.readlines()
lines = [l.strip() for l in lines]
valLines = random.sample(lines, 2000)
with open('wtimit_val.txt.cleaned', 'w') as writeFile:
    for line in valLines:
        writeFile.write(line + "\n")