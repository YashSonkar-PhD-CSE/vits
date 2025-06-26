# # # import text

# # # if __name__ == "__main__":
# # #     filelist = "wtimit_normal_list1.txt"
# # #     textIndex = 2 # Index of transcript in filelist.txt
# # #     print("Starting")
# # #     lines = []
# # #     cleanedLines = []
# # #     print("Reading")
# # #     with open(filelist) as f:
# # #         index = 0
# # #         line = f.readline()
# # #         while line:
# # #             print("Progress: ", index, end="\r")
# # #             index += 1
# # #             line = line.strip()
# # #             lines.append(line)
# # #             line = f.readline()
# # #     print("Cleaning" + " " * 30)
# # #     index = 0
# # #     for i, line in enumerate(lines):
# # #         print("Progress: ", index, end="\r")
# # #         index += 1
# # #         if i <= 4096:
# # #             continue
# # #         parts = line.strip().split("|")
# # #         cleaned_text = text._clean_text(parts[textIndex], ["english_cleaners2"])
# # #         cleanedLines.append(cleaned_text)
# # #         if i % 4096 == 0 and i != 0:
# # #             print(f"Writing current progress {i // 4096}", end="\r")
# # #             with open(filelist + f"_{i // 4096}_" + ".cleaned", "w") as writeFile:
# # #                 for cline in cleanedLines:
# # #                     # Hard-coded for textIndex = 2
# # #                     writeFile.write(f"{parts[0]}|{parts[1]}|{cline}\n")
# # #             cleanedLines.clear()
# # #             print(len(cleanedLines), " " * 40)
# # #     print("Finished")

# # import text
# # import os
# # if __name__ == "__main__":
# #     cleanedFileList = "filelists/wt_custom.txt.cleaned"
# #     fileList = "filelists/wt_custom.txt"
# #     lines = []
# #     with open(fileList, "r+") as readFile:
# #         lines = readFile.readlines()
        
# #     lines = [l.strip() for l in lines]
# #     cleanLines = []
# #     for i, line in enumerate(lines):
# #         print("Progress: ", i, " / ", len(lines), end="\r")
# #         parts = line.split('|')
# #         path = parts[0]
# #         sid = parts[1]
# #         transcript = parts[2]
# #         cleanedText = text._clean_text(transcript, ["english_cleaners2"])
# #         cleanLines.append(f"{path}|{sid}|{cleanedText}\n")
# #     with open(cleanedFileList + "1", "w") as writeFile:
# #         for line in cleanLines:
# #             writeFile.write(line)

# import random

# data = []
# with open("./filelists/wt_custom1_train.txt.cleaned", 'r+') as trainCleanedFile:
#     for line in trainCleanedFile.readlines():
#         if "|" in line:
#             data.append(line)

# with open("./filelists/wt_custom1_val.txt.cleaned", 'r+') as valCleanedFile:
#     for line in valCleanedFile.readlines():
#         if "|" in line:
#             data.append(line)
            
# def extractSpId(path: str) -> int:
#     return path.split("/")[-1].split("_")[0]

# random.shuffle(data)
# testSpeakers = [ "s000", "s001", "s102", "s103"]
# trainData = [l for l in data if extractSpId(l.split("|")[0]) not in testSpeakers]
# trainSpeakers = {}
# for line in trainData:
#     speaker = extractSpId(line.split("|")[0])
#     if speaker not in trainSpeakers.keys():
#         trainSpeakers[speaker] = len(trainSpeakers)
        
# print(trainSpeakers)

# testData = [l for l in data if extractSpId(l.split("|")[0]) in testSpeakers]
# with open("wtimit_44_speakers.txt.cleaned", "w") as trainCleanedFile:
#     for data in trainData:
#         parts = data.strip().split("|")
#         try:
#             parts[-2] = str(trainSpeakers[extractSpId(parts[0])])
#             line = "|".join(parts)
#             trainCleanedFile.write(line.strip() + "\n")
#         except Exception as e:
#             print(data, e)
        
# # with open("wtimit_44_speakers_test.txt.cleaned", "w") as testCleanedFile:
# #     for data in testData:
# #         parts = data.strip().split("|")
# #         parts[-2] = trainSpeakers[parts[-2]]
# #         line = "|".join(parts)
# #         testCleanedFile.write(line.strip() + "\n")
import random

spWiseData = {}
spWiseLen = {}
def extractSpId(path: str) -> int:
    spId = path.split("/")[-1].split("_")[0]
    # print(path, spId)
    return spId

with open("wtimit_44_speakers.txt.cleaned", 'r+') as readFile:
    for line in readFile.readlines():
        parts = line.split("|")
        spId = extractSpId(parts[0])
        if spId not in spWiseData.keys():
            spWiseData[spId] = list()
            spWiseLen[spId] = 0
        spWiseData[spId].append("|".join(parts))
        spWiseLen[spId] = len(spWiseData[spId])
        
trainData, valData = [], []
trainPart = 0.95
for speaker in list(spWiseData.keys()):
    spLen = spWiseLen[speaker]
    trainLen = int(trainPart * spLen)
    print(f"Speaker: {speaker}\t|Train Samples: {trainLen}\t|Val Samples: {spLen - trainLen}")
    # for spData in spWiseData[speaker]:
    # print(spWiseData[speaker][0])
    trainData.extend(spWiseData[speaker][:trainLen])
    valData.extend(spWiseData[speaker][trainLen:])
    # print(len(spWiseData[speaker]), speaker)
# print(trainData[0], len(trainData))
random.shuffle(trainData)
random.shuffle(valData)
with open("wtimit_44_speakers_train.txt.cleaned", "w") as trainFile:
    for line in trainData:
        trainFile.write(line.strip() + "\n")

with open("wtimit_44_speakers_val.txt.cleaned", "w") as valFile:
    for line in valData:
        valFile.write(line.strip() + "\n")
