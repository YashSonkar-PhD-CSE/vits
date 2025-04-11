# speakerIndices = set()
# with open("./filelists/wtimit_normal_list.txt", "r+") as readFile:
#   for line in readFile.readlines():
#        speaker = int(line.strip().split("|")[1])
#        speakerIndices.add(speaker)
#speakerIndices = sorted(list(speakerIndices))
#speakerMap = {}
#for i in range(len(speakerIndices)):
#    speakerMap[speakerIndices[i]] = i
#
#trainLines, valLines = [], []
#with open("./filelists/wtimit_train.txt.cleaned", "r+") as trainFile:
#    for line in trainFile.readlines():
#        parts = line.strip().split("|")
#        path, index, transcript = parts[0], int(parts[1]), parts[2]
#        trainLines.append("|".join([path, str(speakerMap[index]), transcript]) + "\n")
#with open("./filelists/wtimit_val.txt.cleaned", "r+") as valFile:
#    for line in valFile.readlines():
#        parts = line.strip().split("|")
#        path, index, transcript = parts[0], int(parts[1]), parts[2]
#        valLines.append("|".join([path, str(speakerMap[index]), transcript]) + "\n")
#with open("./filelists/wtimit_train1.txt.cleaned", "w") as trainFile:
#    trainFile.writelines(trainLines)
#with open("./filelists/wtimit_val1.txt.cleaned", "w") as valFiles:
#    valFiles.writelines(valLines)
import librosa
import numpy as np

maxWavValue = 0
with open('./filelists/wtimit_normal_list.txt', "r+") as writeFile:
    for line in writeFile.readlines():
        parts = line.split('|')
        path = parts[0]
        audio, sr = librosa.load(path, sr=24_000, dtype=np.float32)
        audioMax = audio.max()
        maxWavValue = maxWavValue if maxWavValue > audioMax else audioMax
print(maxWavValue)
