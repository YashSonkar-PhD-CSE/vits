import text
import os
if __name__ == "__main__":
    cleanedFileList = "wtimit_train2.txt.cleaned"
    fileList = "wtimit_normal_list1.txt_temp"
    lines = []
    with open(fileList, "r+") as readFile:
        lines = readFile.readlines()
        
    lines = [l.strip() for l in lines]
    cleanLines = []
    for i, line in enumerate(lines):
        print("Progress: ", i, " / ", len(lines), end="\r")
        parts = line.split('|')
        path = parts[0]
        sid = parts[1]
        transcript = parts[2]
        cleanedText = text._clean_text(transcript, ["english_cleaners2"])
        cleanLines.append(f"{path}|{sid}|{cleanedText}\n")
    with open(cleanedFileList + "1", "w") as writeFile:
        for line in cleanLines:
            writeFile.write(line)