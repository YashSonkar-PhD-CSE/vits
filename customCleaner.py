import text
import os
if __name__ == "__main__":
    # filelist = "wfilelists/wtimit_val2.txt"
    # textIndex = 2 # Index of transcript in filelist.txt
    # print("Starting")
    # lines = []
    # cleanedLines = []
    # parts = []
    # print("Reading")
    t = "This eBook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever."
    t = "By Adam Smith"
    cleanText = text._clean_text(t, ['english_cleaners2'])
    print(cleanText)
    # with open(filelist) as f:
    #     index = 0
    #     line = f.readline()
    #     while line:
    #         print("Progress: ", index, end="\r")
    #         index += 1
    #         line = line.strip()
    #         lines.append(line.split("|")[textIndex])
    #         parts.append("|".join(line.split("|")[:2]))
    #         line = f.readline()
    # print("Num lines: ", index)
    # print("Cleaning")
    # index = 0
    # for line in lines:
    #     print("Progress: ", index, end="\r")
    #     if index <= 16 * 512:
    #         index += 1
    #         continue
    #     if index % 512 == 0 and index != 0:
    #         with open(filelist + f"_{index//512}_.cleaned", "w") as f:
    #             for i, cleanedLine in enumerate(cleanedLines):
    #                 f.write(cleanedLine + "\n")
    #             cleanedLines.clear()
    #         print("Saved ", index // 512)
    #     cleaned_text = text._clean_text(line, ["english_cleaners2"])
    #     cleanedLines.append("|".join([parts[index], cleaned_text]))
    #     index += 1
    # if len(cleanedLines) != 0:
    #     with open(filelist + f"_{index//512+1}_.cleaned", "w") as f:
    #         for i, cleanedLine in enumerate(cleanedLines):
    #             f.write(cleanedLine + "\n")
    #         cleanedLines.clear()
    #     print("Saved ", index // 512 + 1)
    # # print("Writing")
    # # with open(filelist + ".cleaned", "w") as f:
    # #     for i, cleanedLine in enumerate(cleanedLines):
    # #         print("Writing: ", i, end="\r")
    # #         f.write(cleanedLine + "\n")
    # print("Finished")
    # trainCleanedLines = []
    # for file in os.listdir('./wfilelists'):
    #     if file.startswith('wtimit_val2') and file.endswith('.cleaned'):
    #         with open(os.path.join('./wfilelists', file), 'r+') as f:
    #             trainCleanedLines.extend(f.readlines())
    # with open('./wfilelists/wtimit_val2.txt.cleaned', 'w') as wF:
    #     wF.writelines(trainCleanedLines)