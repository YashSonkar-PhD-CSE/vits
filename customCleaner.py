import text

if __name__ == "__main__":
    filelist = "filelists/wtimit_normal_list.txt"
    textIndex = 2 # Index of transcript in filelist.txt
    print("Starting")
    lines = []
    cleanedLines = []
    print("Reading")
    with open(filelist) as f:
        index = 0
        line = f.readline()
        while line:
            print("Progress: ", index, end="\r")
            index += 1
            line = line.strip()
            lines.append(line.split("|")[textIndex])
            line = f.readline()
    print("Cleaning")
    index = 0
    for line in lines:
        print("Progress: ", index, end="\r")
        index += 1
        cleaned_text = text._clean_text(line, ["english_cleaners2"])
        cleanedLines.append(cleaned_text)
    print("Writing")
    with open(filelist + ".cleaned", "w") as f:
        for i, cleanedLine in enumerate(cleanedLines):
            print("Writing: ", i, end="\r")
            f.write(cleanedLine + "\n")
    print("Finished")
