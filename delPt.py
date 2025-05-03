import os

ROOT_DIR = "/ssd_scratch/cvit/yash/converted_wavs/normal"

for speakers in os.listdir(ROOT_DIR):
    for files in os.listdir(os.path.join(ROOT_DIR, speakers, "wavs")):
        if files.endswith(".spec.pt"):
            os.system(f"rm {os.path.join(ROOT_DIR, speakers, 'wavs', files)}")

