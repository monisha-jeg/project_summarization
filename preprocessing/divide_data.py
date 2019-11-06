import os, json
from tqdm import tqdm
PARENT_DIR = "../"
print("All files: ", len(os.listdir(PARENT_DIR+"xsum-extracts-from-downloads/train")))


split_json_file = "../XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
split_json = json.load(open(split_json_file, 'r'))

origin = PARENT_DIR+"xsum-extracts-from-downloads/train/"
dest = PARENT_DIR+"xsum-extracts-from-downloads/val/"
os.makedirs(dest, exist_ok=True)
for filename in tqdm(split_json["validation"]):
	command = "mv " + origin + filename + ".data " + dest
	os.system(command)

origin = PARENT_DIR+"xsum-extracts-from-downloads/train/"
dest = PARENT_DIR+"xsum-extracts-from-downloads/test/"
os.makedirs(dest, exist_ok=True)
for filename in tqdm(split_json["test"]):
	command = "mv " + origin + filename + ".data " + dest
	os.system(command)

print("Train files: ", len(os.listdir(PARENT_DIR+"xsum-extracts-from-downloads/train")))
print("Val files: ", len(os.listdir(PARENT_DIR+"xsum-extracts-from-downloads/val")))
print("Test files: ", len(os.listdir(PARENT_DIR+"xsum-extracts-from-downloads/test")))


