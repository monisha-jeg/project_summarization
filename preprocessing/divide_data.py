import os, json

print("All files: ", len(os.listdir("../xsum-extracts-from-downloads")))


split_json_file = "../XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
split_json = json.load(open(split_json_file, 'r'))

origin = "../xsum-extracts-from-downloads/"
dest = "../xsum-extracts-from-downloads/val/"
for filename in split_json["validation"]:
	command = "mv " + origin + filename + ".data " + dest
	os.system(command)

origin = "../xsum-extracts-from-downloads/"
dest = "../xsum-extracts-from-downloads/test/"
for filename in split_json["test"]:
	command = "mv " + origin + filename + ".data " + dest
	os.system(command)

print("Train files: ", len(os.listdir("../xsum-extracts-from-downloads")))
print("Val files: ", len(os.listdir("../xsum-extracts-from-downloads/val")))
print("Test files: ", len(os.listdir("../xsum-extracts-from-downloads/test")))


