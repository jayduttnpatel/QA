import os ,glob 
for file in glob.glob("*.txt"):
	file=file[:-4]
	os.rename(file+".txt",file+".srt.txt")