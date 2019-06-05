import os ,glob 
import re
for file in glob.glob("*.txt"):
	a=re.match('.*srt.*',file)
	if a:
		continue
	file=file[:-4]
	os.rename(file+".txt",file+".srt.txt")