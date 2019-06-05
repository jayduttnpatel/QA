import os ,glob 
import re
for file in glob.glob("*.txt"):
	a=re.match('.*srt.*',file)
	file=file[:-8]
	os.rename(file+".srt.txt",file+".txt")