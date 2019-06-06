import re

def Convert(mapping):
	f = open("temp.srt.txt", "r")
	for line in f:
		a=re.match("\d",line)  # check for the line with the number only
		b=re.match(".*-->.*",line) # check with the line with timeline
		c=re.match(".+",line) #atleast one character
		if a or b or not c:
			continue
		#print(line.strip())
	f.close()
	return line.strip()