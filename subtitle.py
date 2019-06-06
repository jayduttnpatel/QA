import re

def Fetch(file):
	f=open(file,'r',encoding='UTF8')
	text=""
	for line in f:
		a=re.match("\d",line)  # check for the line with the number only
		b=re.match(".*-->.*",line) # check with the line with timeline
		c=re.match(".+",line) #atleast one character
		if a or b or not c:
			continue
		text=text+" "+line.strip()
	f.close()
	text=text.strip()
	return text