import glob, os
import operator
import re

#convert words to the numbers for that find the frequency of the word sort it in the increasing order
#WordToNum function will do this task
#subtitle file name have file.str.txt and QA have file.txt
def SubtitleText(file):
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
	
def QAFileText(file):
	f=open(file,'r',encoding='UTF8')
	text=""
	for line in f:
		line=line.strip()
		
		tp=re.match(".+",line) #atleast one character
		if not tp:
			continue
		
		tp=re.match("[Q,A,B,C,D]..+",line)
		if tp:
			line=line[2:] #remove Q. from the begining
			line=line.strip() 
			text=text+" "+line
		
		tp=re.match("\(.\)",line)
		if tp:
			continue
	f.close()
	text=text.strip()
	return text
	
def WordToNum():
	mapping={}
	os.chdir("/temp/files/subtitle")
	freq={}
	for file in glob.glob("*.txt"):
		if re.match(".*srt.*",file): # subtitle
			#print(file+' is subtile')
			text=SubtitleText(file)
		else: # QA file
			#print(file+' is normal')
			text=QAFileText(file)
		for word in text.split():
			freq[word]=freq.get(word,0)+1
	i=1
	for key,value in sorted(freq.items(), key=lambda item: item[1],reverse=True):
		mapping[key]=i
		i=i+1
	#for key,value in mapping.items():
	#	print(key,value)
	return mapping

def Convert(text,mapping):
	#return the text in the encoded manner
	ans=[]
	for t in text.split(' '):
		num=mapping.get(t,0)
		ans.append(num)
	return ans