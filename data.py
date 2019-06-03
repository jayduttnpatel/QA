import WordToNum
import os,glob
import random

def train(mapping):
	Sub=[]
	os.chdir("/temp/train/subtitle")
	cnt=0
	for file in glob.glob("*.txt"):
		text=WordToNum.SubtitleText(file)
		Sub.append( WordToNum.Convert(text,mapping) )
		cnt=cnt+1
		
	Y=[]
	for i in range(cnt):
		Y.append(random.randint(0,1))
	return Sub,Y
	
def test():
	pass