'''
This file collect the data and encode it in the numbers and return it.
'''

import subtitle
import os,glob
import random
import QA
import WordToNum

'''def subtitle(mapping):
	Sub=[]
	os.chdir("/temp/files/subtitle")
	cnt=0
	for file in glob.glob("*.txt"):
		text=WordToNum.SubtitleText(file)
		Sub.append( WordToNum.Convert(text,mapping) )
		cnt=cnt+1
		
	Y=[]
	for i in range(cnt):
		Y.append(random.randint(0,1))
	return Sub,Y   '''
	
def data(mapping):
	Sub=[]
	Q=[]
	A=[]
	B=[]
	C=[]
	D=[]
	T=[]
	
	os.chdir("/temp/files/subtitle")
	for file in glob.glob("*.srt.txt"):
		subfile=file
		qafile=file[:-8]+".txt"
		#print(file[:-8],subfile,qafile)
		
		text=subtitle.Fetch(subfile)
		temp=WordToNum.Convert(text,mapping)
		
		q,a,t=QA.Fetch("D:\\temp\\files\\QA\\"+qafile)
		
		for i in range(len(q)):
			Q.append( WordToNum.Convert(q[i],mapping) )
			Sub.append( temp )
			if t[i]=="A":
				T.append( [1,0,0,0] )
			elif t[i]=="B":
				T.append( [0,1,0,0] )
			elif t[i]=="C":
				T.append( [0,0,1,0] )
			else:
				T.append( [0,0,0,1] )
			A.append( WordToNum.Convert(a[i][0],mapping) )
			B.append( WordToNum.Convert(a[i][1],mapping) )
			C.append( WordToNum.Convert(a[i][2],mapping) )
			D.append( WordToNum.Convert(a[i][3],mapping) )
	
	return Sub,Q,T,A,B,C,D