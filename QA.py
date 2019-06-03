import re

f = open("3_1.txt", "r")

Q=[] # store the questions
A=[] # store the options
T=[] # store the target answers 

for line in f:
	line=line.strip()
	
	tp=re.match(".+",line) #atleast one character
	if not tp:
		continue
	
	tp=re.match("Q..+",line)
	if tp:
		line=line[2:] #remove Q. from the begining
		line=line.strip() 
		Q.append(line)
	
	tp=re.match("A..+",line)
	if tp:
		ls=[]
		for i in range(4):
			if i != 0:
				line=f.readline()
			line=line[2:] #remove A. from the begining
			line=line.strip()
			ls.append(line)
		A.append(ls)
	
	tp=re.match("\(.\)",line)
	if tp:
		line=line[1:2] #remove brackets
		T.append(line)
f.close()
print(Q)
print(A)
print(T)