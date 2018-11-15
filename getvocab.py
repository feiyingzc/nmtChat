f = open('./sgns.weibo.word', 'r', encoding='utf-8')
fdata = open('./sgns.weibo.worddata', 'w+', encoding='utf-8')
fout = open('./vocabsrc', 'w+', encoding='utf-8')
# ftgt = open('./convtgt.txt', 'w+', encoding='utf-8')
print(f.readline())
linedata=[]
i=0
for line in f.readlines():
  #print(line)
  line = line.strip()
  lined = line.split(" ")
  #print(lined[1:])
  #print(len(lined[1:]))
  if(len(lined) < 1 or lined[0] in linedata or len(lined[1:]) != 300):
  	continue
  linedata.append(lined[0])
  #print(lined[0])
  fdata.write(line+"\n")
  i +=1
  if(i%10000 == 0):
    print(i)
#fdata.close()
#fdata = open('./sgns.weibo.worddata', 'w+', encoding='utf-8')
fdata.seek(0)
i=0
fout.write("<unk>"+"\n")
fout.write("<s>"+"\n")
fout.write("</s>"+"\n")
for line in fdata.readlines():
  #print(line)
  lined = line.split(" ")
  if(len(lined) < 1):
  	continue
  fout.write(lined[0]+"\n")
  i+=1
  if(i%10000 == 0):
    print(i)
  # ftgt.write(lined[1].strip()+"\n")
f.close()
fdata.close()
fout.close()
# ftgt.close()