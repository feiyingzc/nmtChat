f = open('./conv.txt', 'r', encoding='utf-8')
ftrain = open('./train', 'w+', encoding='utf-8')
fdev = open('./dev', 'w+', encoding='utf-8')
ftest = open('./test', 'w+', encoding='utf-8')
# ftgt = open('./convtgt.txt', 'w+', encoding='utf-8')
#print(f.readline())
i=0
for line in f.readlines():
  i += 1
  if(len(line.strip()) < 1):
  	continue
  if(i<=6415):
    ftrain.write(line.strip()+"\n")
  elif(i <= 350000):
  	fdev.write(line.strip()+"\n")
  else:
  	ftest.write(line.strip()+"\n")
f.close()
ftrain.close()
fdev.close()
ftest.close()