ftrain = open('./train', 'r', encoding='utf-8')
fdev = open('./dev', 'r', encoding='utf-8')
ftest = open('./test', 'r', encoding='utf-8')
ftrainsrc = open('./trainsrc.txt', 'w+', encoding='utf-8')
ftraintgt = open('./traintgt.txt', 'w+', encoding='utf-8')
fdevsrc = open('./devsrc.txt', 'w+', encoding='utf-8')
fdevtgt = open('./devtgt.txt', 'w+', encoding='utf-8')
ftestsrc = open('./testsrc.txt', 'w+', encoding='utf-8')
ftesttgt = open('./testtgt.txt', 'w+', encoding='utf-8')
#print(f.readline())
for line in ftrain.readlines():
  #print(line)
  emtFlag = False
  lined = line.strip().split("|")
  for element in lined:
    if len(element.strip()) < 1:
      emtFlag = True
  if len(lined) < 2:
  	continue
  lined[0] = ' '.join(lined[0].split())
  lined[1] = ' '.join(lined[1].split())
  if emtFlag == False:
    ftrainsrc.write(lined[0].strip().replace(" ",",")+"\n")
    ftraintgt.write(lined[1].strip().replace(" ",",")+"\n")
for line in fdev.readlines():
  emtFlag = False
  lined = line.strip().split("|")
  for element in lined:
    if len(element.strip()) < 1:
      emtFlag = True
  if len(lined) < 2:
  	continue
  lined[0] = ' '.join(lined[0].split())
  lined[1] = ' '.join(lined[1].split())
  if emtFlag == False:
    fdevsrc.write(lined[0].strip().replace(" ",",")+"\n")
    fdevtgt.write(lined[1].strip().replace(" ",",")+"\n")
for line in ftest.readlines():
  emtFlag = False
  lined = line.strip().split("|")
  for element in lined:
    if len(element.strip()) < 1:
      emtFlag = True
  if len(lined) < 2:
  	continue
  lined[0] = ' '.join(lined[0].split())
  lined[1] = ' '.join(lined[1].split())
  if emtFlag == False:
    ftestsrc.write(lined[0].strip().replace(" ",",")+"\n")
    ftesttgt.write(lined[1].strip().replace(" ",",")+"\n")
ftrain.close()
fdev.close()
ftest.close()
ftrainsrc.close()
ftraintgt.close()
fdevsrc.close()
fdevtgt.close()
ftestsrc.close()
ftesttgt.close()