import jieba
import sys

ftrainsrc = open('./trainsrc.txt', 'r', encoding='utf-8')
ftrainsrccut = open('./trainsrccut.txt', 'w+', encoding='utf-8')
ftraintgt = open('./traintgt.txt', 'r', encoding='utf-8')
ftraintgtcut = open('./traintgtcut.txt', 'w+', encoding='utf-8')
fdevsrc = open('./devsrc.txt', 'r', encoding='utf-8')
fdevsrccut = open('./devsrccut.txt', 'w+', encoding='utf-8')
fdevtgt = open('./devtgt.txt', 'r', encoding='utf-8')
fdevtgtcut = open('./devtgtcut.txt', 'w+', encoding='utf-8')
ftestsrc = open('./testsrc.txt', 'r', encoding='utf-8')
ftestsrccut = open('./testsrccut.txt', 'w+', encoding='utf-8')
ftesttgt = open('./testtgt.txt', 'r', encoding='utf-8')
ftesttgtcut = open('./testtgtcut.txt', 'w+', encoding='utf-8')
for line in ftrainsrc.readlines():
  seg_list = jieba.cut(line, cut_all=False)
  ftrainsrccut.write((" ".join(seg_list)).strip()+"\n")
for line in ftraintgt.readlines():
  seg_list = jieba.cut(line, cut_all=False)
  ftraintgtcut.write((" ".join(seg_list)).strip()+"\n")
for line in fdevsrc.readlines():
  seg_list = jieba.cut(line, cut_all=False)
  fdevsrccut.write((" ".join(seg_list)).strip()+"\n")
for line in fdevtgt.readlines():
  seg_list = jieba.cut(line, cut_all=False)
  fdevtgtcut.write((" ".join(seg_list)).strip()+"\n")
for line in ftestsrc.readlines():
  seg_list = jieba.cut(line, cut_all=False)
  ftestsrccut.write((" ".join(seg_list)).strip()+"\n")
for line in ftesttgt.readlines():
  seg_list = jieba.cut(line, cut_all=False)
  ftesttgtcut.write((" ".join(seg_list)).strip()+"\n")
ftrainsrc.close()
ftrainsrccut.close()
ftraintgt.close()
ftraintgtcut.close()
fdevsrc.close()
fdevsrccut.close()
fdevtgt.close()
fdevtgtcut.close()
ftestsrc.close()
ftestsrccut.close()
ftesttgt.close()
ftesttgtcut.close()
print("finish")
sys.exit()
