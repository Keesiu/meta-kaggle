import sys
subdir=sys.argv[1]
f1=open(subdir+'xgb-raw')
f2=open(subdir+'ave-xgb-part1-part2-random-online.csv')

fo=open(subdir+'ave-xgb-raw-part1-part2-random-online.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()

c=0
for l,m in zip(f1,f2):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])

    p=p1*0.1+p2*0.9
    fo=open(subdir+'ave-xgb-raw-part1-part2-random-online.csv','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print(c,'processed')
f1.close()
f2.close()

fo.close()
