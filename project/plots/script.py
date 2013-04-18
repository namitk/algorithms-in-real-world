import sys

f = open(sys.argv[1], 'r')
l = f.readline()
l1 = []
l2 = []
l3 = []
n = float(sys.argv[2])
while l:
    tok = l.split("\t")
    l1.append(tok[0].strip())
    l2.append(tok[1].strip())
    l3.append("{0:.2f}".format(round((n/float(tok[1])), 2)))
    l = f.readline()

s1="No. of processors & "
s2="Training time (seconds) & "
s3="Speed-up w.r.t serial code & "
for i in range(len(l1)-1, -1, -1):
    s1 += l1[i]+" & "
    s2 += l2[i]+" & "
    s3 += l3[i]+" & "

print s1
print s2
print s3
