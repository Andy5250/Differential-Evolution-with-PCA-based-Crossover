


for i in xrange(5):
    data1 = open('D:/Users/zdz/cec13_10/dataOLD%d.txt'%i, 'r')
    data2 = open('D:/Users/zdz/cec13_10/dataPCA%d.txt'%i, 'r')

    print "old",i+1, data1.readlines()[-1]
    print "PCA",i+1, data2.readlines()[-1]

    print "###########################################################"
    data1.close()
    data2.close()
