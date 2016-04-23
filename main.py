#----------
# build the dataset
#----------
from __future__ import print_function
from TextCorrection import remove_accents
from pybrain.datasets import SupervisedDataSet
from pybrain.optimization.populationbased.ga import GA
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader


norm = 42
txt = open("napoleon.txt")

content = txt.read()
content = remove_accents(content)
content = content.lower()

'''
g5 g4 g3 g2 g1 d1 d2 d3 d4 d5
            ^
            ptr
'''

taille = len(content)


# apprentiddage
g5 = []
g4 = []
g3 = []
g2 = []
g1 = []
d1 = []

G5 = []
G4 = []
G3 = []
G2 = []
G1 = []
D1 = []

for i in range(127):
    g5.append(0)
    g4.append(0)
    g3.append(0)
    g2.append(0)
    g1.append(0)
    d1.append(0)

G1.append(d1)

G2.append(d1)
G2.append(d1)

G3.append(d1)
G3.append(d1)
G3.append(d1)

G4.append(d1)
G4.append(d1)
G4.append(d1)
G4.append(d1)

G5.append(d1)
G5.append(d1)
G5.append(d1)
G5.append(d1)
G5.append(d1)

oldValeurAscii = 0
for i in range(taille):
    valeurAscii = ord(content[i])
    if valeurAscii < 128:
        d1[oldValeurAscii] = 0
        d1[valeurAscii] = 1

        D1.append(d1)

        if i > 4:
            g5[valeurAscii] = 1
            g5[oldValeurAscii] = 0
            G5.append(g5)

        if i > 3:
            g4[valeurAscii] = 1
            g4[oldValeurAscii] = 0
            G4.append(g4)

        if i > 2:
            g3[valeurAscii] = 1
            g3[oldValeurAscii] = 0
            G3.append(g2)

        if i > 1:
            g2[valeurAscii] = 1
            g2[oldValeurAscii] = 0
            G2.append(g2)

        if i > 0:
            g1[valeurAscii] = 1
            g1[oldValeurAscii] = 0
            G1.append(g1)

        oldValeurAscii = valeurAscii

for i in range(127):
    g2[i] = 0

D1.append(g2)


ds = SupervisedDataSet(635, 127)
for v1, v2, v3, v4, v5, v6 in zip(G5, G4, G3, G2, G1, D1):
    t1 = v1 + v2 +v3 +v4 +v5
    t2 = v6
    ds.addSample(t1, t2)

print(len(ds))
#----------
# build the network
#----------
print("build network ...")
net = buildNetwork(635,
                   50, # number of hidden units
                   50, # number of hidden units
                   50, # number of hidden units
                   50, # number of hidden units
                   127,
                   bias=True,
                   hiddenclass=SigmoidLayer,
                   outclass=LinearLayer
                   )
'''
net2 = buildNetwork(254,
                   80, # number of hidden units
                   127,
                   bias=True,
                   hiddenclass=SigmoidLayer,
                   outclass=LinearLayer
                   )'''
#----------
# train
#----------

print("Training ...")
trainer = BackpropTrainer(net, ds, verbose=True, momentum=0.99)
trainer.trainUntilConvergence(maxEpochs=3, validationProportion=0.25)
NetworkWriter.writeToFile(net, 'napoleon_v7_vectors.xml')

'''
print("Training ...")
ga = GA(ds.evaluateModuleMSE, net2, minimize=True)
for i in range(100):
    print("Loop : ", i)
    net2 = ga.learn(0)[0]

NetworkWriter.writeToFile(net2, 'napoleon_ga_v5vectors.xml')'''

#net = NetworkReader.readFrom('napoleon_v6_vectors.xml')

#----------
# evaluate
#----------

letter = []

for i in range(127):
    letter.append(0)

letter[ord('e')] = 1
letter1 = letter

letter[ord('e')] = 0
letter[ord('s')] = 1
letter2 = letter

letter[ord('s')] = 0
letter[ord('t')] = 1
letter3 = letter

letter[ord('t')] = 0
letter[ord(' ')] = 1
letter4 = letter

letter[ord(' ')] = 0
letter[ord('u')] = 1
letter5 = letter

oldAsciiValue = ord('u')

for i in range(10):

    res = net.activate(letter1 + letter2 + letter3 + letter4 + letter5).tolist()
    plt.plot(res)
    asciiValue = int(res.index(max(res)))
    current = chr(asciiValue)
    print(current, end='')

    letter[oldAsciiValue] = 0
    letter[asciiValue] = 1

    letter1 = letter2
    letter2 = letter3
    letter3 = letter4
    letter4 = letter5
    letter5 = letter

    oldAsciiValue = asciiValue

letter[oldAsciiValue] = 0
letter[ord('t')]
test = letter

plt.plot(test, color='green')

plt.show()
print("Fin !")



'''
# neural net approximation
plt.plot(xvalues,
           [ net.activate([x])[0] for x in xvalues ], linewidth = 2,
           color = 'blue', label = 'NN output sin')

plt.plot(xvalues,
           [ net.activate([x])[1] for x in xvalues ], linewidth = 2,
           color = 'green', label = 'NN output cos')

# target function
plt.plot(xvalues,
           y_sin_values, linewidth = 2, color = 'red', label = 'target sin')

# target function
plt.plot(xvalues,
           y_cos_values, linewidth = 2, color = 'black', label = 'target cos')

plt.grid()
plt.legend()
plt.show()'''