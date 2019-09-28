import cPickle as pl 

data = pl.load(open("dump/data.pkl", 'rb'))
print "Data length", len(data)
obj = data[data.keys()[2]]
print(obj)
print(len(obj["text"][0]))
print ""
print("First sentence and parse : ", obj["text"][0], obj["dep"][0])
print("Last sentence and parse : ", obj["text"][-1], obj["dep"][-1])