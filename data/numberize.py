
def numberizeNERlabels():
	f = open('ned.train.labels')
	y = [x.rstrip() for x in list(f)]
	f.close()
	labels = set(y)
	labels.remove('ZERO')
	labels = list(labels)
	numbers = dict()
	for i in range(len(labels)):
		numbers[labels[i]] = i+1
	numbers['ZERO'] = 0
	print labels
	print len(labels)
	print numbers

	fout = open('ned.train.labels.numbers','w')
	for label in y:
		fout.write(str(numbers[label])+'\n')
	fout.close()

	f = open('ned.testa.labels')
	y = [x.rstrip() for x in list(f)]
	f.close()
	fout = open('ned.testa.labels.numbers','w')
	for label in y:
		fout.write(str(numbers[label])+'\n')
	fout.close()

	f = open('ned.testb.labels')
	y = [x.rstrip() for x in list(f)]
	f.close()
	fout = open('ned.testb.labels.numbers','w')
	for label in y:
		fout.write(str(numbers[label])+'\n')
	fout.close()
	

def numberizeWSJlabels():
	f = open('wsj.train.labels')
	y = [x.rstrip() for x in list(f)]
	f.close()
	labels = set(y)
	labels.remove('ZERO')
	labels = list(labels)
	numbers = dict()
	for i in range(len(labels)):
		numbers[labels[i]] = i+1
	numbers['ZERO'] = 0
	print labels
	print len(labels)
	print numbers

	fout = open('wsj.train.labels.numbers','w')
	for label in y:
		fout.write(str(numbers[label])+'\n')
	fout.close()

	f = open('wsj.dev.labels')
	y = [x.rstrip() for x in list(f)]
	f.close()
	fout = open('wsj.dev.labels.numbers','w')
	for label in y:
		fout.write(str(numbers[label])+'\n')
	fout.close()

	f = open('wsj.test.labels')
	y = [x.rstrip() for x in list(f)]
	f.close()
	fout = open('wsj.test.labels.numbers','w')
	for label in y:
		fout.write(str(numbers[label])+'\n')
	fout.close()

def uniqueNERfeatures():
	print "Loading..."
	f = open('ned.train.feats.dat')
	examples = list(f)
	f.close()
	print "Making feature matrix..."
	features = [x.split() for x in examples]
	nExamples = len(features)
	nFeatures = len(features[0])
	numbers = []
	for j in range(nFeatures):
		print "Finding unique values for feature "+str(j)
		uniques = dict()
		for i in range(nExamples):
			f = int(features[i][j])
			if f in uniques:
				uniques[f] = uniques[f]+1
			else:
				uniques[f] = 1
		sorted = []
		for f in uniques.keys():
			if uniques[f] >= 1:
				sorted.append(f)
		sorted.sort()
		print "Number of unique keys: "+str(len(sorted))

		numbers.append(dict())
		for i in range(len(sorted)):
			numbers[j][sorted[i]] = str(i)

		for i in range(nExamples):
			if int(features[i][j]) in numbers[j]:
				features[i][j] = numbers[j][int(features[i][j])]
			else:
				features[i][j] = '0'

	f = open('ned.train.feats.unique.dat','w')
	for example in features:
		f.write(' '.join(example)+'\n')
	f.close()

	f = open('ned.testa.feats.dat')
	examples = list(f)
	f.close()
	features = [x.split() for x in examples]
	nExamples = len(features)
	for j in range(nFeatures):
		for i in range(nExamples):
			numbers[j]
			features[i][j]
			if int(features[i][j]) in numbers[j]:
				features[i][j] = numbers[j][int(features[i][j])]
			else:
				features[i][j] = '0'
	f = open('ned.testa.feats.unique.dat','w')
	for example in features:
		f.write(' '.join(example)+'\n')
	f.close()

	f = open('ned.testb.feats.dat')
	examples = list(f)
	f.close()
	features = [x.split() for x in examples]
	nExamples = len(features)
	for j in range(nFeatures):
		for i in range(nExamples):
			if int(features[i][j]) in numbers[j]:
				features[i][j] = numbers[j][int(features[i][j])]
			else:
				features[i][j] = '0'
	f = open('ned.testb.feats.unique.dat','w')
	for example in features:
		f.write(' '.join(example)+'\n')
	f.close()


def uniqueWSJfeatures():
	print "Loading..."
	f = open('wsj.train.feats.dat')
	examples = list(f)
	f.close()
	print "Making feature matrix..."
	features = [x.split() for x in examples]
	nExamples = len(features)
	nFeatures = len(features[0])
	numbers = []
	for j in range(nFeatures):
		print "Finding unique values for feature "+str(j)
		uniques = dict()
		for i in range(nExamples):
			f = int(features[i][j])
			if f in uniques:
				uniques[f] = uniques[f]+1
			else:
				uniques[f] = 1
		sorted = []
		for f in uniques.keys():
			if uniques[f] >= 3:
				sorted.append(f)
		sorted.sort()
		print "Number of unique keys: "+str(len(sorted))

		numbers.append(dict())
		for i in range(len(sorted)):
			numbers[j][sorted[i]] = str(i)

		for i in range(nExamples):
			if int(features[i][j]) in numbers[j]:
				features[i][j] = numbers[j][int(features[i][j])]
			else:
				features[i][j] = '0'

	f = open('wsj.train.feats.unique.dat','w')
	for example in features:
		f.write(' '.join(example)+'\n')
	f.close()

	f = open('wsj.dev.feats.dat')
	examples = list(f)
	f.close()
	features = [x.split() for x in examples]
	nExamples = len(features)
	for j in range(nFeatures):
		for i in range(nExamples):
			numbers[j]
			features[i][j]
			if int(features[i][j]) in numbers[j]:
				features[i][j] = numbers[j][int(features[i][j])]
			else:
				features[i][j] = '0'
	f = open('wsj.dev.feats.unique.dat','w')
	for example in features:
		f.write(' '.join(example)+'\n')
	f.close()

	f = open('wsj.test.feats.dat')
	examples = list(f)
	f.close()
	features = [x.split() for x in examples]
	nExamples = len(features)
	for j in range(nFeatures):
		for i in range(nExamples):
			if int(features[i][j]) in numbers[j]:
				features[i][j] = numbers[j][int(features[i][j])]
			else:
				features[i][j] = '0'
	f = open('wsj.test.feats.unique.dat','w')
	for example in features:
		f.write(' '.join(example)+'\n')
	f.close()

