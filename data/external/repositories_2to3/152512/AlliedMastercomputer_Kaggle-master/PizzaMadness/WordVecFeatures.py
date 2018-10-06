from PizzaManager import *

trueVectorsV    = []
falseVectorsV   = []
diffVectorsV    = []
trueVectorsN    = []
falseVectorsN   = []
diffVectorsN    = []
model           = None

def similarity_vec(w1,w2):
    return np.dot(matutils.unitvec(w1),matutils.unitvec(w2))

def n_similarity_vec(ws1,ws2):
    v1 = [model[word] for word in ws1]
    v2 = ws2
    return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))

def getVerbProjection(string,verbVector,model,cutoff=3):
    rToken = nltk.word_tokenize(string[cutoff:])
    rToken[:] = [x for x in rToken if x in model.vocab]
    tPOS = nltk.pos_tag(rToken)
    verbs = [x[0] for x in tPOS if 'VB' in x[1]]
    if not verbs:
        return 0.0
    else:
        return n_similarity_vec(verbs,[verbVector])*10
def getNounProjection(string,nounVector,model,cutoff=3):
    rToken = nltk.word_tokenize(string[cutoff:])
    rToken[:] = [x for x in rToken if x in model.vocab]
    tPOS = nltk.pos_tag(rToken)
    nouns = [x[0] for x in tPOS if 'NN' in x[1]]
    if not nouns:
        return 0.0
    else:
        return n_similarity_vec(nouns,[nounVector])*10

def loadModel(modelFile = 'GoogleNews-vectors-negative300.bin'):
    model = Word2Vec.load_word2vec_format(modelFile, binary=True)

def loadEVecs(df,numVecs=20):
# Get text
    trueText = df.query('requester_received_pizza == True').request_text_edit_aware.values
    falseText = df.query('requester_received_pizza == False').request_text_edit_aware.values
# Extract nouns and verbs
    requestTokenTrue = nltk.word_tokenize(' '.join(trueText))
    requestTokenTrue[:] = [x for x in requestTokenTrue if x in model.vocab]
    taggedPOSTrue = nltk.pos_tag(requestTokenTrue)
    verbsTrue = [x[0] for x in taggedPOSTrue if 'VB' in x[1]]
    nounsTrue = [x[0] for x in taggedPOSTrue if 'NN' in x[1]]
    requestTokenFalse = nltk.word_tokenize(' '.join(falseText))
    requestTokenFalse[:] = [x for x in requestTokenFalse if x in model.vocab]
    taggedPOSFalse = nltk.pos_tag(requestTokenFalse)
    verbsFalse = [x[0] for x in taggedPOSFalse if 'VB' in x[1]]
    nounsFalse = [x[0] for x in taggedPOSFalse if 'NN' in x[1]]
# Get frequency of nouns and verbs for received_pizza T,F
    nouncountsTrue = Counter(nounsTrue)
    total = sum(list(nouncountsTrue.values()), 0.0)/100
    for key in nouncountsTrue:
        nouncountsTrue[key] /= total
    nouncountsFalse = Counter(nounsFalse)
    total = sum(list(nouncountsFalse.values()), 0.0)/100
    for key in nouncountsFalse:
        nouncountsFalse[key] /= total
    countsTrue = Counter(verbsTrue)
    total = sum(list(countsTrue.values()), 0.0)/100
    for key in countsTrue:
        countsTrue[key] /= total
    countsFalse = Counter(verbsFalse)
    total = sum(list(countsFalse.values()), 0.0)/100
    for key in countsFalse:
        countsFalse[key] /= total
# Get diff of frequency between T and F
    verbcountsDiff = copy.deepcopy(countsTrue)
    verbcountsDiff.subtract(countsFalse)
    nouncountsDiff = copy.deepcopy(nouncountsTrue)
    nouncountsDiff.subtract(nouncountsFalse)
# Ignore first few boring verbs
    ignoreVerbs = [x[0] for x in verbcountsTotal.most_common(13)]
# Form matrix of word vectors scaled by frequencies to svd
    listFullTrueVerbs = []
    for item in [item for item in countsTrue.most_common(1000) if item[0] not in ignoreVerbs]:
        listFullTrueVerbs.append(model[item[0]]*item[1]*10)
    arrayFullTrueVerbs=(np.vstack(listFullTrueVerbs)).T
    listFullFalseVerbs = []
    for item in [item for item in countsFalse.most_common(1000) if item[0] not in ignoreVerbs]:
        listFullFalseVerbs.append(model[item[0]]*item[1]*10)
    arrayFullFalseVerbs=(np.vstack(listFullFalseVerbs)).T
    # Note that for diff, we want 1000 off of either end (largest in absval)
    listFullDiffVerbs = []
    tempnum = len(list(verbcountsDiff.values()))
    for word in [item[0] for item in verbcountsDiff.most_common(tempnum)[0:500]]:
        listFullDiffVerbs.append(model[word])
    for word in [item[0] for item in verbcountsDiff.most_common(tempnum)[-500:]]:
        listFullDiffVerbs.append(model[word])    
    arrayFullDiffVerbs=(np.vstack(listFullDiffVerbs)).T
    listFullTrueNouns = []
    for item in [item for item in nouncountsTrue.most_common(1000)]:
        listFullTrueNouns.append(model[item[0]]*item[1]*10)
    arrayFullTrueNouns=(np.vstack(listFullTrueNouns)).T
    listFullFalseNouns = []
    for item in [item for item in nouncountsFalse.most_common(1000)]:
        listFullFalseNouns.append(model[item[0]]*item[1]*10)
    arrayFullFalseNouns=(np.vstack(listFullFalseNouns)).T
    listFullDiffNouns = []
    tempnum = len(list(nouncountsDiff.values()))
    for item in [item for item in nouncountsDiff.most_common(tempnum)[0:500]]:
        listFullDiffNouns.append(model[item[0]]*item[1]*10)
    for item in [item for item in nouncountsDiff.most_common(tempnum)[-500:]]:
        listFullDiffNouns.append(model[item[0]]*item[1]*10)
    arrayFullDiffNouns=(np.vstack(listFullDiffNouns)).T
# Finally the SVD
    (uTrueV,sTrueV,vTrueV) = np.linalg.svd(arrayFullTrueVerbs,full_matrices=False)
    (uFalseV,sFalseV,vFalseV) = np.linalg.svd(arrayFullFalseVerbs,full_matrices=False)
    (uDiffV,sDiffV,vDiffV) = np.linalg.svd(arrayFullDiffVerbs,full_matrices=False)
    (uTrueN,sTrueN,vTrueN) = np.linalg.svd(arrayFullTrueNouns,full_matrices=False)
    (uFalseN,sFalseN,vFalseN) = np.linalg.svd(arrayFullFalseNouns,full_matrices=False)
    (uDiffN,sDiffN,vDiffN) = np.linalg.svd(arrayFullDiffNouns,full_matrices=False)
# and save the EVecs
    trueVectorsV = uTrueV[:,0:numVecs]
    falseVectorsV = uFalseV[:,0:numVecs]
    diffVectorsV = uDiffV[:,0:numVecs]
    trueVectorsN = uTrueN[:,0:numVecs]
    falseVectorsN = uFalseN[:,0:numVecs]
    diffVectorsN = uDiffN[:,0:numVecs]

def addWordVecFeatures(df,numVecs=len(trueVectorsV),vecStart=0):
    for i in range(vecStart,numVecs):
        df["evec_true_verb_"+str(i)] = df.total_text.apply(lambda x: getVerbProjection(x,trueVectorsV[:,i],model))
    print("Finished with evec_true_verb")
    for i in range(vecStart,numVecs):
        df["evec_false_verb_"+str(i)] = df.total_text.apply(lambda x: getVerbProjection(x,falseVectorsV[:,i],model))
    print("Finished with evec_false_verb")
    for i in range(vecStart,numVecs):
        df["evec_diff_verb_"+str(i)] = df.total_text.apply(lambda x: getVerbProjection(x,diffVectorsV[:,i],model))
    print("Finished with evec_diff_verb")
    for i in range(vecStart,numVecs):
        df["evec_true_noun_"+str(i)] = df.total_text.apply(lambda x: getNounProjection(x,trueVectorsN[:,i],model))
    print("Finished with evec_true_noun")
    for i in range(vecStart,numVecs):
        df["evec_false_noun_"+str(i)] = df.total_text.apply(lambda x: getNounProjection(x,falseVectorsN[:,i],model))
    print("Finished with evec_false_noun")
    for i in range(vecStart,numVecs):
        df["evec_diff_noun_"+str(i)] = df.total_text.apply(lambda x: getNounProjection(x,diffVectorsN[:,i],model))
    print("Finished with evec_diff_noun")
