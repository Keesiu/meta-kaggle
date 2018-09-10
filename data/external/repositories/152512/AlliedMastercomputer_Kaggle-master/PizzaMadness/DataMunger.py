from PizzaManager import *

def getTrainingData():
    df = pd.read_json('train.json')
    removeList = list(df.columns[map((lambda x:'at_retrieval' in x),df.columns)])
    removeList.append(u'giver_username_if_known')
    removeList.append(u'post_was_edited')
    removeList.append(u'requester_user_flair')
    removeList.append(u'request_text')
    df.drop(removeList,axis=1,inplace=True)
    return df

def getTestData():
    testdf = pd.read_json('test.json')
    return testdf

def getNumericTraining(df=getTrainingData()):
    #df = getTrainingData()
    #numdf = df.drop([u'request_id', u'request_text_edit_aware', u'request_title', u'requester_subreddits_at_request', u'requester_username'],axis=1)
    numdf = df
    dropList = [u'total_text',u'request_id', u'request_text_edit_aware', u'request_title', u'requester_subreddits_at_request', u'requester_username',u'giver_username_if_known']
    for item in dropList:
        if item in numdf:
            numdf.drop(item,axis=1,inplace=True)
    cols = list(numdf)
    cols.insert(0, cols.pop(cols.index('requester_received_pizza')))
    numdf = numdf.ix[:, cols]
#    cols = numdf.columns.tolist()
#    cols = cols[-5:] + cols[:-5]
#    numdf = numdf[cols]
    trainData=numdf.values
    X_train=trainData[0::,1::].astype(float)
    y_train=trainData[0::,0].astype(int)
    return X_train,y_train

def getNumericTest(testdf=getTestData()):
    #testdf = getTestData()
    #numdfTest = testdf.drop([u'giver_username_if_known', u'request_id', u'request_text_edit_aware', u'request_title', u'requester_subreddits_at_request', u'requester_username'],axis=1)
    numdfTest = testdf
    dropList = [u'total_text',u'request_id', u'request_text_edit_aware', u'request_title', u'requester_subreddits_at_request', u'requester_username',u'giver_username_if_known']
    for item in dropList:
        if item in numdfTest:
            numdfTest.drop(item,axis=1,inplace=True)
#    cols = numdfTest.columns.tolist()
#    cols = cols[-4:] + cols[:-4]
#    numdfTest = numdfTest[cols]
    testData=numdfTest.values
    X_test=testData[0::,0::].astype(float)
    return X_test

def getTestCheat():
    testdf = getTestData()
    return (testdf.giver_username_if_known!='N/A').values
    

def writeSolution(soln,fname):
    ids = getTestData()['request_id'].values
    predictions_file = open(fname+".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["request_id","requester_received_pizza"])
    open_file_object.writerows(zip(ids, soln))
    predictions_file.close()

if __name__ == '__main__':
    print "hello"
    getTrainingData
