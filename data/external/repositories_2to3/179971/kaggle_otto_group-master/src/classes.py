class SingleModel:
    """ Individual category model object trained using numeric features """
    def __init__(self,type,model):
        self.type = type
        self.model = model
        self.auc = -1
        self.f1 = -1
        self.acc = -1
        self.logloss = 1e309
        self.grid = {}
    def __str__(self):
        from numpy import round
        s = "Results for the " + self.type + " model.\n"
        s += "\tAUC     : " + str(round(self.auc,4)) + "\n"
        s += "\tF1      : " + str(round(self.f1,4)) + "\n"
        s += "\tACC     : " + str(round(self.acc,4)) + "\n"
        s += "\tLog Loss: " + str(round(self.logloss,4)) + "\n"
        s += "\tGrid: " + str(self.grid)
        return s
    def updateModel(self,pred,fullpred,obs,model,vars,grid,optMetric='auc'):
        """ Update the model if it is better than previous models """
        from sklearn.metrics import roc_auc_score, f1_score, log_loss
        from numpy import round, mean		
        improve = False
        try:
	        AUC = roc_auc_score(obs,pred)
	except:
	    	AUC = 0
        F1 = f1_score(obs,round(pred))
        acc = mean(round(pred) == obs)
        logLoss = log_loss(obs, fullpred)
        #determine if the model is better than previous models
        if optMetric == 'auc':
            if AUC > self.auc:
                improve = True
        elif optMetric == 'f1':
            if F1 > self.f1:
                improve = True
        elif optMetric == "acc":
            if acc > self.acc:
                improve = True
        elif optMetric == "logloss":
            if logLoss < self.logloss:
                improve = True
        # update the model if it is the best
        if improve:
            self.auc = AUC
            self.f1 = F1
            self.acc = acc
            self.logloss = logLoss
            self.model = model
            self.grid = grid
            self.vars = vars
    def predict(self,dat):
        """ Make prediction using model """
        return self.model.predict_proba(dat)[:,1]


class MultiModel:
    """ Individual category model object trained using numeric features """
    def __init__(self,type,model):
        self.type = type
        self.model = model
        self.logloss = 1e309
    def __str__(self):
        from numpy import round
        s = "Results for the " + self.type + " model.\n"
        s += "\tLog Loss: " + str(round(self.logloss,4)) + "\n"
        s += "\tGrid: " + str(self.grid) + "\n"
        return s
    def updateModel(self,pred,obs,model,vars,grid):
        """ Update the model if it is better than previous models """
        from sklearn.metrics import roc_auc_score, f1_score, log_loss
        from numpy import round, mean		
        improve = False
        logLoss = log_loss(obs, pred)
        #determine if the model is better than previous models
        if logLoss < self.logloss:
            self.logloss = logLoss
            self.model = model
            self.grid = grid
            self.vars = vars
    def predict(self,dat):
        """ Make prediction using model """
        return self.model.predict_proba(dat)[:,1]


class AdjustVariable(object):	
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None  
	def __call__(self, nn, train_history):
		import numpy as np
		from helperFunctions import float32
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
		epoch = train_history[-1]['epoch']
		new_value = float32(self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


