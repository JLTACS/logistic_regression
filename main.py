import getData
import logistic_regresion

data = getData.prepareData('Albury')
train,test = getData.separateDataTrainTest(data,75)

xtrain,ytrain,xtest,ytest = getData.separateDataXY(train,test)

W,b,train_costs = logistic_regresion.training(xtrain,ytrain)

pYtest = logistic_regresion.logistic_regression(xtest,ytest,W,b)

print(pYtest)