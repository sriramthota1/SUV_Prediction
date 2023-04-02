import pickle
from flask import Flask, render_template, request
#model.pkl is trained ML model

#deserialize-read the binary file-ML model
from sklearn.preprocessing import StandardScaler

clf=pickle.load(open('mode.pkl','rb'))

#for getting range decided on xtrain
import pandas as pd
df=pd.read_csv("SUV_Purchase.csv")

df=df.drop(['User ID','Gender'],axis =1)

#step3: oading the data
#setting the data into inout and output values
X=df.iloc[:,:-1].values #iloc==>index location 2D array
Y=df.iloc[:,-1:].values #2D array

#step4: split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


app=Flask(__name__)

@app.route('/')#annotation triggers the methods
def hello():
    return render_template('index.html')

#
@app.route('/Predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()]
    print(features)
    sst=StandardScaler().fit(X_train)#range would be decided on X train
    output=clf.predict(sst.transform([features]))
    print(output)
    if output[0] == 0:
        return render_template('index.html', pred=f'The person will not be able to purchase the SUV')
    else:
        return render_template('index.html', pred=f'The person will be able to purchase the SUV')


#def index():
 # return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)