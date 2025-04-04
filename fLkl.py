import numpy as np
import scipy.special
import dill
from flask import Flask, request, jsonify
class nn:
    def __init__(self,srd,zjd,scd,lrate):
        self.inodes=srd
        self.hnodes=zjd
        self.onodes=scd
        self.lr=lrate
        self.wih=np.random.rand(self.hnodes,self.inodes)-0.5
        self.who=np.random.rand(self.onodes,self.hnodes)-0.5
        self.activation_function=lambda	x:scipy.special.expit(x)
        pass
    
    def train(self, inputsss, targetsss):
        inputs = np.array(inputsss, ndmin=2).T
        targets = np.array(targetsss, ndmin=2).T
        hiddenins = np.dot(self.wih, inputs)
        hiddenous = self.activation_function(hiddenins)
        finalins = np.dot(self.who, hiddenous)
        finalous = self.activation_function(finalins)
        outputers = targets - finalous
        hiddeners = np.dot(self.who.T, outputers) 
        self.who += self.lr * np.dot((outputers * finalous * (1.0 - finalous)), np.transpose(hiddenous))
        self.wih += self.lr * np.dot((hiddeners * hiddenous * (1.0 - hiddenous)), np.transpose(inputs))
        pass
    def query(self,inputs):
        inputss=np.array(inputs,ndmin=2).T
        hiinputs=np.dot(self.wih,inputss)
        hioutputs=self.activation_function(hiinputs)
        finalin=np.dot(self.who,hioutputs)
        finalout=self.activation_function(finalin)
        return finalout
    def save(self, path):
        obj = dill.dumps(self)
        with open(path, "wb") as f:
            f.write(obj)
    def load(path):
        try:
            with open(path, "rb") as f:
                obj = dill.load(f)
            return obj
        except (FileNotFoundError, EOFError, dill.UnpicklingError) as e:
            print(f"Error loading file: {e}")
            return None
app = Flask(__name__)   
model=nn.load("threeminst.model")
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    inputs=(np.asfarray(data) / 255.0 * 0.99) + 0.01
    pdt=model.query(inputs)
    return jsonify({'ans': int(np.argmax(prediction))})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)