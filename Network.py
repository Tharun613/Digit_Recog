import json
import numpy as np

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Network():

    def __init__(self,sizes,biases=None,weights = None):
        '''
            sizes has the size of each layer
            biases has the bias value for the (i+2)th layer in the ith index
            weights has the weight matrix connecting the (i+1)th and (i+2)th layers
        '''
        self.sizes = sizes
        self.nlayers = len(sizes)
        if biases == None:
            self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        else:
            self.biases = biases
        if weights == None:
            self.weights = [np.random.randn(x,y)  for (x,y) in zip(sizes[1:],sizes)]
        else:
            self.weights = weights

    def saveNetwork(self,filename):
        with open(filename,'w') as f:
            data = {
                "sizes": self.sizes,
                "biases": [b.tolist() for b in self.biases],
                "weights": [w.tolist() for w in self.weights]

            }
            json.dump(data,f)

    @classmethod
    def loadNetwork(filename):
        with open(filename,'r') as f:
            data = json.load(filename)
            net  = Network(data['sizes'])
            net.weights = [np.array(w) for w in data['weights']]
            net.biases = [np.array(b) for b in data['biases']]
        return net


    def printSelf(self):
        print(f"number of layers : {self.nlayers}")
        print("Printing biases...")
        for row in self.biases:
            print(row)
        print("Printing weights")
        for i,matrix in enumerate(self.weights):
            print("Matrix " + str(i))
            for row in matrix:
                print(row)

    def output(self,inputValues):
        '''
            input is the values of the input node

            output is the value of the output layer

        '''
        currentValue = inputValues
        for i in range(self.nlayers-1):
            nextValue = np.dot(self.weights[i],currentValue)
            nextValue = nextValue + self.biases[i]
            nextValue = sigmoid(nextValue)
            currentValue = nextValue
        return nextValue

    def compute(self, inputValues):
        '''
           input is the values of the input layer

           output is the activation value of all the layers excluding the input
        '''
        currentValue = inputValues
        nodeOutputs = []
        zs = []
        #nodeOutputs.append(inputValues)
        for i in range(self.nlayers - 1):
            nextValue = np.dot(self.weights[i],currentValue)
            # nextValue = np.add(nextValue,self.biases[i].reshape(len(self.biases[i]),1))
            nextValue = nextValue + self.biases[i]
            zs.append(nextValue)
            nextValue = sigmoid(nextValue)
            currentValue = nextValue
            nodeOutputs.append(currentValue)
        return nodeOutputs,zs


    def SGD(self,training_set,mini_batch_size,learning_rate,epochs,test_data = None):
        '''
            in each epoch
                training data is split into mini batches
                each mini batch is used to adjust the weights
            Algorithm:
            for each examples in a mini batch
                find the acitvation values of all the nodes.  --> using compute function
                find the delta value of all the nodes ---> using backprop function

            wl --> denotes the weight matrix from the l-1 th layer to the lth layer
            al ---> denotes the activation values of the lth layer
            zl ---> denotes the input to the nodes in the lth layer
            bl ---> denotes the biases of the (l)th layer
            deltal --> denotes the error in the nodes of the lth layer

            DELTAL --> denotes the error in the nodes of the (l+2)th layer
            AL ---> denotes the activation values of the (l+2)th layer
            ZL ---> denotes the input to the nodes in the (l+2)th layer
            BL ---> denotes the biases of the (l+2)th layer
            Wl  --> denotes the weight matrix from the l+1th layer to the l+2th layer



        '''
        print("SGD called")
        print(f"mini_batch_size : {mini_batch_size}")
        print(f"learning rate : {learning_rate}")
        print(f"Epochs : {epochs}")

        n = len(training_set)

        for i in range(epochs):
            random.shuffle(training_set)
            mini_batches = [training_set[k:k+mini_batch_size]   for k in range(0,n,mini_batch_size)]
            for j,mini_batch in enumerate(mini_batches):
                self.update(mini_batch,learning_rate)
                #print(f"mini_batch {j} completed")
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {i} completed")

    def update(self,mini_batch,learning_rate):
        '''

             until you exhaust all the examples
                call compute
                call backprop
                compute the change matrix for weights DELTA(L-2)
                compute the change matrix for biases

        '''
        mini_batch_size = len(mini_batch)
        biasChange = [np.zeros(b.shape)  for b in self.biases] # of size nlayers-1
        weightChange = [np.zeros(w.shape) for w in self.weights] # of size nlayers-1
        for example in mini_batch:
            inputValue,outputValue = example
            nodeOutputs,zs = self.compute(inputValue)
            deltaMatrix = self.backprop(zs,nodeOutputs,outputValue)

            for i in range(self.nlayers-1):
                if i == 0:
                    weightChange[i]  = np.dot(deltaMatrix[i],np.transpose(inputValue)) + weightChange[i]
                else:
                    weightChange[i] = np.dot(deltaMatrix[i], np.transpose(nodeOutputs[i-1])) + weightChange[i]
                biasChange[i] = biasChange[i] + deltaMatrix[i]
        for i in range(self.nlayers-1):
            coefficient = learning_rate / mini_batch_size
            self.weights[i] = self.weights[i] - (coefficient * weightChange[i])
            self.biases[i]  = self.biases[i]  - (coefficient * biasChange[i])

    def backprop(self,zs,nodeOutputs,outputValue):

        derivative = self.computeDerivative(nodeOutputs[-1],outputValue)
        sigvalue = sigmoid_prime(zs[-1])
        delta_L = derivative * sigvalue
        delta_matrix = []
        delta_matrix.append(delta_L)
        for x in range(self.nlayers-2,0,-1):
            newValue = np.dot(np.transpose(self.weights[x]),delta_matrix[-1])
            newValue = newValue * sigmoid_prime(zs[x-1])
            delta_matrix.append(newValue)
        return delta_matrix[::-1]
    def evaluate(self,test_data):
        #print("Evaluating")

        total_correct = 0
        for i,(x,y) in enumerate(test_data):
            a = np.argmax(self.output(x))
            if a==y:
                total_correct = total_correct +1
        return total_correct
    def computeDerivative(self,obtaineOutput,ActualOutput):
        return obtaineOutput - ActualOutput

