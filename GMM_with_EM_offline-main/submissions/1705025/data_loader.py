import numpy as np

class DataLoader:

    def __init__(self, path):
        self.path = path
        #check the extension of the file
        if path.endswith('.txt'):
            self.loadFromText()



    def loadFromText(self):
        #load data from text file
        #get the dimension which the number of data in a line
        #get the number of data in a line
        with open(self.path, 'r') as f:
            line = f.readline()
            self.dim = len(line.split())


        #save the datapoints in a numpy array
        self.data = np.loadtxt(self.path, delimiter=' ', usecols=range(self.dim))

        

    
    def getData(self):
        return self.data, self.dim