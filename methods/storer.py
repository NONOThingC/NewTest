import pickle

class Storer:
    def __init__(self, path):
        self.path = path
        
    def save(self, data):
        with open(self.path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)
        return data