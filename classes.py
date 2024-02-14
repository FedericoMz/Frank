import random

class User:
    def __init__(self, believe_level, rethink_level, fairness):
        self.believe_level = believe_level
        self.rethink_level = rethink_level
        self.fairness = fairness
        
    def believe(self):
        
        if self.believe_level > 1:
            return None
        else:
            return random.choices(population=[True, False], 
                              weights=[self.believe_level, 1-self.believe_level], k=1)[0]
    
    def rethink(self):
        return random.choices(population=[True, False], 
                              weights=[self.rethink_level, 1-self.rethink_level], k=1)[0]
    
    def fairness_percentage(self):
        return self.fairness
    
class GroundTruther(User):
    def __init__(self, believe_level, rethink_level, fairness, expertise):
        super().__init__(believe_level, rethink_level, fairness)
        self.expertise = expertise
        
    def predict(self, record, ground):

        return random.choices(population=[ground, not ground], 
                              weights=[self.expertise, 1-self.expertise], k=1)[0]

class ModelBased(User):
    def __init__(self, believe_level, rethink_level, fairness, model, X, Y):
        super().__init__(believe_level, rethink_level, fairness)
        self.model = model.fit(X, Y)
        
    def predict(self, record, ground):
        return self.model.predict([record])[0]
    
class Real():
        
    def predict(self):
        return input()
   
    def believe(self):
        return input()
    
    def rethink(self):
        return input()