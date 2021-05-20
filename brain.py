#Implements the artificial brain for our model

#Importing libraries

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#BUILDING THE BRAIN
class brain(object):
    def __init__(self,learning_rate =0.001,number_actions =5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x= Dense(units = 64,activation = 'sigmoid')(states)  #Added (states) to make connection between the 'x' hidden layer and the 'states' input layer
        y = Dense(units = 32,activation = 'sigmoid')(x)
        q_values = Dense(units = number_actions , activation = 'softmax')(y)
        self.model = Model(inputs = states,outputs=q_values)
        self.model.compile(loss = 'mse',optimizer = Adam(lr = learning_rate))