#Implement training of our AI which will consist of brain and dqn in environment 

#Importing libraries
import os
import numpy as np
import random as rn
import environment
import dqn
import brain

#Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#Setting the parameters

epsilon  = 0.3  #Exploration , '0.3' is 30%
number_actions = 5
direction_boundary = (number_actions - 1) / 2 
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

#BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0,24.0),initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#BUILDING THE BRAIN BY SIMPLY CREATING AN OBJECT OF THE BRAIN CLASS
brain = brain.brain(learning_rate =0.00001,number_actions = number_actions)

#BUILDING THE DQN BY SIMPLY CREATING AN OBJECT OF THE DQN CLASS
dqn = dqn.DQN(max_memory = max_memory, discount_factor =0.9)

#CHOOSING THE MODE
train = True

#TRAINING THE AI
env.train = train
model = brain.model
if(env.train == True):
    #Starting the loop all over epochs (1 epoch = 5 months)
    for epoch in range(1,number_epochs):
        #INITIALIAZING ALL THE VARIABLES OF BOTH THE ENVIRONMENT AND THE TRAINING LOOP
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0,12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        #STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
        while((not game_over) and timestep<= 5*30*24*60): #5*30*24*60 is total number of minutes in 5 months
            # PLAYING THE NEXT ACTION BY EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0,number_actions)
                if(action - direction_boundary < 0 ):
                    direction = -1              #-1 when ai cools down the server, +1 when ai heats up the server
                else:
                    direction= 1
                energy_ai = abs(action - direction_boundary)  +  temperature_step
            
            #PLAYING THE NEXT ACTION BY INFERENCE
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            #UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            next_state , reward , game_over = env.update_env(direction,energy_ai,int(timestep/(30*24*60)))
            total_reward += reward
            
            #STORING THIS NEW TRANSITION INTO THE MEMORY
            dqn.remember([current_state,action,reward,next_state],game_over)
            
            #GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
            inputs,targets = dqn.get_batch(model,batch_size= batch_size)
            
            #COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
        
        #PRINTING THE RESULTS FOR EACH EPOCH
        print('\n')
        print("Epoch : {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        
        #SAVING OUR MODEL
        model.save('model.h5')