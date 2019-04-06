import tensorflow as tf
import numpy as np
import retro #plat form to get the game
from skimage import transform #to handel frames
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
import random
import warning  # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')



#create the environment

env=retro.make(game='SpaceInvaders-Atari2600')
print("The size of our frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)


#we crete one hot encoder of our actions to feed  to the neural network
possible_action=np.array(np.identity(env.action_space.n,dtype=int).tolist())

#preprocess the frame(grey_scale,crop the screen,normalize pixels)
def preprocess_frame(frame):
    gray=rgb2gray(frame)

    #crop the screen
    cropped_frame=gray[8:-12,4:-12]

    #normalize pixel values
    normalized_frame=cropped_frame/255.0

    #resize the frame
    preprocessed_frame=transform.resize(normalized_frame,[110,84])




#after 4 frame we stike frame
stack_size=4  #number of frames

#clear the frames
stacked_frames=deque([np.zeros((110,84),dtype=np.int) for i in range(stack_size)],maxlen=4)
def stack_frames(stacked_frames,state,is_new_episode):
    #preprocess the frame
    frame=preprocess_frame(state)

    #if we are in new episode we should crop the frame 4 time
    if is_new_episode:
      stacked_frames.append(frame)
      stacked_frames.append(frame)
      stacked_frames.append(frame)
      stacked_frames.append(frame)

      #stack the frames
      stacked_state=np.stack(stacked_frames,axis=2)
    else:
        # append fram to the queue
          stacked_frames.append(frame)
        #then stacked them together
    stacked_state=np.stack(stacked_frames,axis=2)

    return stacked_state, stacked_frames



#set the neural network hypreparameters
state_size=[110,84,4]  #(width,hight,4 stacked_frame)
stacked_frames=4
learning_rate=0.00025
action_size=env.action_space.n    #know the number of possible action

#triannig hyper parameter
total_episodes=50
max_steps=50000
batch_size=64

#exploration parameters
explore_start=1.0
explore_end=0.01
decay_rate=0.00001

#q_learning parameters
gamma=0.9


#we will create some thing like folder to save experence
pretrained_lenght=batch_size
memory_size=1000000


training = True
episode_render = True


#create deep learning network

class DQNetwork:
    def __init__(self,state_size,action_size,learning_rate,name='DQNetwork'):
        self.state_size=state_size
        self.action_size=action_size
        self.learning_rate=learning_rate


        #to share varibals betwwen layers that we have in dqlearning
        with tf.variable_scope(name):
            #we use placeholder to convert state_size,action_size to the type that cnn can work with
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            #we need to convert q_target to calculate the loss
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q=tf.placeholder(tf.float32,[None],name='target')


            #set the cnn
            #first layer
            #kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d use to implement the weights
            self.conv1=tf.layers.conv2d(inputs=self.inputs_,filters=32,kernel_size=[8,8],strides=[4,4],padding='VALID',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),name='conv1')
            self.conv1_out=tf.nn.elu(self.conv1,name='conv1_out')

            #second layer
            self.conv2 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding='VALID',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            self.conv2_out = tf.nn.elu(self.conv1, name='conv2_out')

            #3d layer
            self.conv3 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding='VALID',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3')
            self.conv3_out = tf.nn.elu(self.conv3, name='conv3_out')

            self.flatten=tf.contrib.layers.flatten(self.conv3_out)
            self.fc=tf.layers.dense(inputs=self.flatten,units=512,activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='fc1')
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)
            #our predicted Q
            self.Q=tf.reduce_sum(tf.multiply(self.output,self.actions_))

            #calculate the loss :The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss=tf.reduce_mean(tf.square(self.target_Q-self.Q))
            self.optimizer=tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)

            tf.reset_default_graph()
            DQNetwork = DQNetwork(state_size, action_size, learning_rate)


#as we said we will make memory for the network to help it to remember
class memory():
    def__init__(self,max_size):
    self.buffer=deque(maxlen=max_size)

    #add new experence to the memory
    # add new experence to the memory
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        buffer_size=len(self.buffer)
        index=np.random.choice(np.arange(buffer_size),size=batch_size,replace=False)
        return [self.buffer[i] for i in index]

#fix the empty memory problem,we will take a random action then stored the result in memory

memory = Memory(max_size = memory_size)
for i in range (pretrained_lenght):
    #if this the firest step we will go to env.reset then staked frames
    if i==0:
        state=env.reset
        state,stacked_frames=stack_frames(stacked_frames,state,True)
#if we finish for i=0 ,take a random action then save the result
#get the next state,the rewards

choice=random.randint(1,len(possible_action))-1   #take random action from all the possible action
action=possible_action[choice]  #take this action feed it to the function (possible_action)to convert it to one hot encoder to feed it the network
next_state, reward, done, _ = env.step(action)

#if we finished we will stake the frame again

#if the epsiode finished after stacked 4 times
if done:
    next_state=np.zeros(state.shape)   #if we finished
    memory.add(state,action,reward,next_state)  #we add experence to the memory then start new episode
    state=env.reset
    state,stacked_frames=stack_frames(stacked_frames,state,True)

else:
    memory.add(state, action, reward, next_state)  # we add experence to the memory then start new episode

    state=next_state


writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


#now we will trin the model.we will divided to two parts one we take actions and collext experence to the memory
#second part we will use this experence to train the model

#first part of training
def predict_action(explore_start,explore_end,decay_rate,decay_stop,state,actions):
    #first we choose a random action
    exp_exp_tradeoff=np.ramdom.rand()
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    #we use it to know if we are in exploration or exploitation
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability>exp_exp_tradeoff):
        #make a random action
         choice=random.randint(1,len(possible_action))-1
        action=possible_action[choice]
    else:
    #take action and see the result from q net to see what is the biggest q for action the take the action that givr biggest q
         Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

    choice=np.argmax(Qs)
    action=possible_action[choice]  #take the action that have the biggest q


  return action, explore_probability





#sec part (traing1)
#saver help us to save the model
saver=tf.train.saver()


if training==True:
    #initilize variable
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #initslize the decay rate
        decay_step=0
        for episode in range(total_episodes):
            step=0

            #init the reward
            episode_rewards=[]

            #make new episode and new state
            state=env.reset()
            #we staked 4 state becouse we are in new episode
            state,stacked_frames=stack_frames(stacked_frames,state,True)

            while step <max_steps:
                step+=1
                decay_step+=1

                #predict action to take
                action,explore_probability=predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                #perform the action and take the result
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()


                #add reward to total reward
                episode_rewards.append(reward)

                if done:  #the game finished
                    next_state=np.zeros((110,84),dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    #get rhe total reward for the epo
                    total_reward=np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {:.4f}'.format(loss))
                    reward_list.append((episode,total_reward))
                    #store in memory
                    memory.add((state,action,reward,next_state,done))

                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))

                    state=next_state



                #learning part
                #take what we do in (traing1) from memory usee it ti train the model
                batch=memory.sample(batch_size)  #take a random batch
                states_mb=np.array([each[0]for each in batch],ndim=3) #if we day 3 time game over
                action_mb=np.array([each[1]for each in batch])
                reward_mb = np.array([each[2] for each in batch])
                next_state_mb = np.array([each[3] for each in batch])
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []


                #get q for next state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0,len(batch)):
                    terminal = dones_mb[i]
                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(reward_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                       feed_dict={DQNetwork.inputs_: states_mb,
                                                  DQNetwork.target_Q: targets_mb,
                                                  DQNetwork.actions_: actions_mb})

                    summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                            DQNetwork.target_Q: targets_mb,
                                                            DQNetwork.actions_: actions_mb})
                    writer.add_summary(summary, episode)
                    writer.flush()

                    # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")







#test model
with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(1):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    env.close()

