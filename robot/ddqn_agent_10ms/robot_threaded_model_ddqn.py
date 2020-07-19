####
#Bot converted to start in room 200. Includes reset. This is the baseline ddqn.
#This bot stores and trains on a cumulative reward over seval episodes
#
####

import sys
import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['openmp'] = 'True'
import shutil
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import telnetlib
from datetime import datetime
from threading import Thread
import numpy as np
from numpy import array
from collections import deque
import random
import copy


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
##from keras import Input
##from keras.callbacks import TensorBoard
##from keras.engine import Model
# from keras.layers import LSTM, Dense, Embedding, Dot, Dropout, Activation, Flatten
# from keras.layers import MaxPooling1D, MaxPooling2D



# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from PyQt5.QtWidgets import QApplication, QWidget
import pickle
import time


# import math
# import random


class ModelThread(QtCore.QThread):

	def __init__(self, parent):  # allow passing parent into the eventthread object
		super(ModelThread, self).__init__(parent)
		# import weakref #https://stackoverflow.com/questions/10791588/getting-container-parent-object-from-within-python
		self.parent = parent
		self.initialize_model()

	def initialize_model(self):
		# initialize memory
		self.memory = deque(maxlen=100000)
		# self.state_multiplier = 16 # stack multiple states to create memory
		self.state_multiplier = 4  # stack multiple states to create memory
		self.batch_size = 4
		self.priority_fraction = 0.25
		# initialize model training parameters
		self.epsilon_start = 1
		self.epsilon = 1  # threshold for action to be random, will decay to .05
		self.epsilon_array = [self.epsilon]
		self.training_allowed = True
		# self.training_allowed=False
		self.epsilon_min = 0.05
		self.epsilon_decay = .999  # 0.993
		# discount for future rewards
		self.gamma = .125#0.99  # 0.99  # google uses 0.99 discount factor
		self.tau = .125  # 0.125 #1E-3 for soft update of target parameters
		self.learning_rate = 0.01#0025  # learning rate
	def final_startup(self):
		# create the model
		self.model = self.create_model()
		self.target_model = self.create_model()
		#self.parent.model = copy.deepcopy(self.model)

	def create_model(self):
		# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
		model = Sequential()
		multiplier = self.state_multiplier
		myinputdim = len(self.parent.state_array) * multiplier  # state array includes actions
		myoutputdim = len(self.parent.actiondict) #+ 1  # outputs are actions, add one extra for unrecognized actions
		print('input dims: ', myinputdim)

		model.add(Dense(2*myinputdim, input_dim=myinputdim, activation='relu'))
		#model.add(Dense(768, activation='relu'))
		model.add(Dense(myinputdim+myoutputdim+1, activation='relu'))
		#model.add(Dense(myinputdim + myoutputdim + 1, activation='relu'))
		#model.add(Dense(myinputdim + myoutputdim + 1, activation='relu'))

		model.add(Dense(myoutputdim, activation='linear')) #can try softmax too
		roptimizer = RMSprop(lr=self.learning_rate)
		model.compile(optimizer=roptimizer, loss='mean_squared_error')

		model.summary()
		# Visualize the model
		try:
			plot_model(model, to_file='model.png', show_shapes=True)
		except ImportError as e:
			print('couldnt print model to image', e)
		return model

	def train_model_old(self):
		print('training model')
		#self.thread.eventState = -1# set action loop to pause
		training_cycles = 5000
		for i in np.arange(training_cycles):
			self.replay()
			# print('try target_train')
			self.target_train()
		#self.thread.eventState = 0 # resume actions

	def get_prioritized_samples(self, memory, batch_size):

		pfraction = self.priority_fraction
		regular_memory = deque(maxlen=100000)
		prioritized_memory = deque(maxlen=100000)

		for i in np.arange(len(memory)-self.state_multiplier):
			state, action, reward_a, state2, done = memory[i]
			state_b, action_b, reward_b, state2_b, done_b = memory[i+self.state_multiplier] #make sure to skip ahead enough states
			#for multiplier 4, each state has 4 frames, but there is overlap states i+1 has frames 2:5, state i has frame 1:4
			#skip ahead by mult to get far enough away
			if reward_b - reward_a > 0: #positive reward change
				#if reward_a > 0 : insist on positive cumitave rewards only?
				prioritized_memory.append((state, action, reward_a, state2, done))
			elif reward_b - reward_a < 0: #negative reward
				regular_memory.append((state, action, reward_a, state2, done))
		#now get a sample from both regular and priotized memory using p_fraction
		if batch_size < 4:
			batch_size = 4


		rbatch = int(np.ceil(batch_size * (1-pfraction)))
		pbatch = int(np.floor(batch_size * pfraction))


		if len(prioritized_memory) > pbatch:
			#print('ok here')
			#print(rbatch,pbatch)
			rsamples = random.sample(regular_memory, rbatch)
			psamples = random.sample(prioritized_memory, pbatch)
		elif len(prioritized_memory) <= pbatch:
			rsamples = random.sample(regular_memory, rbatch)
			psamples = random.sample(regular_memory, pbatch)

		return rsamples, psamples



	def train_model(self):
		self.replay()
		self.target_train()

	def replay(self):
		try:
			#print('replay triggered')
			batch_size = self.batch_size
			if len(self.memory) < batch_size:
				print('memory too short', len(self.memory))
				return

			#rsamples = random.sample(self.memory, batch_size)
			#rsamples = random.sample(self.memory, batch_size)
			#psamples = random.sample(self.memory, batch_size)
			#print('prioritized samples')
			rsamples, psamples = self.get_prioritized_samples(self.memory, batch_size)
			#print('got prioritized samples')
			for rsample in rsamples:
				state, action, reward, new_state, done = rsample
				target = self.target_model.predict(state)
				#print(target)
				#print(max(target))
				#print(max(target[0]))
				if done:
					target[0][action] = reward
				else:
					Q_future = max(self.target_model.predict(new_state)[0])
					newval = reward + Q_future * self.gamma
					target[0][action] = newval
				self.model.fit(state, target, epochs=1, verbose=0)






			for psample in psamples:

				state, action, reward, new_state, done = psample
				target = self.target_model.predict(state)
				# print(target)
				# print(max(target))
				# print(max(target[0]))
				if done:
					target[0][action] = reward
				else:
					Q_future = max(self.target_model.predict(new_state)[0])
					newval = reward + Q_future * self.gamma
					target[0][action] = newval
				self.model.fit(state, target, epochs=1, verbose=0)
			#print('replay complete')
		except:
			print('------replay-error----')
			print(sys.exc_info()[0])
			print(sys.exc_info())

	def target_train(self):
		try:
			#print('target train called')
			weights = self.model.get_weights()
			#print('weights')
			target_weights = self.target_model.get_weights()
			#print('okk')
			for i in range(len(target_weights)):
				target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
			self.target_model.set_weights(target_weights)
			#print('target train complete')
		except:
			print('------target_train-error----')
			print(sys.exc_info()[0])
			print(sys.exc_info())

class EventThread(QtCore.QThread):
	mysignal = QtCore.pyqtSignal(str)  # sends out 1 string

	def __init__(self, parent):  # allow passing parent into the eventthread object
		super(EventThread, self).__init__(parent)
		# import weakref #https://stackoverflow.com/questions/10791588/getting-container-parent-object-from-within-python
		self.parent = parent
		# QtCore.QThread.__init__(self, parent)
		self.exiting = False
		self.action_from_parent = 'none'
		self.delta = 0
		self.eventState = -1
		self.pause_flag = False

	def begin(self):
		#self.start()
		print('Starting Event Loop (Robot Mind)')
	def pause(self):
		self.eventState = -1

	def unpause(self):
		self.eventState = 0
	def run(self):
		# main thread for event loop. not called directly, but runs
		# after thread gets setup (self.start()
		# actions = ['score', 'get all','kill rabbit','kill raccoon','wait']
		# exits = ['east', 'west', 'north', 'south']
		# for e in exits:
		#     actions.append('go '+e)
		try:
			self.eventState = -1  # start in off state. start bot button gets bot to idle state (0)
			self.ms_per_tick = 20
			#self.isWalking = 0
			#self.isFighting = 0  # initially not fighting
			# status:
			# global variables for event stuff
			# self.hp = 60
			# self.hplimit = 60
			# self.hpHazy = 15
			# self.mp = 5
			# self.monsterList = []  # initially no monsters in room
			# self.possibleTarget = 0  # no target
			# self.target = 0  # no target

			#self.max_steps = 1000000
			#self.steps = 0
			#self.train_interval = 10000
			#self.steps_until_train = self.train_interval  # try increment of 32 steps may be enough not sure

			# protocol for sending a signal to the main thread with command:
			# the main thread has a corresponding command to write to socket. check connection
			self.mysignal.connect(self.parent.msgFromEventLoop)
			# text = 'socket>*info \n\r\n\r'
			# self.mysignal.emit(text)
			# print('emitted without exception')
			# certain tags mark kind of output
			#  socket>* write to socket
			#  setRooms>* sets up path
			#  onestep>* takes a step

			# initialize timers
			self.logtimer = datetime.now()

			while self.exiting == False:
				if self.eventState == 99:
					print('quitting event Thread')
					self.exiting = True
				if self.eventState == 0:
					if self.eventState < 0:
						print('event state is',self.eventState)
					self.currenttimer = datetime.now()
					# limit actions every 1 second for now
					self.diff = self.currenttimer - self.logtimer
					self.delta = self.diff.total_seconds()*1000*(1/self.ms_per_tick) #convert to ms, then adjust
					if self.delta >= 2:
						# act = random.choice(actions) #do a random action once per tick
						# do action from parent
						act = self.action_from_parent
						if act == 'killrabbit':
							act = 'kill rabbit'
						elif act == 'killraccoon':
							act = 'kill raccoon'
						elif act == 'killgolem':
							act = 'kill golem'
						#self.steps_until_train = self.steps_until_train - 1
						if act != 'none':  # if none command do nothing
							cmd = 'socket>*' + act + ' \n'
							self.mysignal.emit(cmd)
							#self.steps = self.steps + 1

							#if self.steps > self.max_steps:
							#	self.eventState = 99
							#	print('reached max steps')

						elif act == 'none':  # send a blank line, needed to get reward for waiting
							cmd = 'socket>*' + 'time \n'
							self.mysignal.emit(cmd)
							#self.steps = self.steps + 1

						# self.emit(QtCore.SIGNAL("output(QString)"), cmd)
						newcmd = 'request_action>*'
						self.mysignal.emit(newcmd)
						self.logtimer = datetime.now()

					'''if self.steps_until_train <= 0:
						traincmd = 'train_model>*'
						self.mysignal.emit(traincmd)
						self.steps_until_train = self.train_interval  # train every 160 steps, after initial training period.

					if expdiff.microseconds / 1000 >= 300000:
						# update reward
						cmd = 'socket>*' + 'score' + ' \n\r'
						self.mysignal.emit(cmd)
						exptimer = datetime.now()'''


		except:
			print('------eventloop-error----')
			print(sys.exc_info()[0])
			print(sys.exc_info())


class Worldstate():
	def __init__(self):
		self.hp = 0
		self.mp = 0
		self.room_name = ''
		self.room_description = ''
		self.exits = []
		self.objmonlist = []
		self.delaystr = ''
		self.hostile_str = ''
		self.damage_str = ''
		self.reward = 0
		self.prev_action = ['none']
		self.last_action = ['none']

		self.exp = 0
		self.kill_monster_reward_string = ''
		self.kill_monster_reward_flag = False
		self.hit_monster_reward_string = ''
		self.hit_monster_reward_flag = False
		self.attempted_bad_exit_flag = False
		self.attempted_hit_missing_monster_flag = False
		self.character_died_flag = False

	def refresh_transient_flags(self):
		#reset some of the flags
		#for now decide to keep the room name
		self.delaystr = ''
		self.hostile_str = ''
		self.damage_str = ''
		self.last_action = ['none']

		self.kill_monster_reward_string = ''
		self.kill_monster_reward_flag = False
		self.hit_monster_reward_string = ''
		self.hit_monster_reward_flag = False
		self.attempted_bad_exit_flag = False
		self.attempted_hit_missing_monster_flag = False
		self.character_died_flag = False

	def refresh_reward(self):
		self.reward = 0

	def state_to_string(self):
		selfstrings = []
		selfstrings.append(self.hp)
		#selfstrings.append(self.mp)
		selfstrings.append(self.room_name)
		# selfstrings.append(self.room_description)
		# selfstrings.append(str(self.exits))
		selfstrings.append('objmonlist')
		selfstrings.append(str(self.objmonlist))
		# selfstrings.append(self.delaystr)
		selfstrings.append('hostile_str')
		selfstrings.append(self.hostile_str)
		# selfstrings.append(self.damage_str)
		# selfstrings.append(str(self.last_action))
		return selfstrings
	# print('statestring: ',selfstrings)


class MudBotClient(QtWidgets.QWidget):
	def __init__(self):
		# start user interface and socket
		super(MudBotClient, self).__init__()
		# start model thread
		self.ini_model_thread()
		# start event handling
		self.iniEventThread()
		self.iniMain()
		self.initialize_variables()
		self.modelthread.final_startup()


	def initialize_variables(self):
		self.initialize_encoders()
		self.initialize_self()
		self.initialize_world_state()


	def initialize_encoders(self):
		# initialize dictionaries
		self.objmondict = ['rabbit', 'raccoon']
		self.exitdict = ['north', 'east', 'south', 'west']
		self.room_name_dict = ['Limbo', 'Love', 'Brownhaven', 'Alley', 'Pawn', 'Path ', 'Petting']
		self.actiondict = ['none', 'north', 'east', 'south', 'west', 'killrabbit', 'killraccoon', 'look']
		self.hostiledict = ['rabbit','raccoon']

		# self.objmondict = ['golem', 'golems', 'gold']
		# self.exitdict = ['none']
		# self.room_name_dict = ['Limbo', 'Love']
		# self.actiondict = ['north', 'east', 'south', 'west', 'killrabbit', 'look']
		# self.actiondict = ['none','killgolem', 'look']

		# initialize tokenizations
		self.objmon_tokenizer = Tokenizer(num_words=len(self.objmondict) + 1)
		self.objmon_tokenizer.fit_on_texts(self.objmondict)
		self.exit_tokenizer = Tokenizer(num_words=len(self.exitdict) + 1)
		self.exit_tokenizer.fit_on_texts(self.exitdict)
		self.action_tokenizer = Tokenizer(num_words=len(self.actiondict) + 1)
		self.action_tokenizer.fit_on_texts(self.actiondict)
		self.reward_action_tokenizer = Tokenizer(num_words=len(self.actiondict) + 1)
		self.reward_action_tokenizer.fit_on_texts(self.actiondict)
		self.room_tokenizer = Tokenizer(num_words=len(self.room_name_dict) + 1)
		self.room_tokenizer.fit_on_texts(self.room_name_dict)
		self.hostile_tokenizer = Tokenizer(num_words=len(self.hostiledict) + 1)
		self.hostile_tokenizer.fit_on_texts(self.hostiledict)

	def initialize_self(self):
		self.maxhp = 125
		self.oldEXP = 0
		self.plot_interval = 500
		self.step_counter = 0
		self.steps_per_episode = 500
		self.train_interval = 16
		#self.training_interval
		self.initialize_rewards()


	def initialize_rewards(self):
		self.reward= 0
		self.killreward= 100
		self.diepenalty = 100

	def ini_model_thread(self):
		self.modelthread = ModelThread(self)  # pass self into the thread so that parent functions can be accessed
		self.modelthread.start()


	def iniEventThread(self):
		# start a thread to handle events
		# eventThread=Thread(None, self.eventHandler, None, (), {})
		# eventThread.start()
		# start a QThread instead, use class EventThread
		self.thread = EventThread(self)  # pass self into the thread so that parent functions can be accessed
		self.modelthread.thread = self.thread
		#start the thread
		# add: clicking ont he button stats the startLoop funciton which calls
		# a thread class function which starts the loop
		# QtCore.QObject.connect(self.botStart, QtCore.SIGNAL("clicked()"), self.startBot)


	def initialize_world_state(self):
		# initialize world state
		self.last_action = ['none']
		self.reward = 0
		self.world_state = Worldstate()
		self.state_array = self.encode_state(self.world_state)
		self.state_multiplier = self.modelthread.state_multiplier
		self.world_state_history = []
		self.state_array_history = []
		self.loss_history = []





	def store_memory(self, state, action, reward, next_state, done):
		#store memory inside the modelthread
		self.modelthread.memory.append((state, action, reward, next_state, done))

	def calculate_reward_from_state(self,state):
		#create rewards for the state
		reward = 0
		#assign reward for having killed a monster
		if state.kill_monster_reward_flag:
			reward += 100
		if state.hit_monster_reward_flag:
			reward += 1 #reward for hitting too
		#penalties for dying/going after monster not present, taking wrong exit
		if state.character_died_flag:
			reward -= 1000
		if state.attempted_bad_exit_flag:
			#could add a verification check here, using self.exits and self.last_action
			reward -= 1
		if state.attempted_hit_missing_monster_flag:
			#could add a verification check here, using self.objmonlist and self.last_action
			reward -= 1

		#add a reward for ticking
		if state.hp < self.maxhp:
			if state.room_name =='Order of Love':
				reward += 100 #ignore what action,a lways reward in order
				#if state.last_action == 'none':
			#		reward += 100
		# add a reward for ticking

		if state.hp > self.maxhp * 0.9:
			if state.room_name == 'Scranlin\'s Petting Zoo':
				reward += 1

		if state.hp < self.maxhp * 0.5: #penalty for seeking monsters when hp is low
			if state.room_name == 'Scranlin\'s Petting Zoo':
				reward -= 1


		# add a penalty for starting combat while health is low
		if state.hp < self.maxhp/2:
			if state.last_action == 'killrabbit':
					reward -= 5
			elif state.last_action == 'killraccoon':
					reward -= 5
		# add a penalty for idle at full health
		if state.hp == self.maxhp:
			#print(state.last_action)
			if state.last_action == 'none':
				reward -= 5
			if state.last_action == 'look':
				reward -= 5
		return reward

	def calculate_reward_from_state_simple(self,state):
		#create rewards for the state this version is very pure
		reward = 0
		#assign reward for having killed a monster
		if state.kill_monster_reward_flag:
			reward += 10

		if state.character_died_flag:
			reward -= 1000

		return reward

	def encode_state(self, state):
		# start with any empty matrices that would be needed
		backup_sum_objmon_matrix = np.array(np.zeros(len(self.objmondict) + 1))
		backup_sum_exit_matrix = np.array(np.zeros(len(self.exitdict) + 1))
		backup_sum_hostile_matrix = np.array(np.zeros(len(self.hostiledict) + 1))
		# be aware there may be other unexpected empty errors that need to be patched
		hp_encoded = np.array([state.hp / self.maxhp])
		mp_encoded = np.array([state.mp / 5])
		room_matrix = self.room_tokenizer.texts_to_matrix([state.room_name])
		room_matrix = sum(room_matrix)
		exit_matrix = self.exit_tokenizer.texts_to_matrix(state.exits)
		if len(exit_matrix) == 0:
			sum_exit_matrix = backup_sum_exit_matrix
		else:
			sum_exit_matrix = sum(exit_matrix)
		#print('exit_matrix check')
		#print(state.exits)
		#print(exit_matrix)
		#print(sum_exit_matrix)
		#print('state encoder hostile: ',state.hostile_str)
		hostile_matrix = self.hostile_tokenizer.texts_to_matrix([state.hostile_str])
		#print('hostile matrix:',hostile_matrix)
		if len(hostile_matrix) == 0:
			sum_hostile_matrix = backup_sum_hostile_matrix
		else:
			sum_hostile_matrix = sum(hostile_matrix)
		objmon_matrix = self.objmon_tokenizer.texts_to_matrix(state.objmonlist)
		action_encoded = sum(self.action_tokenizer.texts_to_matrix([self.last_action]))
		if len(objmon_matrix) == 0:
			sum_objmon_matrix = backup_sum_objmon_matrix
		else:
			sum_objmon_matrix = sum(objmon_matrix)
			# only allow values to be equal to 0 or 1. For gold coins piles seem to stack and can get 2,3, etc
			sum_objmon_matrix = np.where(sum_objmon_matrix > 0.5, 1.0, 0.0)
		#print(state.hostile_str)
		#state_array = np.concatenate((hp_encoded, room_matrix, sum_exit_matrix, sum_objmon_matrix, action_encoded),
		#							 axis=0)
		#stop encoding actions for now
		state_array = np.concatenate((hp_encoded,
									  room_matrix,
									  #sum_exit_matrix,
									  sum_objmon_matrix,
									  sum_hostile_matrix,
									  action_encoded,
									  ),
									 axis=0)
		return state_array

	def decode_state(self, state_array):
		#work in progress below
		#state_array = np.concatenate((hp_encoded, room_matrix, sum_exit_matrix, sum_objmon_matrix),
		print(state_array)
		lenhp = 1
		lenrooms = len(self.room_name_dict) + 1
		lenexits = len(self.exitdict) + 1
		lenobjmon = len(self.objmondict) + 1
		print('ok')



		index=0
		hp_encoded = state_array[0]
		print(hp_encoded)
		index=index+lenhp
		room_matrix = state_array[index:index+lenrooms]
		print(room_matrix)
		index+=lenrooms
		exit_matrix = state_array[index:index+lenexits]
		print(exit_matrix)
		index+=lenexits
		objmon_matrix = state_array[index:index+lenobjmon]
		print(objmon_matrix)

		#decode matrix into text
		hp = hp_encoded*self.maxhp
		print(hp)
		nonzeroind = np.nonzero(room_matrix)[0][0]
		room = self.room_name_dict[nonzeroind-1]
		print(room)
		nonzeroind = np.nonzero(exit_matrix)[0][0]
		print(nonzeroind)
		print(self.exitdict[nonzeroind-1])

	# self.room_name_dict = ['Limbo', 'Love', 'Brownhaven', 'Alley', 'Pawn', 'Path ', 'Petting']




	def listtostring(self, mylistobj):
		thestring = ''.join(str(x) + ' ' for x in mylistobj)
		return thestring


	def msgFromEventLoop(self, qstring):

		# takes signal from event thread, uses string to write appropriate command to socket/walk/etc
		# print 'received qstring: ' + str(qstring)
		# print str(qstring)
		# check for walking header
		if str(qstring).find('setRooms>') != -1:
			self.pathReady = 0  # have to recalculate path
			# pull out header, start dest, stop dest,
			walkString = str(qstring).split('*')
			header = walkString[0]
			self.roomIdStart = int(walkString[1])
			self.roomIdStop = int(walkString[2])
			self.pathFinderClicked()  # this stes up the path
			self.pathReady = 1  # path setup
		elif str(qstring).find('log>') != -1:
			# log text
			print('got command to log text')
			self.logText()


		elif str(qstring).find('onestep>') != -1:
			# then take one step
			# self.oneStep()
			if self.pathReady == 1:
				# print ' onestep received but is path setup?'
				self.oneStep()

		# check for command to write to socket
		elif str(qstring).find('socket>*') != -1:
			#print('socket signal')
			socketString = str(qstring).split('*')
			header = socketString[0]
			msg = socketString[1]
			# write to socket
			bmsg = msg.encode('utf-8')
			self.write_socket(QtCore.QByteArray(bmsg))
			self.display.append("EventLoop>" + msg)


		elif str(qstring).find('request_action>*') != -1:

			try:
				self.advance_one_step()  # adavnce the world one step
				# pass action from model thread to action thread. may need to go model -> eventthread to go faster
				self.generate_new_action(self.actiondict,
										 self.state_array_history,
										 self.state_multiplier
										 )
				self.store_states_and_memory()
			except:
				print('memory not ready for storage')




		elif str(qstring).find('train_model>*') != -1:
			# print('skipping train_model>* request')
			self.modelthread.train_model()

		# check for target info
		elif str(qstring).find('possibleTarget>') != -1:
			socketString = str(qstring).split('*')
			header = socketString[0]
			self.possibleTarget = socketString[1]

	def generate_new_action(self,actiondict,state_array_history,state_multiplier):
		epsilon = self.modelthread.epsilon
		#if epsilon == 1 or (epsilon > 0 and 1 > epsilon > random.random()):
		if np.random.rand() <= epsilon: #if rnom number is less than epsilon
			#print('made it this far')
			action = random.choice(actiondict)
			self.last_action = [action]
			# print('chose action: ', action)


		else:
			# calculate state with maximum Q value
			action = self.get_action_from_model(actiondict,state_array_history,state_multiplier)
			self.last_action = [action]
			#print('model chose action: ', action)



		self.modelthread.epsilon = self.modelthread.epsilon * self.modelthread.epsilon_decay
		self.modelthread.epsilon = max(self.modelthread.epsilon_min, self.modelthread.epsilon)
		#print('generated action',action)
		self.thread.action_from_parent = action
		#print('updated action to thread')
		#print(self.thread.action_from_parent )

	def get_action_from_model(self,actiondict,state_array_history,state_multiplier):
		mult = state_multiplier
		inputx = self.reshape_x_and_combine(state_array_history[-mult:], mult)  # careful changing multiplier from 16
		best_action = 'none'
		try:
			predictions = self.modelthread.model.predict(inputx)
			max_val_index = np.argmax(predictions)
			best_action = actiondict[max_val_index]
		except:
			print('failed to get prediction array ')


		#print('model chose: ',best_action)
		return best_action


	def reshape_x(self, x_input):
		# shape for X should be:
		# (num_samples,num_features)
		# reshape to get into the right format for the neural net

		newx = np.stack(x_input, axis=1)
		newx = np.transpose(newx)

		return newx

	def reshape_x_and_combine(self, x_input, multiplier):
		# X for a neural net that takes one state at a time
		# has shape num_samples, num_features
		# if we combine to take say 16 states as input into neural net
		# new shape is 16*num_sample, num_features

		x = self.reshape_x(x_input)

		# now we want to reshape X based on the multiplier.
		# start by reshaping as standard. now x is num_sample x num_features
		[num_samples, num_features] = x.shape
		nn_s = num_samples / multiplier
		nn_f = num_features * multiplier

		# have to make a few adjustments to deal with odd shaped matrices
		nnn_s = int(np.floor(nn_s))
		nnn_f = int(np.ceil(nn_f))
		extra_rows = num_samples % multiplier
		x_corrected = x[extra_rows:]
		newX = x_corrected.reshape(nnn_s, nnn_f)
		return newX



	def combine_y_array(self, y_array, multiplier):
		# print('combine_y_array function started')
		# print(y_array)
		y_cat = np.concatenate(y_array, axis=0)
		# print(y_cat)

		# y array is of shape num_samples, num_labels
		# we need to stack it so that it becomes 16 num_samples in a row
		# new shape will be num_samples/16 , 16*num_labels

		# now we want to reshape X based on the multiplier.
		# start by reshaping as standard. now x is num_sample x num_features
		[num_samples, num_features] = y_cat.shape
		# print('y_cat shape', y_cat.shape)
		nn_s = num_samples / multiplier
		# nn_f = num_features * multiplier
		# print('nn_s nn_f',nn_s,nn_f)
		# have to make a few adjustments to deal with odd shaped matrices
		nnn_s = int(np.floor(nn_s))
		# nnn_f = int(np.ceil(nn_f))
		extra_rows = num_samples % multiplier
		# 5/26 we have an offset of 1 to fix. current reward corresponds to past action currently\
		# we add an offset here to fix the issue
		offset = -1
		y_corrected = y_cat[extra_rows + offset:offset]
		# print('ycorr', y_corrected)

		# now get average of reward in the combined set of inputs
		# split into segments
		y_segments = np.split(y_corrected, nnn_s)
		# print('y seg', y_segments)
		# get sum of each segment
		newY = np.sum(y_segments, axis=1)

		# newY = y_corrected.reshape(nnn_s, nnn_f)
		# print(newY)

		return newY





	def iniMain(self):
		self.iniSockets()
		self.initUI()
		self.connect_buttons()

	def iniSockets(self):
		# load telnet objects
		self.HOST = 'localhost'
		self.port = 4000

		# make a QT socket
		self.tcpSocket = QtNetwork.QTcpSocket()
		# telnet
		self.tn = telnetlib.Telnet(self.HOST, self.port)
		# get the socket from telnetlib and feed it to the QtNetwork QTCSocket object!
		self.tcpSocket.setSocketDescriptor(self.tn.sock.fileno())
		# start timer
		#self.parsetimer = datetime.now()
		self.login()

	def closeSockets(self):
		try:
			# close the socket
			self.tn.close()
			#self.tcpSocket.close() doesnt work
		except:
			print('unable to close tn socket unable to close telnet socket')

	def reconnect(self):
		try:
			self.closeSockets()
			self.tn = telnetlib.Telnet(self.HOST, self.port)
			#print('recreated tn object')
			self.tcpSocket.setSocketDescriptor(self.tn.sock.fileno())
			#print('set socket descriptor to telnet file')
			if self.tcpSocket.isValid():
				print('valid socket')
				return True
			elif not self.tcpSocket.isValid():
				print('bad socket')
				return False
		except:
			print('fail')
			return False

	def dmlogin(self):
		# auto login, set flags (clear long)
		# add b before strings to turn them into bytes
		# txt = b'test' + b'\n' + b'asdfasdf' + b'\r\n' #+ 'clear long\r\n' + 'set auto' + '\r\n'

		# make a QT socket
		self.dmSocket = QtNetwork.QTcpSocket()
		# telnet
		self.dmtn = telnetlib.Telnet(self.HOST, self.port)
		# get the socket from telnetlib and feed it to the QtNetwork QTCSocket object!
		self.dmSocket.setSocketDescriptor(self.dmtn.sock.fileno())

		a = datetime.now()
		done = False
		print('DM RESET IN PROGRESS')
		commandlist = ['Rex',
					   'asdfasdf',
					   'broadcast hello',
					   'broadcast hello',
					   'broadcast hello',
					   'broadcast hello',
					   'broadcast hello'
					   ]
		i = 0
		while not done:
			diff = datetime.now() - a
			tick = diff.microseconds / (1000*1000) #every 25 ms send a command
			if tick > 1:
				cmd = commandlist[i].encode('utf-8')
				print(cmd)
				self.dmSocket.write(QtCore.QByteArray(cmd))
				i = i + 1
				if i > 6:
					i = 0
				socket_data = self.dmSocket.readAll()
				print(socket_data)
				a = datetime.now()

	def login(self):
		# auto login, set flags (clear long)
		# add b before strings to turn them into bytes
		# txt = b'test' + b'\n' + b'asdfasdf' + b'\r\n' #+ 'clear long\r\n' + 'set auto' + '\r\n'
		text = 'tester\nasdfasdf\nlook\n'
		btext = text.encode('utf-8')
		try:
			self.write_socket(QtCore.QByteArray(btext))
			print('login info sent')
		except:
			print('not able to log in')


	def logout(self):
		# auto login, set flags (clear long)
		# add b before strings to turn them into bytes
		# txt = b'test' + b'\n' + b'asdfasdf' + b'\r\n' #+ 'clear long\r\n' + 'set auto' + '\r\n'
		#text = '\nquit\r\n'
		#btext = text.encode('utf-8')
		#self.write_socket(QtCore.QByteArray(btext))
		#time.sleep(1) #pause to let server close connection
		#print('resumed after pause')
		#try skipping quit message for now
		self.closeSockets()


	def copy_backup_file(self):
		src_dir = os.getcwd()
		filedir = src_dir + "\\..\\..\\mordor\\player\\"
		backupfile = 'Tester_backup'
		#backupfile= 'Tester_limbotesting'
		copyfile = 'Tester'
		fullpath = filedir + backupfile
		copypath = filedir + copyfile
		shutil.copyfile(fullpath, copypath)
		print('Tester loaded from backup', fullpath)

	def reset_player_file(self):
		#log the player out
		try:
			self.logout()
			print('logged out')
			self.copy_backup_file()
			print('file copied')

		except:
			print('reset error')

		#try:  # print('Reconnecting complete. Now logging in')
		#print('try to dmlog in')
	#		#self.dmlogin()
		#except:
		#	print('dm reset error')

		success = False
		while not success:
			try:
				print('trying to reconnect')
				success = self.reconnect()
				self.world_state.room_name ='Order'
				print(self.world_state.state_to_string())
				# reset world with help of DM
			except:
				print('connection error, trying again')

		try:#print('Reconnecting complete. Now logging in')
			print('try to log in')
			self.login()

		except:
			print('could not log in')



	def initUI(self):
		# initialize layouts
		self.hlay = QtWidgets.QHBoxLayout(self)  # horizontal layout to hold main game and plots
		self.lay = QtWidgets.QVBoxLayout(self) #this vertical box will hold main game items
		self.graphics_lay = QtWidgets.QVBoxLayout(self)  # this vertical box will hold main game items
		self.exitBox = QtWidgets.QHBoxLayout(self)
		self.mapCmdsBox = QtWidgets.QHBoxLayout(self)
		self.action_box = QtWidgets.QHBoxLayout(self)

		# create widget/box to show output from mud
		self.display = QtWidgets.QTextBrowser()
		#QSizePolicy takes two intgers (horizontal tstretchfactor, verticalstretchfactor) 0,255
		#self.display.setSizePolicy(QtWidgets.QSizePolicy(1, 10))
		# now connect the output

		self.tcpSocket.readyRead.connect(self.tcpSocketReadyReadEmitted)

		# create widget/box to show input from user
		self.lineInput = QtWidgets.QLineEdit()

		# connect the input now
		# QtCore.QObject.connect(self.lineInput, QtCore.SIGNAL("returnPressed()"), self.lineeditReturnPressed)
		self.lineInput.returnPressed.connect(self.lineeditReturnPressed)

		# add widget for room name
		self.roomInfo = QtWidgets.QStatusBar()
		self.roomInfo.showMessage('Room Info:')  # this gets updated with information by

		self.botStart = QtWidgets.QPushButton("Start Bot")
		self.save_model_btn = QtWidgets.QPushButton("Save Model")
		self.load_model_btn = QtWidgets.QPushButton("Load Model")
		self.train_model_btn = QtWidgets.QPushButton("Train Model")
		self.evaluate_model_btn = QtWidgets.QPushButton("Evaluate Model")
		self.evaluate_data_btn = QtWidgets.QPushButton("Evaluate Data")
		self.save_data_btn = QtWidgets.QPushButton("Save Data")
		self.load_data_btn = QtWidgets.QPushButton("Load Data")

		# ['wait', 'north', 'east', 'south', 'west', 'killrabbit', 'killraccoon', 'look']
		self.wait_btn = QtWidgets.QPushButton("look")
		self.north_btn = QtWidgets.QPushButton("north")
		self.east_btn = QtWidgets.QPushButton("east")
		self.south_btn = QtWidgets.QPushButton("south")
		self.west_btn = QtWidgets.QPushButton("west")
		self.krabbit_btn = QtWidgets.QPushButton("killrabbit")
		self.kraccoon_btn = QtWidgets.QPushButton("killraccoon")
		self.look_btn = QtWidgets.QPushButton("look")

		self.action_box.addWidget(self.wait_btn)
		self.action_box.addWidget(self.north_btn)
		self.action_box.addWidget(self.east_btn)
		self.action_box.addWidget(self.south_btn)
		self.action_box.addWidget(self.west_btn)
		self.action_box.addWidget(self.krabbit_btn)
		self.action_box.addWidget(self.kraccoon_btn)
		self.action_box.addWidget(self.look_btn)

		self.mapCmdsBox.addWidget(self.botStart)
		self.mapCmdsBox.addWidget(self.train_model_btn)
		self.mapCmdsBox.addWidget(self.evaluate_model_btn)
		self.mapCmdsBox.addWidget(self.evaluate_data_btn)
		self.mapCmdsBox.addWidget(self.save_model_btn)
		self.mapCmdsBox.addWidget(self.load_model_btn)
		self.mapCmdsBox.addWidget(self.save_data_btn)
		self.mapCmdsBox.addWidget(self.load_data_btn)
		# widget for health and mana
		self.hpmp = QtWidgets.QStatusBar()
		self.hpmp.showMessage('Hp:    Mp:    Xp:')

		# widget for epsilon plot
		self.epsilon_graph = pg.PlotWidget()
		#self.roomMonsters.showMessage('Epsilon vs Steps:')
		x = [0]
		y = [0]
		self.epsilon_graph.setBackground('w')
		self.epsilon_graph.setTitle('Epsilon')
		pen = pg.mkPen(color=(0, 0, 0))
		self.epsilon_graph.plot(x,y,pen=pen)
		# QSizePolicy takes two intgers (horizontal tstretchfactor, verticalstretchfactor) 0,255
		#self.epsilon_graph.setSizePolicy(QtWidgets.QSizePolicy(1,1))


		# widget for reward plot
		self.reward_graph = pg.PlotWidget()
		# self.roomMonsters.showMessage('Epsilon vs Steps:')
		rwd = [0]
		steps = [0]
		self.reward_graph.setBackground('w')
		self.reward_graph.setTitle('Reward')
		pen = pg.mkPen(color=(0, 0, 0))
		self.reward_graph.plot(steps, rwd, pen=pen)
		#self.reward_graph.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, 200))

		# widget for model loss plot
		self.model_loss_graph = pg.PlotWidget()
		self.model_loss_graph.setBackground('w')
		self.model_loss_graph.setTitle('Model Loss')
		pen = pg.mkPen(color=(0, 0, 0))
		self.model_loss_graph.plot([0],[0], pen=pen)

		# finalize layout:

		self.lay.addWidget(self.roomInfo)
		# self.lay.addWidget(self.roomExits)

		# self.lay.insertLayout(3,self.exitBox)
		self.lay.insertLayout(4, self.mapCmdsBox)
		self.lay.insertLayout(4, self.action_box)

		#self.lay.addWidget(self.epsilon_graph)
		#self.lay.addWidget(self.reward_graph)
		self.lay.addWidget(self.hpmp)
		self.lay.addWidget(self.display)
		self.lay.addWidget(self.lineInput)

		#add graphs to graphics_layout
		self.graphics_lay.addWidget(self.epsilon_graph)
		self.graphics_lay.addWidget(self.reward_graph)
		self.graphics_lay.addWidget(self.model_loss_graph)
		# self.lay.addLayout(self.exitBox)

		self.hlay.insertLayout(1,self.lay)
		self.hlay.insertLayout(2,self.graphics_lay)
		# setup window size and icon
		self.setGeometry(100, 100, 1200, 900)
		self.setWindowTitle('Mud Client Botter')
		self.setWindowIcon(QtGui.QIcon('Robot-icon.png'))

		self.show()
	def connect_buttons(self):
		self.botStart.clicked.connect(self.startBot)
		self.save_model_btn.clicked.connect(self.save_model)
		self.load_model_btn.clicked.connect(self.load_model)
		self.train_model_btn.clicked.connect(self.modelthread.train_model)
		self.evaluate_model_btn.clicked.connect(self.plot_model_loss)
		self.evaluate_data_btn.clicked.connect(self.evaluate_data)
		self.save_data_btn.clicked.connect(self.save_data)
		self.load_data_btn.clicked.connect(self.load_data)

		self.wait_btn.clicked.connect(self.act_btn_response)
		self.north_btn.clicked.connect(self.act_btn_response)
		self.east_btn.clicked.connect(self.act_btn_response)
		self.south_btn.clicked.connect(self.act_btn_response)
		self.west_btn.clicked.connect(self.act_btn_response)
		self.krabbit_btn.clicked.connect(self.act_btn_response)
		self.kraccoon_btn.clicked.connect(self.act_btn_response)
		self.look_btn.clicked.connect(self.act_btn_response)

	def act_btn_response(self):
		sending_button = self.sender()
		action = sending_button.text()

		# store the action
		self.last_action = [action]
		# write action to socket
		if action == 'killrabbit':
			action = 'kill rabbit'
		elif action == 'killraccoon':
			action = 'kill raccoon'
		elif action == 'killgolem':
			action = 'kill golem'

		action = action + ' \n'
		bmsg = action.encode('utf-8')

		self.write_socket(QtCore.QByteArray(bmsg))


	def logText(self):
		# this function appends the text to a file, and clears the window
		file = open("log.txt", "a")

		# appendText = str(unicode(self.display.toPlainText()).split('u\'\''))
		#appendText = str(unicode(self.display.toPlainText()).split('\n'))
		#appendText = self.cleanText(appendText)
		# print appendText
		appendList = appendText.split('u\'\'')
		# write the date first
		a = datetime.now()
		file.write(str(a) + '\n')
		# append all lines
		for k in appendList:
			file.write(k + '\n')
		file.close()

		# now clear QTextBrowser
		self.display.clear()

	def lineeditReturnPressed(self):
		txt = str(self.lineInput.text()) + '\r\n'
		# add command stacking : replace ; with \r\n
		txt = txt.replace(';', '\r\n')
		btxt = txt.encode('utf-8')

		self.write_socket(QtCore.QByteArray(btxt))
		# self.display.append("client")
		self.display.append("Client>" + txt)
		self.lineInput.clear()

	def startBot(self):
		print('attempting to start event loop')
		# start the event loop
		self.thread.begin()
		self.thread.start()
		print('event loop thread has been started')
		# set the event state to 0
		self.thread.unpause()
		print('set thread eventState to 0')

	def save_data(self):
		print('save data function called')
		pickle.dump(self.modelthread.memory, open("memory.p", "wb"))
		pickle.dump(self.modelthread.epsilon_array, open("epsilon_array.p", "wb"))

	def load_data(self):
		print('load data function called')
		self.modelthread.memory = pickle.load(open("memory.p", "rb"))
		self.modelthread.epsilon_array = pickle.load(open("epsilon_array.p", "rb"))
		#update epsilon so training can resume
		self.modelthread.epsilon = self.modelthread.epsilon_array[-1]



	def save_model(self):
		# save model and architecture to single file
		self.modelthread.model.save("model.h5")
		self.modelthread.target_model.save("target_model.h5")
		#model_json = self.model.to_json()
		#with open("model.json", "w") as json_file:
			#json_file.write(model_json)
		print('wrote model to model.h5')

		# also have to load and save weights
		#self.model.save_weights('model_weights.h5')
		#print('weights stored in model_weights.h5')

	def load_model(self):
		# load    model
		self.modelthread.model = load_model('model.h5')
		self.modelthread.target_model = load_model('target_model.h5')
		#json_file = open('model.json', 'r')
		#loaded_model_json = json_file.read()
		#json_file.close()
		# load weights too

		#self.model = model_from_json(loaded_model_json)
		#self.model.load_weights("model_weights.h5")
		print('loaded model model.h5')

	def closeEvent(self, event):
		reply = QtWidgets.QMessageBox.question(self, 'message',
											   "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
											   QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

		# get event handler thread to break
		self.thread.eventState = 99

		if reply == QtWidgets.QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()
	def parse_worldstate_old(self, txt):
		# want to continuously parse and update the world state
		# but only store and reset the world state once per second
		newstate = self.world_state


		# consider reset of award
		#self.reward = 0
		#Reasont o reset reward is that the value of the action is determined by the immediate reward.
		# state_n -> sate_n+1 due to action n causes reward r
		# integral of r is good to evaluate overall agent, but the Q network should predict
		# expected reward from each action, which should not be cumulative.
		newstate.reward = self.reward
		status, room_name = self.get_room_name(txt)

		if status == True:
			newstate.room_name = room_name
			#print('new room detected',room_name)
			# shoud also reset monsters present
			# UPON ENTERING a new room, reset obj/monsters

			newstate.objmonlist = []
			if self.last_action != 'look':
				# self.reward = self.reward + 0.001  # small positive reward for a room name
				#tmpreward = self.reward + 0.001
			# only give award when last action was not simply loooking around
			# meant to encourage looking around as well as exploring
				newstate.reward = self.reward

		status, exitstr = self.get_exits_str(txt)
		if status == True:
			exits = self.parse_exits_str(exitstr)
			newstate.exits = exits
		status, room_description = self.get_room_description(txt)
		if status == True:
			newstate.room_description = room_description
		status, objmonstr = self.getobjmonstr(txt)
		if status == True:
			objmonlist = self.parseobjmonstr(objmonstr)
			newstate.objmonlist = objmonlist
		status, hpmpstring = self.gethpmp(txt)
		if status == True:
			# print('hpmpstring:', hpmpstring)
			hp, mp = self.parse_hpmpstr(hpmpstring)
			newstate.hp = hp
			newstate.mp = mp
			'''
			if newstate.hp < 50:
				self.thread.action_from_parent = 'jump'
				newstate.reward = self.reward
			elif newstate.hp < 30:
				self.thread.action_from_parent = 'bleed'
				newstate.reward = self.reward
			'''
		status, delaystr = self.getdelaystr(txt)
		if status == True:
			newstate.delaystr = delaystr
		status, hostile_str = self.get_hostile_str(txt)
		if status == True:
			newstate.hostile_str = hostile_str
			print('hostile',hostile_str)

		status, damage_str = self.got_hit_str(txt)
		if status == True:
			newstate.damage_str = damage_str
			print('hostile', hostile_str)

		status, expstr = self.get_experience(txt)
		if status == True:
			if self.oldExp == 0:  # not updated yet
				self.oldExp = int(expstr)
			elif self.oldExp > 0:
				self.newExp = int(expstr)

		status, killrewardstr = self.get_kill_reward(txt)
		if status == True:
			self.reward = self.reward + 150  # reward for killing a mosnter is big
			newstate.reward = self.reward
		#print('reward is: ', self.reward)
		# print('last action is:',self.last_action)

		status, hitrewardstr = self.get_hit_reward(txt)
		# need to update hostile string here

		if status == True:
			#reward/penalty for engaging in combat
			if newstate.hp < 50:
				self.reward = self.reward - 50
				newstate.reward=self.reward
			elif newstate.hp > 90:
				self.reward=self.reward + 40 * ((newstate.hp-90) / (35))
				newstate.reward = self.reward
			# 50 - 100 scaling penalty
			elif newstate.hp < 90:
				#goes from 0 to -5 as health decreases
				self.reward = self.reward - 40*((90 - newstate.hp)/ (40))
				newstate.reward = self.reward

		status, monsterdiddamage = self.get_monsterhit_reward(txt)
		if status == True:
			newstate.hostilestr = monsterdiddamage
			print('hostilestr', newstate.hostilestr)

		status, monstermissstring = self.get_monstermiss_reward(txt)
		#need to update hostile string here
		if status == True:
			newstate.hostilestr = monsterdiddamage
			print('hostilestr',newstate.hostilestr)
			# reward/penalty for engaging in combat
			if newstate.hp < 50:
				self.reward = self.reward - 50
				newstate.reward = self.reward
			elif newstate.hp > 80:
				self.reward = self.reward + 40 * ((newstate.hp-80) / (35))
				newstate.reward = self.reward
			# 50 - 100 scaling penalty
			elif newstate.hp < 80:
				# goes from 0 to -5 as health decreases
				self.reward = self.reward - 40 * ((80 - newstate.hp) /40)
				newstate.reward = self.reward


		# print('reward is: ', self.reward)
		# print('last action is:',self.last_action)

		status, movepenaltystr = self.get_move_penalty(txt)
		if status == True:
			self.reward = self.reward - 1#small negative rewar
			newstate.reward = self.reward

		status, nomonsterpenaltystr = self.get_nomonster_penalty(txt)
		if status == True:
			self.reward = self.reward  -1  # small negative rewar
			newstate.reward = self.reward

		status, killedstr = self.get_killed_penalty(txt)
		if status == True:
			self.reward = self.reward - 150  #negative penalty for dying
			newstate.reward = self.reward

		# give nice bonus for ticking
		if newstate.room_name == 'Order of Love':
			if newstate.hp < self.maxhp:
				# print('should be getting a bonus, are we?')
				self.reward = self.reward + (self.maxhp-newstate.hp)/43 +0.05 #big positive reward for ticking, modulated by differential
				newstate.reward = self.reward
			elif newstate.hp < 60:
				self.reward = self.reward + 2
				#self.thread.action_from_parent = 'jump'
				newstate.reward = self.reward

			elif newstate.hp == self.maxhp:
				self.reward = self.reward -.1 #small negative reward for spending too much time in tick room
				newstate.reward = self.reward

		# give penalty for not  ticking
		if newstate.room_name != 'Order of Love':
			if newstate.hp < 60:

				self.reward = self.reward #- (60-newstate.hp)/43 -0.05 #big neg reward for not ticking, modulated by differential
				newstate.reward = self.reward



		# state dictionary for reference below
		# {'hp': 0, 'mp': 0, 'room_name': '', 'room_description': '', 'exits': [], 'objmonlist': [], 'delaystr': '',
		# 'hostile_str': '', 'damage_str': '', 'reward': 0}

		# return thenew state
		return newstate

	def parse_worldstate(self, txt):
		# want to continuously parse and update the world state
		# do not apply rewards directly here
		# do not reset the world_state here, laod previous state and update
		# make sure to carefully consider hwo often to store the world_state into memory
		# want to avoid duplicate rewards/entries for the same state action pair
		newstate = self.world_state

		status, room_name = self.get_room_name(txt)
		if status == True:
			newstate.room_name = room_name

		status, exitstr = self.get_exits_str(txt)
		if status == True:
			newstate.exits = self.parse_exits_str(exitstr)

		status, room_description = self.get_room_description(txt)
		if status == True:
			newstate.room_description = room_description

		status, objmonstr = self.getobjmonstr(txt)
		if status == True:
			objmonlist = self.parseobjmonstr(objmonstr)
			newstate.objmonlist = objmonlist

		status, hpmpstring = self.gethpmp(txt)
		if status == True:
			newstate.hp, newstate.mp = self.parse_hpmpstr(hpmpstring)

		status, delaystr = self.getdelaystr(txt)
		if status == True:
			newstate.delaystr = delaystr

		status, hostile_str = self.get_hostile_str(txt)
		if status == True:
			newstate.hostile_str = hostile_str
			#print('hostile',hostile_str)

		status, damage_str = self.got_hit_str(txt)
		if status == True:
			newstate.damage_str = damage_str
			newstate.hostile_str = hostile_str
			#print('damagestr', damage_str)

		status, expstr = self.get_experience(txt)
		if status == True:
			newstate.exp=int(expstr)

		status, killrewardstr = self.get_kill_reward(txt)
		if status == True:
			newstate.kill_monster_reward_string = killrewardstr
			newstate.kill_monster_reward_flag = True

		status, hitrewardstr = self.get_hit_reward(txt)
		# need to update hostile string here
		if status == True:
			newstate.hit_monster_reward_string = hitrewardstr
			newstate.hit_monster_reward_flag = True


		status, monsterdiddamage = self.get_monsterhit_reward(txt)
		if status == True:
			newstate.hostilestr = monsterdiddamage

		status, monstermissstring = self.get_monstermiss_reward(txt)
		#need to update hostile string here
		if status == True:
			newstate.hostilestr = monsterdiddamage

		status, movepenaltystr = self.get_move_penalty(txt)
		if status == True:
			newstate.attempted_bad_exit_flag = True

		status, nomonsterpenaltystr = self.get_nomonster_penalty(txt)
		if status == True:
			newstate.attempted_hit_missing_monster_flag = True

		status, killedstr = self.get_killed_penalty(txt)
		if status == True:
			newstate.character_died_flag = True

		return newstate

	def parse_hpmpstr(self, hpmpstring):
		hp = int(hpmpstring.split('H')[0])
		mp = int(hpmpstring.split('H')[1].split('M')[0])
		return hp, mp

	def get_kill_reward(self, txt):
		start_indicator = '\\n\\rYou gained'
		end_indicator = 'experience for the death'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_hit_reward(self, txt):
		start_indicator = 'You hit for'
		end_indicator = 'damage'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_monstermiss_reward(self, txt):
		start_indicator = 'The '
		end_indicator = 'missed you'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_monsterhit_reward(self, txt):
		start_indicator = 'The '
		end_indicator = 'hit you for'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_move_penalty(self, txt):
		start_indicator = 'You can'
		end_indicator = 'go that way.'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_nomonster_penalty(self, txt):
		start_indicator = 'You don'
		end_indicator = 'see that here.'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_killed_penalty(self, txt):
		start_indicator = 'The'
		end_indicator = 'killed you.'
		status, kexpstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, kexpstr

	def get_experience(self, txt):
		# Experience Counter
		start_indicator = '\\n\\r\\x1b[33m'
		end_indicator = 'Experience'
		status, expstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, expstr

	def getdelaystr(self, txt):
		start_indicator = "Please wait "
		end_indicator = " more second"
		status, delaystr = self.pull_text(txt, start_indicator, end_indicator)
		return status, delaystr

	def getobjmonstr(self, txt):
		start_indicator = ".\\n\\r\\x1b[37m\\x1b[37m"
		end_indicator = "\\n\\r\\x1b[37m\\n\\r\\x1b[37m("
		status, objmonstr = self.pull_text(txt, start_indicator, end_indicator)
		return status, objmonstr

	def parseobjmonstr(self, objmonstr):
		objmondict = self.objmondict
		objmonstr = objmonstr.replace('.', ' ')
		objmonstr = objmonstr.replace(',', ' ')
		objmonstr = objmonstr.replace('rabbits', 'rabbit')
		objmonstr = objmonstr.replace('raccoons', 'raccoon')
		word_list = objmonstr.split(' ')
		objmonlist = []
		for w in word_list:
			if w in objmondict:
				objmonlist.append(w)
		return objmonlist

	def get_room_name(self, txt):
		start_indicator = "\\n\\r\\x1b[36m"
		end_indicator = "\\n\\r\\n\\r\\x1b[37m"
		status, room_name = self.pull_text(txt, start_indicator, end_indicator)
		return status, room_name

	def get_room_description(self, txt):
		start_indicator = "\\n\\r\\n\\r\\x1b[37m"
		end_indicator = "\\n\\r\\x1b[32m"
		status, txtstring = self.pull_text(txt, start_indicator, end_indicator)
		txtstring.replace("\\n\\r", " ")
		return status, txtstring

	def get_exits_str(self, txt):
		start_indicator = "\\n\\r\\x1b[32mObvious exits: "
		end_indicator = ".\\n\\r\\x1b[37m"
		status, txtstring = self.pull_text(txt, start_indicator, end_indicator)
		return status, txtstring

	def parse_exits_str(self, exits_str):
		exitlist = exits_str.split(',')
		return exitlist

	def gethpmp(self, txt):
		start_indicator = "\\x1b[37m("
		end_indicator = "): \\x1b[37m"
		status, txtstring = self.pull_text(txt, start_indicator, end_indicator)
		return status, txtstring

	def get_hostile_str(self, txt):
		start_indicator = "\\x1b[31mThe "
		end_indicator = "is attacking you."
		status, txtstring = self.pull_text(txt, start_indicator, end_indicator)
		return status, txtstring

	def got_hit_str(self, txt):
		start_indicator = "\\x1b[31mThe"
		end_indicator = "damage."
		status, txtstring = self.pull_text(txt, start_indicator, end_indicator)
		return status, txtstring

	def pull_text(self, txt, indicator_start, indicator_end):
		status = False
		pullstring = ''
		try:
			start_ind = 0
			end_ind = 0
			name_end = indicator_end  # "\\n\\r\\n\\r\\x1b[37m"
			if txt.find(indicator_end) != -1:
				if txt.find(indicator_start) != -1:
					start_ind = txt.find(indicator_start) + len(indicator_start)
					end_ind = txt.find(indicator_end)
					pullstring = txt[start_ind:end_ind]
					status = True
		except:
			print('didnt find the text')
		return status, pullstring


	def plot_reward(self):
		try:
			rewards = [] #get reward from memory
			for i in np.arange(len(self.modelthread.memory)):
				(state, actionindex, reward, state2, done) = self.modelthread.memory[i]
				rewards.append(reward)
			steps = np.arange(0,len(self.modelthread.memory))
			pen = pg.mkPen(color=(0, 0, 0))
			self.reward_graph.plot(steps, rewards , pen=pen)
		except:
			print('bug plotting reward')
			print(sys.exc_info()[0])
			print(sys.exc_info())


	def plot_epsilon(self,epsilon_array):
		steps = np.arange(0,len(epsilon_array))
		pen = pg.mkPen(color=(0, 0, 0))
		self.epsilon_graph.plot(steps, epsilon_array, pen=pen)



	def evaluate_data(self):
		# This function provides information on the current dataset
		print('Memory Size:', len(self.modelthread.memory))
		# calculate total possible number of states
		# get one state
		sample = random.sample(self.modelthread.memory, 1)[0]
		state, action, reward, new_state, done = sample

		hp_states = self.maxhp
		room_states = len(self.room_name_dict) + 1
		exit_states = 2 ** (len(self.exitdict) + 1)
		objectstates = 2 ** (len(self.objmondict) + 1)
		hostile = 2 ** (len(self.hostiledict) + 1)

		total_possible_states = hp_states * room_states * exit_states * objectstates * hostile

		# number of states should consider mutualy exclusive or not

		# total possible states =
		# hp states x
		# room states x
		# exits x
		# objmon matrix x
		# hostile_str

		'''
		self.objmondict = ['rabbit', 'rabbits', 'raccoon', 'raccoons', 'book', 'gold']
		self.exitdict = ['north', 'east', 'south', 'west']
		self.room_name_dict = ['Limbo', 'Love', 'Brownhaven', 'Alley', 'Pawn', 'Path ', 'Petting']
		self.actiondict = ['none', 'north', 'east', 'south', 'west', 'look']
		self.hostiledict = ['rabbit','raccoon']

			hp states = 125 
			room states = 7+1 (mutually exclusive)
			exit states =  2^(4+1) = 32	
			objectstates = 2^(6+1) = 128 
			hostile = 2^(2+1)  = 8
			+1 above becaus enone is an option

			total states = 125 x 8 x 32 x 128 x 8 = 32768000

			simplify: remove 

		'''
		'''#(hp_encoded, room_matrix, sum_exit_matrix, sum_objmon_matrix)
		hp		R     R     R    R    R     R      R     R     E     E     E          
		[0.448 0.    0.    0.    0.    1.    0.    0.    0.    0.    1.    1.
			E      E     O     O     O     O     O     O    O 
			 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]'''

		total_game_states = total_possible_states / (len(self.exitdict) + 1)
		# fewer game states because exits in rooms are fixed
		print('Total number of possible states: ', total_possible_states)
		print('Likely actual game states ', total_game_states)
		# each state will look like [hp,b,b,b,b,b,b,...] where b is either a 0 or a 1
		'''#example below:		
		#(hp_encoded, room_matrix, sum_exit_matrix, sum_objmon_matrix)
		hp		R     R     R    R    R     R      R     R     E     E     E          
		[0.448 0.    0.    0.    0.    1.    0.    0.    0.    0.    1.    1.
			E      E     O     O     O     O     O     O    O 
			 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
			 '''
		unique_states = set()
		# now count each state
		for m in self.modelthread.memory:
			state, action, reward, new_state, done = m
			unique_states.add(tuple(state[0]))
		print('Unique states in memory: ', len(unique_states))
		percentage_mapped = 100 * (len(unique_states) / total_game_states)
		print('% of accessible game space in data: ', percentage_mapped, '%')
		print('If greater than 100%, estimates wrong or maybe float error on hp value')

	def evaluate_model_loss(self):
		print('model loss called')
		training_cycles = 5000
		results = []
		xparam = np.arange(training_cycles)
		for i in np.arange(training_cycles):
			batch_size = 20
			samples = random.sample(self.modelthread.memory, batch_size)
			# do this the slow way for now, vectorize later for speed
			x_values = []
			y_values = []
			for sample in samples:
				state, action, reward, new_state, done = sample
				target = self.target_model.predict(state)
				x_values.append(state)
				y_values.append(target)

			#print('here we go')
			x_values = np.vstack(x_values)
			y_values = np.vstack(y_values)

			# print('xvals',x_values)
			# print('yvals',y_values)
			#print('xvalshape',x_values.shape)
			#print('yvalshape',y_values.shape)
			#print('stateshape',state.shape)
			# print('targetshape',target.shape)
			#print('made it here')
			result = self.model.evaluate(x_values, y_values, verbose=0)
			results.append(result)
			#print('append')
			# now train model again
			self.replay()
			self.target_train()
		return xparam,results

	def plot_model_loss(self):
		xparam, results = self.evaluate_model_loss()
		#now plot
		pen = pg.mkPen(color=(0, 0, 0))
		self.model_loss_graph.plot(xparam, results, pen=pen)

	def check_if_reset_needed(self):
		if self.world_state.room_name == 'Limbo':
			# after dying reset the player file (deals with de-leveling)
			print('time to reset character, player in Limbo. Reset to Order')
			self.world_state.room_name == 'Order of Love'
			self.reset_player()
			self.world_state.refresh_transient_flags()
			self.world_state.room_name == 'Order of Love'
			#self.initialize_world_state()
		#print('resume event loop after reset')
		self.thread.unpause()


	def reset_player(self):
		self.thread.pause()# set action loop to pause
		save_enabled = True
		if save_enabled:
			print('calling save data and save model')
			self.save_data()
			self.save_model()
		try:
			self.reset_player_file()
			#self.world_state.refresh_transient_flags()
		except:
			print('issue with resetting the player file')
			print(sys.exc_info()[0])
			print(sys.exc_info())

	def write_socket(self,qbytemsg):
		if self.tcpSocket.isValid():
			#print('thinks socket is valid')
			self.tcpSocket.write(qbytemsg)
			#print('write complete',qbytemsg)


		elif not self.tcpSocket.isValid():
			print('socket not valid')
			print(qbytemsg)
			#success = self.reconnect()
			#print('reconnection went ok?',success)
		else:
			print('hmmm')



	def tcpSocketReadyReadEmitted(self):
		try:
			if self.tcpSocket.isValid():
				socket_data = self.tcpSocket.readAll()
				txt = ''
				try:
					txt = str(socket_data)[2:-1] #if tryign to read from socket to soon, socket data will be empty
				except:
					print('issue')
				self.world_state = self.parse_worldstate(txt) #update world_state
			elif not self.tcpSocket.isValid():
				print('bro, not connected.')
			self.display.append(self.cleanText(txt))  # clean text before displaying
			self.display.verticalScrollBar().setValue(self.display.verticalScrollBar().maximum())  # scroll to bottom
		except:
			print('------- tcpSocketReadyReadEmitted ----')
			print(sys.exc_info()[0])
			print(sys.exc_info())

	def store_states_and_memory(self):
		self.world_state_history.append(copy.deepcopy(self.world_state))
		old_world_state = self.world_state_history[-2]
		# print('try decode')
		# decode not functional yet
		# self.decode_state(self.state_array) # print('decode ok')
		self.modelthread.epsilon_array.append(self.modelthread.epsilon)
		self.state_array = self.encode_state(self.world_state)
		self.state_array_history.append(self.state_array)
		'''
		oworld_state = self.world_state_history[-1]
		old_old_world_state = self.world_state_history[-3]
		old_old_old_world_state = self.world_state_history[-4]
		#print('-4',old_old_old_world_state.state_to_string())
		#print('-3',old_old_world_state.state_to_string())
		print('-2',old_world_state.state_to_string())
		#print('-1',oworld_state.state_to_string())
		print('rwd',self.reward)
		#self.actiondict = ['none', 'north', 'east', 'south', 'west', 'killrabbit', 'killraccoon', 'look']
		print(self.last_action)
		#print('prv action',self.previous_action)
		print('curr',self.world_state.state_to_string())
		print('--------------------')
		'''
		if len(self.state_array_history) > 2 * self.state_multiplier:
			# print('generating curr adn old states')
			curr_state_array = self.reshape_x_and_combine(self.state_array_history[-self.state_multiplier:],
														  self.state_multiplier)
			old_state_array = self.reshape_x_and_combine(
				self.state_array_history[-2 * self.state_multiplier:-self.state_multiplier], self.state_multiplier)
			action_index = self.actiondict.index(self.last_action[0])
			reward_to_store = self.calculate_reward_from_state(old_world_state)
			# reward_to_store = self.calculate_reward_from_state_simple(old_world_state)
			self.reward += reward_to_store
			# print(self.reward)
			# print('store memory')
			self.store_memory(old_state_array,
							  action_index,
							  self.reward,
							  curr_state_array,
							  False)  # done initially False?

	def advance_one_step(self):
		try:

			if self.step_counter % self.train_interval == 0: #specified interval
				if self.modelthread.training_allowed == True:
					self.thread.pause()
					try:
						#print('attempt to replay')
						self.modelthread.train_model()

						#print('target_train done')
					except:
						print('could not run replay and target_train successfully')
					self.thread.unpause() #resume actions

				#Partially reset world state. Also partially reset last_action
				#self.world_state.refresh_reward() no reward reset here
			self.world_state.refresh_transient_flags()

			self.previous_action = copy.deepcopy(self.last_action)
			self.last_action = ['none'] #may want to drop this

				# plot every n steps
			if self.step_counter % 100 == 0:
				print('current step: ',self.step_counter,'reward: ',self.reward)
			if self.step_counter % self.plot_interval == 0:
				#self.thread.eventState = -1  # pause action loop
				try:
					print('attempt to plot')
					self.plot_reward()
					self.plot_epsilon(self.modelthread.epsilon_array)
				except:
					print('could not plot')
					#self.thread.eventState = 0  # resume action loop
			self.step_counter += 1

			#reset episode every n steps
			reset_in_progress=True
			if self.step_counter % self.steps_per_episode == 0:
				print('Episode Step Limit Reached')
				msg = 'say resetworld\n'
				bmsg = msg.encode('utf-8')
				self.write_socket(QtCore.QByteArray(bmsg))
				while reset_in_progress:
					try:

							#reset player

						self.thread.pause()
						self.reset_player()
						self.world_state.refresh_transient_flags()
						self.world_state_history=self.world_state_history[-2*self.state_multiplier:-1*self.state_multiplier]
						self.state_array_history=self.state_array_history[-2*self.state_multiplier:-1*self.state_multiplier]
						self.world_state = Worldstate()
						self.state_array = self.encode_state(self.world_state)
					#	self.world_state = Worldstate()
						self.reward = 0
							#self.epsilon = self.epsilon_start
						#update model
						#self.model = copy.deepcopy(self.modelthread.model)


						print('Episode RESET done')
						self.thread.unpause()
						reset_in_progress = False

					except:
						print('error  on episode reset')
						print(sys.exc_info()[0])
						print(sys.exc_info())
					'''if self.step_counter > 5000:
						self.epsilon_start -= 0.01
						self.epsilon_start = np.max([self.epsilon_start, 0.2])
						self.epsilon_decay -= 0.01
						self.epsilon_decay = np.max([self.epsilon_decay, 0.99])
						self.epsilon = self.epsilon_star
						print('updated epsilon')'''
					#self.step_counter =

			self.thread.pause()
			self.check_if_reset_needed()  # check if player needs a reset, and if so apply reset



		except:
			print('------- tcpSocketReadyReadEmitted ----')
			print(sys.exc_info()[0])
			print(sys.exc_info())



	def cleanText(self, txt):  # b'\n\r\x1b[36m
		self.escapelist = ["b\'\\n\\r\\x1b[36m", "b\'\\n\\r", "\\x1b[36m",
						   '\x1b[0m', '\x1b[1m', '\x1b[2m', '\x1b[3m', '\x1b[4m', '\x1b[5m', '\x1b[6m', '\x1b[7m',
						   '\x1b[7m', '\x1b[9m', '\x1b[22m', '\x1b[23m', '\x1b[24m', '\x1b[27m', '\x1b[29m', '\x1b[30m',
						   '\x1b[31m', '\x1b[32m', '\x1b[33m', '\x1b[34m', '\x1b[35m', "\\x1b[32m",
						   '\x1b[37m', '\x1b[37m', 'b\\x1b[37m', '\\x1b[37m',
						   '\x1b[39m', '\x1b[40m', '\x1b[41m', '\x1b[42m', '\x1b[43m', '\x1b[44m', '\x1b[45m',
						   '\x1b[46m', '\x1b[47m', '\x1b[48m', '\x1b[49m']
		self.escapeHTML = ['</font>', '<b>', '<i>', '<u>', '<font color="black">', '<font color="black">',
						   '</b>', '</i>', '</u>', '<font color="black">', '<font color="black">', '<font color="red">',
						   '<font color="red">',
						   '<font color="green">', '<font color="yellow">', '<font color="blue">',
						   '<font color="magneta">',
						   '<font color="cyan">', '<font color="white">', '<font color="white">',
						   '<font color="black">',
						   '<font color="black">', '<font color="black">', '<font color="black">',
						   '<font color="black">',
						   '<font color="black">', '<font color="black">', '<font color="black">',
						   '<font color="black">']
		for k, l in zip(self.escapelist, self.escapeHTML):
			# find escape character
			if txt.find(k) != -1:
				txt = txt.replace(k, "")
		if txt.find("\\n\\r\\n\\r") != -1:
			txt = txt.replace("\\n\\r\\n\\r", "\n")
		if txt.find("\\n\\r\\x1b[32m") != -1:
			txt = txt.replace("\'\\n\\r\\x1b[32m", "\n")
		if txt.find("\\n\\r") != -1:
			txt = txt.replace("\\n\\r", "\n")
		return txt


def main():  ##startup code here
	app = QtWidgets.QApplication(sys.argv)
	ex = MudBotClient()
	exit(app.exec_())


if __name__ == '__main__':
	main()