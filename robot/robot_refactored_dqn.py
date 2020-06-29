####
#This version runs a bot in room 500. Bot is meant to play in tandem with a dungeon master that heals them.
#Uploaded for now since code has been cleaned up. This will be converted back into the version that plays starting in room 50
####

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
import telnetlib
from datetime import datetime
from threading import Thread
import numpy as np
from numpy import array
from collections import deque
import random
import copy

from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
##from keras import Input
##from keras.callbacks import TensorBoard
##from keras.engine import Model
# from keras.layers import LSTM, Dense, Embedding, Dot, Dropout, Activation, Flatten
# from keras.layers import MaxPooling1D, MaxPooling2D


# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from PyQt5.QtWidgets import QApplication, QWidget
import pickle


# import math
# import random
# import time

class Modelingthread(QtCore.QThread):
	def __init__(self, parent):  # allow passing parent into the eventthread object
		super(EventThread, self).__init__(parent)
		# import weakref #https://stackoverflow.com/questions/10791588/getting-container-parent-object-from-within-python
		self.state_array = parent.state_array
		self.actiondict = parent.actiondict

	def create_model(self):
		# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
		model = Sequential()
		multiplier = 16
		myinputdim = len(self.state_array) * multiplier  # state array includes actions
		myoutputdim = len(self.actiondict) + 1  # outputs are actions, add one extra for unrecognized actions
		print('input dims: ', myinputdim)

		model.add(Dense(2, input_dim=myinputdim, activation='relu'))
		#model.add(Dense(768, activation='relu'))
		model.add(Dense(2, activation='relu'))

		model.add(Dense(myoutputdim, activation='softmax')) #can try softmax too
		roptimizer = RMSprop(lr=0.0001)
		model.compile(optimizer=roptimizer, loss='mean_squared_error')

		model.summary()
		# Visualize the model
		try:
			plot_model(model, to_file='model.png', show_shapes=True)
		except ImportError as e:
			print('couldnt print model to image', e)
		return model
	def reshape_x(self, x_input):
		# shape for X should be:
		# (num_samples,num_features)
		# reshape to get into the right format for the neural net

		newx = np.stack(x_input, axis=1)
		newx = np.transpose(newx)

		return newx

	def reshape_xy_and_combine(self, x_input, y_input, multiplier):
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

		# now combine the rewards y as well
		y = np.concatenate(y_input, axis=0)
		# print('y',len(y))
		y_corrected = y[extra_rows:]
		# print('y_corr',len(y_corrected))
		# now get average of reward int he combined set of inputs
		# split into segments
		y_segments = np.split(y_corrected, nnn_s)
		# print('y seg', y_segments)
		# get average of each esgment
		newY = np.mean(y_segments, axis=1)
		# print('newY', newY)
		# print('num samples',num_samples, 'updated # inputs', nnn_s)

		return newX, newY

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

	def train_model(self):
		print('here')
		# set action loop to pause
		self.thread.eventState = -1
		self.replay()
		# print('try target_train')
		print('stillhere')
		self.target_train()
		# resume actions
		self.thread.eventState = 0

	def train_model_old(self):
		# print('train model function called')
		# newX=self.reshape_x(self.state_array_history)
		# newX, newY = self.reshape_xy_and_combine(self.state_array_history,self.reward_history, 16)
		newX = self.reshape_x_and_combine(self.state_array_history, 16)
		# careful changing multiplier from 16

		try:

			y = [self.reward_array_history]
			newY = self.combine_y_array(y, 16)

		except:
			print('something went wrong with y array in training')

		try:
			# https://keras.io/api/models/model_training_apis/

			history = self.model.fit(x=newX, y=newY, batch_size=10, epochs=1, verbose=1)

		# print('did we make it here?')
		# pickle.dump(newX, open("newX.p", "wb"))
		# pickle.dump(newY, open("newY.p", "wb"))
		# self.loss_history.append(history.history)
		# ypredict = self.model.predict(newX)
		# print('prediction:', ypredict)
		# print('history:', history.history)

		# save the model
		# serialize model to JSON
		# model_json = self.model.to_json()
		# with open("model.json", "w") as json_file:
		#    json_file.write(model_json)
		except:
			import pickle
			print('Warning! model training code did not work')
			pickle.dump(newX, open("newX.p", "wb"))
			pickle.dump(newY, open("newY.p", "wb"))
			print('dumped with pickle')



	def get_action_from_model(self):
		# print('trying to get action from model')
		# get the latest states
		# newX = self.reshape_x_and_combine(self.state_array_history, 16)
		inputx = self.reshape_x_and_combine(self.state_array_history[-16:], 16)  # careful changing multiplier from 16

		# input = newx[-1].reshape(1, 352) #the latest input is all thats needed , drop rest
		# note we had to reshape the above inptu to use it, needs to be shape 1,352.
		# print('input shape', input.shape)
		# default action is to wait
		best_action = 'look'

		try:

			predict_list = list(self.model.predict(inputx)[0])
			# print('predict_list',predict_list)
			predicted_action_index = predict_list.index(max(predict_list))
			# print('predicted index',predicted_action_index)
			if predicted_action_index == 0:
				predicted_action_index = 1  # switch to wait
			best_action = self.reward_action_tokenizer.index_word[predicted_action_index]
		# print(best_action)
		except:
			print('failed to get prediction array ')
		#    predict_array = self.model.predict(input)
		#    print('prediction array: ', predict_array)
		#    print('action dict: ', self.actiondict)
		# best_action = act

		if best_action == 'killrabbit':
			self.thread.action_from_parent = 'kill rabbit'
		if best_action == 'killraccoon':
			self.thread.action_from_parent = 'kill raccoon'
		if best_action == 'killgolem':
			self.thread.action_from_parent = 'kill golem'

		return best_action

	def save_model(self):
		# save model and architecture to single file
		self.model.save("model.h5")
		self.target_model.save("target_model.h5")
		#model_json = self.model.to_json()
		#with open("model.json", "w") as json_file:
			#json_file.write(model_json)
		print('wrote model to model.h5')

		# also have to load and save weights
		#self.model.save_weights('model_weights.h5')
		#print('weights stored in model_weights.h5')

	def load_model(self):
		# load    model
		self.model = load_model('model.h5')
		self.target_model = load_model('target_model.h5')
		#json_file = open('model.json', 'r')
		#loaded_model_json = json_file.read()
		#json_file.close()
		# load weights too

		#self.model = model_from_json(loaded_model_json)
		#self.model.load_weights("model_weights.h5")
		print('loaded model model.h5')

	def replay(self):
		# print('replay triggered')
		batch_size = 4
		if len(self.memory) < batch_size:
			print('memory too short', len(self.memory))

			return

		samples = random.sample(self.memory, batch_size)
		for sample in samples:
			try:

				state, action, reward, new_state, done = sample
				target = self.target_model.predict(state)
			except:
				print('problem getting state action reward newstate from sample')
			if done:
				target[0][action] = reward
			else:
				Q_future = max(self.target_model.predict(new_state)[0])
				newval = reward + Q_future * self.gamma
				target[0][action] = newval
			self.model.fit(state, target, epochs=1, verbose=0)

	def target_train(self):
		# print('1')
		weights = self.model.get_weights()
		# print('2')
		target_weights = self.target_model.get_weights()
		# print('3')
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		# print('4')
		self.target_model.set_weights(target_weights)

	# print('5')


class EventThread(QtCore.QThread):
	mysignal = QtCore.pyqtSignal(str)  # sends out 1 string

	def __init__(self, parent):  # allow passing parent into the eventthread object
		super(EventThread, self).__init__(parent)
		# import weakref #https://stackoverflow.com/questions/10791588/getting-container-parent-object-from-within-python
		self.parent = parent
		# QtCore.QThread.__init__(self, parent)
		self.exiting = False
		self.action_from_parent = 'look'

	def begin(self):
		#self.start()
		print('Starting Event Loop (Robot Mind)')

	def run(self):
		# main thread for event loop. not called directly, but runs
		# after thread gets setup (self.start()
		# actions = ['score', 'get all','kill rabbit','kill raccoon','wait']
		# exits = ['east', 'west', 'north', 'south']
		# for e in exits:
		#     actions.append('go '+e)
		try:
			self.eventState = -1  # start in off state. start bot button gets bot to idle state (0)
			self.isWalking = 0
			self.isFighting = 0  # initially not fighting
			# status:
			# global variables for event stuff
			# self.hp = 60
			# self.hplimit = 60
			# self.hpHazy = 15
			# self.mp = 5
			# self.monsterList = []  # initially no monsters in room
			# self.possibleTarget = 0  # no target
			# self.target = 0  # no target

			self.max_steps = 1000000
			self.steps = 0
			self.train_interval = 10000
			self.steps_until_train = self.train_interval  # try increment of 32 steps may be enough not sure

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
			logtimer = datetime.now()
			exptimer = datetime.now()

			# update current exp once
			already_updated = 0

			while self.exiting == False:
				if self.eventState == 99:
					print('quitting event Thread')
					self.exiting = True
				while self.eventState == 0:
					currenttimer = datetime.now()

					# limit actions every 1 second for now
					diff = currenttimer - logtimer
					expdiff = currenttimer - exptimer

					# self tick timer. go every half
					# 0.5 seconds for code development
					# change to 1 or 2 sec for real bot

					# get milliseconds
					# currently aim for 50 ms ticks
					#get number of milliseconds

					#slow version
					delta_milli = diff.seconds

					#fast version - every 100 ms right now
					#delta_milli = diff.microseconds / 1000.0
					#delta_milli = delta_milli/100

					if delta_milli >= 2:

						# act = random.choice(actions) #do a random action once per tick
						# do action from parent
						act = self.action_from_parent
						self.steps_until_train = self.steps_until_train - 1
						# print('action is',act)
						if act != 'none':  # if none command do nothing
							cmd = 'socket>*' + act + ' \n'
							self.mysignal.emit(cmd)
							self.steps = self.steps + 1

							if self.steps > self.max_steps:
								self.eventState = 99
								print('reached max steps')

						elif act == 'none':  # send a blank line, needed to get reward for waiting
							cmd = 'socket>*' + ' \n'
							self.mysignal.emit(cmd)
							self.steps = self.steps + 1

						# self.emit(QtCore.SIGNAL("output(QString)"), cmd)
						newcmd = 'request_action>*'
						self.mysignal.emit(newcmd)
						logtimer = datetime.now()

					if self.steps_until_train <= 0:
						traincmd = 'train_model>*'
						self.mysignal.emit(traincmd)
						self.steps_until_train = self.train_interval  # train every 160 steps, after initial training period.

					if expdiff.microseconds / 1000 >= 300000:
						# update reward
						cmd = 'socket>*' + 'score' + ' \n\r'
						self.mysignal.emit(cmd)
						exptimer = datetime.now()


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

	def state_to_string(self):
		selfstrings = []
		selfstrings.append(self.hp)
		#selfstrings.append(self.mp)
		selfstrings.append(self.room_name)
		# selfstrings.append(self.room_description)
		selfstrings.append(str(self.exits))
		selfstrings.append(str(self.objmonlist))
		# selfstrings.append(self.delaystr)
		# selfstrings.append(self.hostile_str)
		# selfstrings.append(self.damage_str)
		# selfstrings.append(str(self.last_action))
		return selfstrings
	# print('statestring: ',selfstrings)


class MudBotClient(QtWidgets.QWidget):
	def __init__(self):
		# start user interface and socket
		super(MudBotClient, self).__init__()
		self.iniMain()
		self.initialize_variables()
		# start event handling
		self.iniEventThread()

	def initialize_variables(self):
		self.initialize_encoders()
		self.initialize_self()
		self.initialize_world_state()
		self.initialize_model()

	def initialize_encoders(self):
		# initialize dictionaries
		# self.objmondict = ['rabbit', 'rabbits', 'raccoon', 'raccoons', 'book', 'gold']
		self.objmondict = ['golem', 'golems', 'gold']
		# self.exitdict = ['north', 'east', 'south', 'west']
		self.exitdict = ['none']
		# self.room_name_dict = ['Limbo', 'Love', 'Brownhaven', 'Alley', 'Pawn', 'Path ', 'Petting']
		self.room_name_dict = ['Limbo', 'Love']
		# self.actiondict = ['wait', 'north', 'east', 'south', 'west', 'killrabbit', 'killraccoon', 'look']
		# self.actiondict = ['north', 'east', 'south', 'west', 'killrabbit', 'look']
		self.actiondict = ['none','killgolem', 'look']

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

	def initialize_self(self):
		self.maxhp = 125
	def initialize_model(self):
		# initialize_model
		self.memory = deque(maxlen=10000)
		self.epsilon = 1  # threshold for action to be random, will decay to .05
		self.training_allowed = True
		# self.training_allowed=False
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.999
		self.gamma = 0.99  # google uses 0.99
		self.tau = 0.125
		# create the model
		self.model = self.create_model()
		self.target_model = self.create_model()

	def initialize_world_state(self):
		# initialize world state
		self.last_action = ['none']
		self.reward = 0
		self.world_state = Worldstate()
		self.state_array = self.encode_state(self.world_state)
		self.world_state_history = []
		self.state_array_history = []
		self.reward_history = []
		self.reward_array_history = []
		self.loss_history = []



	def store_memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def encode_state(self, state):

		# start with any empty matrices that would be needed
		backup_sum_objmon_matrix = np.array(np.zeros(len(self.objmondict) + 1))
		backup_sum_exit_matrix = np.array(np.zeros(len(self.exitdict) + 1))
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

		objmon_matrix = self.objmon_tokenizer.texts_to_matrix(state.objmonlist)
		action_encoded = sum(self.action_tokenizer.texts_to_matrix([self.last_action]))

		if len(objmon_matrix) == 0:
			sum_objmon_matrix = backup_sum_objmon_matrix
		else:
			sum_objmon_matrix = sum(objmon_matrix)
			# only allow values to be equal to 0 or 1. For gold coins piles seem to stack and can get 2,3, etc
			sum_objmon_matrix = np.where(sum_objmon_matrix > 0.5, 1.0, 0.0)
		#print(state.hostile_str)
		state_array = np.concatenate((hp_encoded, room_matrix, sum_exit_matrix, sum_objmon_matrix, action_encoded),
									 axis=0)

		return state_array

	def encode_rewards(self, state):
		action_encoded = sum(self.reward_action_tokenizer.texts_to_matrix([self.last_action]))
		state_reward = state.reward
		# print('action',self.last_action, 'reward: ', state_reward, 'encoded act', action_encoded)
		reward_array = state_reward * action_encoded
		return reward_array

	def create_model(self):
		# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
		model = Sequential()
		multiplier = 16
		myinputdim = len(self.state_array) * multiplier  # state array includes actions
		myoutputdim = len(self.actiondict) + 1  # outputs are actions, add one extra for unrecognized actions
		print('input dims: ', myinputdim)

		model.add(Dense(2, input_dim=myinputdim, activation='relu'))
		#model.add(Dense(768, activation='relu'))
		model.add(Dense(2, activation='relu'))

		model.add(Dense(myoutputdim, activation='softmax')) #can try softmax too
		roptimizer = RMSprop(lr=0.0001)
		model.compile(optimizer=roptimizer, loss='mean_squared_error')

		model.summary()
		# Visualize the model
		try:
			plot_model(model, to_file='model.png', show_shapes=True)
		except ImportError as e:
			print('couldnt print model to image', e)
		return model

	def listtostring(self, mylistobj):
		thestring = ''.join(str(x) + ' ' for x in mylistobj)
		return thestring

	def iniEventThread(self):
		# start a thread to handle events
		# eventThread=Thread(None, self.eventHandler, None, (), {})
		# eventThread.start()

		# start a QThread instead, use class EventThread
		self.thread = EventThread(self)  # pass self into the thread so that parent functions can be accessed
		#start the thread
		# add: clicking ont he button stats the startLoop funciton which calls
		# a thread class function which starts the loop
		# QtCore.QObject.connect(self.botStart, QtCore.SIGNAL("clicked()"), self.startBot)

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
		elif str(qstring).find('socket>') != -1:
			socketString = str(qstring).split('*')
			header = socketString[0]
			msg = socketString[1]
			# write to socket
			bmsg = msg.encode('utf-8')
			self.tcpSocket.write(QtCore.QByteArray(bmsg))
			self.display.append("EventLoop>" + msg)

		elif str(qstring).find('request_action>*') != -1:
			self.generate_new_action()  # this will update the next action for the child


		elif str(qstring).find('train_model>*') != -1:
			# print('skipping train_model>* request')
			self.train_model()

		# check for target info
		elif str(qstring).find('possibleTarget>') != -1:
			socketString = str(qstring).split('*')
			header = socketString[0]
			self.possibleTarget = socketString[1]

	def reshape_x(self, x_input):
		# shape for X should be:
		# (num_samples,num_features)
		# reshape to get into the right format for the neural net

		newx = np.stack(x_input, axis=1)
		newx = np.transpose(newx)

		return newx

	def reshape_xy_and_combine(self, x_input, y_input, multiplier):
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

		# now combine the rewards y as well
		y = np.concatenate(y_input, axis=0)
		# print('y',len(y))
		y_corrected = y[extra_rows:]
		# print('y_corr',len(y_corrected))
		# now get average of reward int he combined set of inputs
		# split into segments
		y_segments = np.split(y_corrected, nnn_s)
		# print('y seg', y_segments)
		# get average of each esgment
		newY = np.mean(y_segments, axis=1)
		# print('newY', newY)
		# print('num samples',num_samples, 'updated # inputs', nnn_s)

		return newX, newY

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

	def train_model(self):
		# set action loop to pause
		self.thread.eventState = -1
		self.replay()
		# print('try target_train')
		self.target_train()
		# resume actions
		self.thread.eventState = 0



	def generate_new_action(self):
		epsilon = self.epsilon
		print(epsilon)
		if epsilon == 1 or (epsilon > 0 and 1 > epsilon > random.random()):
			action = random.choice(self.actiondict)
			self.last_action = [action]
			# print('chose action: ', action)
			if action == 'killrabbit':
				self.thread.action_from_parent = 'kill rabbit'
			elif action == 'killraccoon':
				self.thread.action_from_parent = 'kill raccoon'
			elif action == 'killgolem':
				self.thread.action_from_parent = 'kill golem'
			else:
				self.thread.action_from_parent = action

		else:
			# calculate state with maximum Q value
			action = self.get_action_from_model()
			self.last_action = [action]
			#print('model chose action: ', action)
			if action == 'killrabbit':
				self.thread.action_from_parent = 'kill rabbit'
			elif action == 'killraccoon':
				self.thread.action_from_parent = 'kill raccoon'
			elif action == 'killgolem':
				self.thread.action_from_parent = 'kill golem'
			else:
				self.thread.action_from_parent = action
		self.epsilon = self.epsilon * self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)

	def get_action_from_model(self):
		# print('trying to get action from model')
		# get the latest states
		# newX = self.reshape_x_and_combine(self.state_array_history, 16)
		inputx = self.reshape_x_and_combine(self.state_array_history[-16:], 16)  # careful changing multiplier from 16

		# input = newx[-1].reshape(1, 352) #the latest input is all thats needed , drop rest
		# note we had to reshape the above inptu to use it, needs to be shape 1,352.
		# print('input shape', input.shape)
		# default action is to wait
		best_action = 'look'

		try:

			predict_list = list(self.model.predict(inputx)[0])
			# print('predict_list',predict_list)
			predicted_action_index = predict_list.index(max(predict_list))
			# print('predicted index',predicted_action_index)
			if predicted_action_index == 0:
				predicted_action_index = 1  # switch to wait
			best_action = self.reward_action_tokenizer.index_word[predicted_action_index]
		# print(best_action)
		except:
			print('failed to get prediction array ')
		#    predict_array = self.model.predict(input)
		#    print('prediction array: ', predict_array)
		#    print('action dict: ', self.actiondict)
		# best_action = act

		if best_action == 'killrabbit':
			self.thread.action_from_parent = 'kill rabbit'
		if best_action == 'killraccoon':
			self.thread.action_from_parent = 'kill raccoon'
		if best_action == 'killgolem':
			self.thread.action_from_parent = 'kill golem'

		return best_action



	def iniMain(self):
		self.iniSockets()
		self.initUI()
		self.connect_buttons()

	def iniSockets(self):
		# load telnet objects
		self.HOST = 'localhost'
		self.port = 4000

		# make a QT socket
		self.tcpSocket = rr = QtNetwork.QTcpSocket()
		# telnet
		self.tn = telnetlib.Telnet(self.HOST, self.port)
		# get the socket from telnetlib and feed it to the QtNetwork QTCSocket object!
		self.tcpSocket.setSocketDescriptor(self.tn.sock.fileno())
		self.login()

	def login(self):
		# auto login, set flags (clear long)
		# add b before strings to turn them into bytes
		# txt = b'test' + b'\n' + b'asdfasdf' + b'\r\n' #+ 'clear long\r\n' + 'set auto' + '\r\n'
		text = 'tester\nasdfasdf\r\n'
		btext = text.encode('utf-8')

		self.tcpSocket.write(QtCore.QByteArray(btext))
		#reset hp when starting out
		text = 'bleed\r\n'
		btext = text.encode('utf-8')
		self.tcpSocket.write(QtCore.QByteArray(btext))

	def initUI(self):
		# initialize layouts
		self.lay = QtWidgets.QVBoxLayout(self)  # vertical box must come first
		self.exitBox = QtWidgets.QHBoxLayout(self)
		self.mapCmdsBox = QtWidgets.QHBoxLayout(self)
		self.action_box = QtWidgets.QHBoxLayout(self)

		# create widget/box to show output from mud
		self.display = QtWidgets.QTextBrowser()
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
		self.mapCmdsBox.addWidget(self.save_model_btn)
		self.mapCmdsBox.addWidget(self.load_model_btn)
		self.mapCmdsBox.addWidget(self.save_data_btn)
		self.mapCmdsBox.addWidget(self.load_data_btn)
		# widget for health and mana
		self.hpmp = QtWidgets.QStatusBar()
		self.hpmp.showMessage('Hp:    Mp:    Xp:')

		# widget for monsters in room
		self.roomMonsters = QtWidgets.QStatusBar()
		self.roomMonsters.showMessage('monsters: ')

		# widget for items in room
		self.roomItems = QtWidgets.QStatusBar()
		self.roomItems.showMessage('items: ')

		# finalize layout:

		self.lay.addWidget(self.roomInfo)
		# self.lay.addWidget(self.roomExits)

		# self.lay.insertLayout(3,self.exitBox)
		self.lay.insertLayout(4, self.mapCmdsBox)
		self.lay.insertLayout(4, self.action_box)

		self.lay.addWidget(self.roomMonsters)
		self.lay.addWidget(self.roomItems)
		self.lay.addWidget(self.hpmp)
		self.lay.addWidget(self.display)
		self.lay.addWidget(self.lineInput)

		# self.lay.addLayout(self.exitBox)

		# setup window size and icon
		self.setGeometry(100, 100, 600, 900)
		self.setWindowTitle('Mud Client Botter')
		self.setWindowIcon(QtGui.QIcon('Robot-icon.png'))

		self.show()
	def connect_buttons(self):
		self.botStart.clicked.connect(self.startBot)
		self.save_model_btn.clicked.connect(self.save_model)
		self.load_model_btn.clicked.connect(self.load_model)
		self.train_model_btn.clicked.connect(self.train_model)
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

		self.tcpSocket.write(QtCore.QByteArray(bmsg))


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

		self.tcpSocket.write(QtCore.QByteArray(btxt))
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
		self.thread.eventState = 0
		print('set thread eventState to 0')

	def save_data(self):
		print('save data function called')

		pickle.dump(self.state_array_history, open("Xdata.p", "wb"))
		pickle.dump(self.reward_array_history, open("Ydata.p", "wb"))
		pickle.dump(self.memory, open("memory.p", "wb"))

	def load_data(self):
		print('load data function called')
		self.state_array_history = pickle.load(open("Xdata.p", "rb"))
		self.reward_array_history = pickle.load(open("Ydata.p", "rb"))
		self.memory = pickle.load(open("memory.p", "rb"))

	def save_model(self):
		# save model and architecture to single file
		self.model.save("model.h5")
		self.target_model.save("target_model.h5")
		#model_json = self.model.to_json()
		#with open("model.json", "w") as json_file:
			#json_file.write(model_json)
		print('wrote model to model.h5')

		# also have to load and save weights
		#self.model.save_weights('model_weights.h5')
		#print('weights stored in model_weights.h5')

	def load_model(self):
		# load    model
		self.model = load_model('model.h5')
		self.target_model = load_model('target_model.h5')
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
	def parse_worldstate(self, txt):
		# returns a worldstate object
		# print('----------------oneliner----------------')
		# print(txt)
		# print('----------------parsed_state----------------')

		#newstate = copy.deepcopy(self.world_state)
		#try instead of a deep copy
		newstate = Worldstate()

		#pull in some info from the old state
		try:
			newstate.reward = self.world_state.reward
			newstate.last_action = self.world_state.last_action
			newstate.room_name = self.world_state.room_name
			newstate.room_description = self.world_state.room_description
			#newstate.objmonlist = self.world_state.objmonlist

		except:
			pass
		# consider reset of award
		# self.reward = 0
		newstate.reward = self.reward
		status, room_name = self.get_room_name(txt)

		if status == True:
			newstate.room_name = room_name
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
			if newstate.hp < 50:
				self.thread.action_from_parent = 'jump'
				newstate.reward = self.reward
			elif newstate.hp < 30:
				self.thread.action_from_parent = 'bleed'
				newstate.reward = self.reward

		status, delaystr = self.getdelaystr(txt)
		if status == True:
			newstate.delaystr = delaystr
		status, hostile_str = self.get_hostile_str(txt)
		if status == True:
			newstate.hostile_str = hostile_str

		status, damage_str = self.got_hit_str(txt)
		if status == True:
			newstate.damage_str = damage_str

		status, expstr = self.get_experience(txt)
		if status == True:
			if self.oldExp == 0:  # not updated yet
				self.oldExp = int(expstr)
			elif self.oldExp > 0:
				self.newExp = int(expstr)

		status, killrewardstr = self.get_kill_reward(txt)
		if status == True:
			self.reward = self.reward + 125  # reward for killing a mosnter is big
			newstate.reward = self.reward
		#print('reward is: ', self.reward)
		# print('last action is:',self.last_action)

		status, hitrewardstr = self.get_hit_reward(txt)
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
		status, monstermissstring = self.get_monstermiss_reward(txt)
		if status == True:
			# reward/penalty for engaging in combat
			if newstate.hp < 50:
				self.reward = self.reward - 50
				newstate.reward = self.reward
			elif newstate.hp > 90:
				self.reward = self.reward + 40 * ((newstate.hp-90) / (35))
				newstate.reward = self.reward
			# 50 - 100 scaling penalty
			elif newstate.hp < 90:
				# goes from 0 to -5 as health decreases
				self.reward = self.reward - 40 * ((90 - newstate.hp) /40)
				newstate.reward = self.reward


		# print('reward is: ', self.reward)
		# print('last action is:',self.last_action)

		# status, movepenaltystr = self.get_move_penalty(txt)
		# if status == True:
		# self.reward = self.reward - 0.01 #small negative rewar
		# newstate.reward = self.reward

		status, nomonsterpenaltystr = self.get_nomonster_penalty(txt)
		if status == True:
			# self.reward = self.reward -0.001  # small negative rewar
			newstate.reward = self.reward

		status, killedstr = self.get_killed_penalty(txt)
		if status == True:
			self.reward = self.reward - 1000  # big negative penalty for dying
			newstate.reward = self.reward

		# give nice bonus for ticking
		if newstate.room_name == 'Order of Love':
			if newstate.hp < self.maxhp:
				# print('should be getting a bonus, are we?')
				self.reward = self.reward + (self.maxhp-newstate.hp)/43 +0.05 #big positive reward for ticking, modulated by differential
				newstate.reward = self.reward
			elif newstate.hp < 50:
				# self.reward = self.reward + 1
				#self.thread.action_from_parent = 'jump'
				newstate.reward = self.reward

			elif newstate.hp == self.maxhp:
				# self.reward = self.reward -.1 #small negative reward for spending too much time in tick room
				newstate.reward = self.reward

		# give penalty for not  ticking
		if newstate.room_name != 'Order of Love':
			if newstate.hp < 30:
				# print('should be getting a bonus, are we?')
				# self.reward = self.reward - (43-newstate.hp)/43 -0.05 #big positive reward for ticking, modulated by differential
				newstate.reward = self.reward

		# hardcode escape from limbo
		if newstate.room_name == 'Limbo':
			self.thread.action_from_parent = 'go green'
		if newstate.room_name == 'The Tree of Life':
			self.thread.action_from_parent = 'go down'
		start_indicator = "I don"
		end_indicator = "see that exit."
		xstatus, noxtstr = self.pull_text(txt, start_indicator, end_indicator)
		if xstatus == True:
			self.thread.action_from_parent = 'look'


		# state dictionary for reference below
		# {'hp': 0, 'mp': 0, 'room_name': '', 'room_description': '', 'exits': [], 'objmonlist': [], 'delaystr': '',
		# 'hostile_str': '', 'damage_str': '', 'reward': 0}

		# return thenew state
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
		start_indicator = 'missed'
		end_indicator = 'you'
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

	def tcpSocketReadyReadEmitted(self):
		try:
			socket_data = self.tcpSocket.readAll()

			# print(socket_data)
			txt = str(socket_data)[2:-1]
			self.world_state_history.append(self.world_state)
			# generate new state
			newstate = self.parse_worldstate(txt)
			self.world_state = newstate
			# print('worldstate: ',self.world_state.state_to_string(),'reward: ',newstate.reward)
			# print(len(self.world_state_history))

			self.state_array = self.encode_state(self.world_state)
			self.reward_array = self.encode_rewards(self.world_state)
			self.reward_array_history.append(self.reward_array)
			# self.objmondict = ['rabbit', 'rabbits', 'raccoon', 'raccoons', 'book', 'gold']
			# self.exitdict = ['north', 'east', 'south', 'west']
			# self.room_name_dict = ['Limbo', 'Love', 'Brownhaven', 'Alley', 'Pawn', 'Path ', 'Petting']
			# self.actiondict = ['wait', 'north', 'east', 'south', 'west', 'killrabbit', 'killraccoon', 'look']
			# print('[H. _. R. R. R. R. R. R. R. _. E. E. E. E. _. O. O. O. O. O. O. _. A. A. \nA. A. A. A. A. A.]')
			# )
			# print(self.state_array)
			# print(len(self.state_array))
			self.state_array_history.append(self.state_array)
			self.reward_history.append(np.array([self.reward]))  # update reward too. Note covnersion to np.array
			#print('world state is', self.state_array)
			#print('action was', self.last_action, 'reward', self.reward)
			# print(' [_. W. N. E. S. W. K. k. L.]')
			# print('-',np.where(self.reward_array < 0, 1., 0.))
			# print('[_. W. N. E. S. W. K. k. L.]')
			# print(np.where(self.reward_array > 0, 1., 0.))

			if len(self.state_array_history) > 32:
				# print('generating curr adn old states')
				curr_state = self.reshape_x_and_combine(self.state_array_history[-16:], 16)
				old_state = self.reshape_x_and_combine(self.state_array_history[-32:-16], 16)
				# encode action
				# print('get action index',self.last_action[0])
				# print('hmmm',self.actiondict.index(self.last_action[0]))
				action_index = self.actiondict.index(self.last_action[0])
				# print(action_index)
				# print('store memory')

				self.store_memory(old_state,
								  action_index,
								  self.world_state.reward,
								  curr_state,
								  False)  # done initially False?
				# print('replay')
				#set action loop to pause
				if self.training_allowed == True:
					self.thread.eventState = -1
					try:

						self.replay()
						self.target_train()
					except:
						print('could not run replay and target_train successfully')
					#resume actions
					self.thread.eventState = 0

			# reset last action
			self.last_action = ['none']
			# clean text
			txt = self.cleanText(txt)
			self.display.append(txt)
			# scroll to bottom
			self.display.verticalScrollBar().setValue(self.display.verticalScrollBar().maximum())
		except:
			print('------- tcpSocketReadyReadEmitted ----')
			print(sys.exc_info()[0])
			print(sys.exc_info())

	def replay(self):
		# print('replay triggered')
		batch_size = 4
		if len(self.memory) < batch_size:
			print('memory too short', len(self.memory))

			return

		samples = random.sample(self.memory, batch_size)
		for sample in samples:

			state, action, reward, new_state, done = sample
			target = self.target_model.predict(state)
			if done:
				target[0][action] = reward
			else:
				Q_future = max(self.target_model.predict(new_state)[0])
				newval = reward + Q_future * self.gamma
				target[0][action] = newval
			self.model.fit(state, target, epochs=1, verbose=0)

	def target_train(self):
		# print('1')
		weights = self.model.get_weights()
		# print('2')
		target_weights = self.target_model.get_weights()
		# print('3')
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		# print('4')
		self.target_model.set_weights(target_weights)

	# print('5')

	def splitString(self, stri):
		# a function to take a comma separated string and get out a list
		splitList = []
		while len(stri) > 1:
			i = stri.find(',')
			if i != -1:  # if found comma
				item = stri[:i]  # grab everything up to comma
				splitList.append(item)  # add item to list
				stri = stri[i + 2:]  # strip string
			elif i == -1:  # out of commads, rest of string is an entry
				splitList.append(stri)
				stri = ''
		return splitList



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