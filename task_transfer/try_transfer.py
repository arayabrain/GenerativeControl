import numpy as np
import gym
import matplotlib.pyplot

from math import *

import theano
import theano.tensor as T

import lasagne

from PIL import Image
import scipy.misc
import sys
import pickle

np.random.seed(5342)

env = gym.make('CartPole-v1')

weights = np.array([1.2, 1.8, 3.0, 0.8])

LATENT = 2
FUTURE = 16
HIDDEN = 256

context_var = T.matrix()
latent = T.matrix()
d_input = T.matrix()
target = T.vector()
targs = T.vector()

context_input = lasagne.layers.InputLayer((None,4), input_var = context_var)
latent_input = lasagne.layers.InputLayer((None,LATENT), input_var = latent)
state_input = lasagne.layers.InputLayer((None,FUTURE*5), input_var = d_input)

plist = []
dense1 = lasagne.layers.DenseLayer(state_input, num_units = HIDDEN)
plist.append(dense1.W)
plist.append(dense1.b)
dense2 = lasagne.layers.DenseLayer(dense1, num_units = HIDDEN)
plist.append(dense2.W)
plist.append(dense2.b)
dense3 = lasagne.layers.DenseLayer(dense2, num_units = HIDDEN)
plist.append(dense3.W)
plist.append(dense3.b)
enc = lasagne.layers.DenseLayer(dense3, num_units = LATENT, nonlinearity = lasagne.nonlinearities.tanh)
plist.append(enc.W)
plist.append(enc.b)
enc_noise = lasagne.layers.GaussianNoiseLayer(enc, sigma=0.2)
stack2 = lasagne.layers.ConcatLayer([context_input, enc_noise])
ddense1 = lasagne.layers.DenseLayer(stack2, num_units = HIDDEN)
plist.append(ddense1.W)
plist.append(ddense1.b)
ddense2 = lasagne.layers.DenseLayer(ddense1, num_units = HIDDEN)
plist.append(ddense2.W)
plist.append(ddense2.b)
ddense3 = lasagne.layers.DenseLayer(ddense2, num_units = HIDDEN)
plist.append(ddense3.W)
plist.append(ddense3.b)
out = lasagne.layers.DenseLayer(ddense3, num_units = 5*FUTURE, nonlinearity = None)
plist.append(out.W)
plist.append(out.b)

def addBlock(ctx_in, state_in, params):
	dense1 = lasagne.layers.DenseLayer(state_in, num_units = HIDDEN, W=params[0], b=params[1])
	dense2 = lasagne.layers.DenseLayer(dense1, num_units = HIDDEN, W=params[2], b=params[3])
	dense3 = lasagne.layers.DenseLayer(dense2, num_units = HIDDEN, W=params[4], b=params[5])
	enc = lasagne.layers.DenseLayer(dense3, num_units = LATENT, nonlinearity = lasagne.nonlinearities.tanh, W=params[6], b=params[7])
	enc_noise = lasagne.layers.GaussianNoiseLayer(enc, sigma=0.2)
	stack2 = lasagne.layers.ConcatLayer([ctx_in, enc_noise])
	ddense1 = lasagne.layers.DenseLayer(stack2, num_units = HIDDEN, W=params[8], b=params[9])
	ddense2 = lasagne.layers.DenseLayer(ddense1, num_units = HIDDEN, W=params[10], b=params[11])
	ddense3 = lasagne.layers.DenseLayer(ddense2, num_units = HIDDEN, W=params[12], b=params[13])
	out = lasagne.layers.DenseLayer(ddense2, num_units = 5*FUTURE, nonlinearity = None, W=params[14], b=params[15])
	
	return enc, out

enc2, out2 = addBlock(context_input, out, plist)
enc3, out3 = addBlock(context_input, out2, plist)
enc4, out4 = addBlock(context_input, out3, plist)
enc5, out5 = addBlock(context_input, out4, plist)
enc6, out6 = addBlock(context_input, out5, plist)
enc7, out7 = addBlock(context_input, out6, plist)

params = lasagne.layers.get_all_params(out7,trainable=True)

outs = lasagne.layers.get_output([out,out2,out3,out4,out5,out6,out7])
encs = lasagne.layers.get_output([enc,enc2,enc3,enc4,enc5,enc6,enc7])

loss = 0
for i in range(len(outs)):
	loss = loss + T.mean((outs[i] - d_input)**2)
	
reg = lasagne.regularization.regularize_network_params(out7, lasagne.regularization.l2)*5e-4
lr = theano.shared(np.array([5e-4],dtype=np.float32))

updates = lasagne.updates.adam(loss+reg, params, learning_rate = lr[0], beta1=0.5)

train = theano.function([context_var, d_input], loss, updates=updates, allow_input_downcast=True)
encode = theano.function([d_input], encs[0], allow_input_downcast=True)
stack2.input_layers[1] = latent_input
gen_out = lasagne.layers.get_output(out)

reward = T.mean(weights[0]*abs(gen_out[:,0+5*(FUTURE-1)]-targs[0]))+T.mean(T.sum(weights[1:]*(gen_out[:,1+5*(FUTURE-1):5*(FUTURE-1)+4]-targs[1:])**2,axis=1),axis=0)

sample = theano.function([context_var, latent], gen_out, allow_input_downcast = True)
latent_grad = theano.function([context_var, latent, targs], [theano.grad(reward, latent), reward], allow_input_downcast = True)	

lasagne.layers.set_all_param_values(out7,pickle.load(open("network.params","rb")))

def getPolicy(obs, targ, platent):
	latent = platent.copy()
	obs2 = obs
	for i in range(100):
		grad,rw = latent_grad(obs2, latent, targ)
		grad = -grad/np.sqrt(np.sum(grad**2,axis=1)+1e-16)
		latent += 0.05*grad - 0.001*latent
	return sample(obs2, latent)[0], latent

def attemptTask(amp, period):			
	obs = env.reset()
	obs[0] *= 10
	obs[2] *= 10
	latent = np.random.randn(1,LATENT)
	targ = np.zeros(4)

	policy,latent = getPolicy(np.array(obs).reshape((1,4)),targ,latent)
	done = False

	xtarg = []
	xpos = []
	
	step = 0
	j = 0
	
	while (not done) and (step<5000):
		targ[0] = 10*amp*sin(2*3.1415*step/period)
		act = (np.random.rand()<(0.5*(policy[4+j*5]+1)))*1
		obs, reward, done, info = env.step(act)
		obs[0] *= 10
		obs[2] *= 10
		err = np.mean( (obs-policy[j*5:j*5+4])**2 )
		xtarg.append(targ[0]/10.0)
		xpos.append(obs[0]/10.0)
		
		j += 1
		
		if j>1 or err>0.05:
			policy,latent = getPolicy(np.array(obs).reshape((1,4)),targ,latent)
			j = 0
			
		step += 1

	return step, np.hstack([np.array(xpos).reshape((len(xpos),1)), np.array(xtarg).reshape((len(xpos),1))])

amp = float(sys.argv[1])
period = float(sys.argv[2])

r,x = attemptTask(amp,period)

np.savetxt("traj/%.3g_%.3g.txt" % (amp,period), x)

rewards = []
for i in range(20):
	r,x = attemptTask(amp,period)
	rewards.append(r)

f = open("performance.txt","a")
f.write("%.6g %.6g %.6g\n" % (amp, period, np.mean(rewards)))
f.close()
