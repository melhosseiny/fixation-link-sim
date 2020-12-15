import csv
import json

import threading
import time
import random

import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.feature_extraction import DictVectorizer
from hmmlearn.hmm import GaussianHMM

import socketio
from aiohttp import web

import asyncio

sio = socketio.AsyncServer(cors_allowed_origins='*', logger=True, engineio_logger=True)
app = web.Application()
sio.attach(app)

@sio.event
def connect(sid, environ):
    print('connect ', sid)
    asyncio.create_task(generate())

gaze_data_from_csv = []

with open('gaze.csv') as f:
    reader = csv.DictReader(f, delimiter=' ')
    for row in reader:
        if int(row['lv']) != 4 and int(row['rv'] != 4):
            gaze_data_from_csv.append({k:float(row[k]) for k in ('r2x','r2y','rp','rprx','rpry','rprz','l2x','l2y','lp','lprx','lpry','lprz')})

vec = DictVectorizer()

X = vec.fit_transform(gaze_data_from_csv).toarray()
print(X)

###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...",)
n_components = 9

# make an HMM instance and execute fit
model = GaussianHMM(n_components, 'full')
print('trying to fit')
model.fit(X)

# predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done\n")

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()

print("means and vars of each hidden state")
for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

async def generate():
    samples = model.sample(1000, random.randint(0,9999999))

    for sample in samples[0]:
        await sio.emit('news', { 'X': (sample[0] + sample[6]) / 2, 'Y': (sample[1] + sample[7]) / 2, 'Timestamp': 159391614.476313 } )
        await asyncio.sleep(0.03)

    asyncio.create_task(generate())

if __name__ == '__main__':
    web.run_app(app, port=80)
