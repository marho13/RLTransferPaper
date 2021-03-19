import numpy as np
import win32api as wapi
import time
import gym
import cv2
import msvcrt
import os
import re

file_name = "D:/CarracingData/training_data-"

starting_value = 0
keyList = ["\b"]

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

# def key_check():
#     keys = []
#     output = ""
#     for key in keyList:
#         if wapi.GetAsyncKeyState(ord(key)):
#             keys.append(key)
#     return keys

def key_check():
    while True:
        if msvcrt.kbhit():
            key_stroke = msvcrt.getch()
            if key_stroke == b' ':
                return "space"
            return key_stroke.decode("utf-8")


# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]
# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]
# nk = [0,0,0,0,0,0,0,0,1]

def findStartIter():
    max = 0
    files = os.listdir("D:/CarracingData/")
    for f in files:
        num = (re.search('([1-9][0-9]*)', f).group(0))
        if int(num) > max:
            max = int(num)
    print(max)
    return max

def envInit(env_name="CarRacing-v0", render=False):
    env = gym.make(env_name)
    state = env.reset()
    findStartIter()
    return env, state

def actionTranslation(keys):
    actionDict = {"w": np.array([0, 1.0, 0]), "a": np.array([-1.0, 0, 0]), "s": np.array([0, 0, 1]),
                  "d": np.array([1.0, 0, 0]), "wa": np.array([-1.0, 1.0, 0]), "wd": np.array([1.0, 1.0, 0]),
                  "space": np.array([0.0, 0.0, 0.0])}
    # print(key, actionDict)
    try:
        return actionDict[keys]
    except:
        return None

def main(file_name, starting_value):
    env, state = envInit()
    env.render()
    file_name = file_name
    starting_value = starting_value
    training_data = []
#
    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
#
        if not paused:
            # run a color convert:
            state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)

            keys = key_check()
            # print(keys)
            output = actionTranslation(keys)
            if type(output) != type(None):
                env.step(output)
                env.render()
                training_data.append([state, output, keys])
            # last_time = time.time()

            if len(training_data) % 100 == 0 and len(training_data)!=0:
                print(len(training_data))

                if len(training_data) == 500:
                    np.save(file_name, training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'D:/CarracingData/training_data-{}.npy'.format(starting_value)

        # keys = key_check()
            if 't' in keys:
                if paused:
                    paused = False
                    print('unpaused!')

                else:
                    print('Pausing!')
                    paused = True

# #
# #
# main(file_name, starting_value)
# output = key_check()
# print(output, actionTranslation(output))
env, state = envInit()
# env.render()
# while True:
#     if msvcrt.kbhit():
#         key_stroke = msvcrt.getch()
#         print(key_stroke)   # will print which key is pressed