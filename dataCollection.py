import numpy as np
import win32api as wapi
import time
import gym

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def envInit(env_name="Carracing-v0", render=False):
    env = gym.make(env_name)
    state = env.reset()
    return env, state

def actionTranslation():
    actionDict = {"w": np.array([[0, 1.0, 0]]), "a": np.array([[-1.0, 0, 0]]), "s": np.array([[0, 0, 1]]),
                  "d": np.array([[1.0, 0, 0]]), "wa": np.array([[-1.0, 1.0, 0]]), "wd": np.array([[1.0, 1.0, 0]])}
    try:
        act = input("Which action to take?").lower()
        return actionDict[act]

    except:
        action = actionTranslation()
        return action


# def main(file_name, starting_value):
#     env, state = envInit()
#     file_name = file_name
#     starting_value = starting_value
#     training_data = []
#     for i in list(range(4))[::-1]:
#         print(i+1)
#         time.sleep(1)
#     state = env.reset()
#     last_time = time.time()
#     paused = False
#     print('STARTING!!!')
#     while(True):
#
#         if not paused:
#             # run a color convert:
#             state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
#
#             keys = key_check()
#             output = keys_to_output(keys)
#             #env.act(output)
#             training_data.append([state, output])
#
#             # last_time = time.time()
#
#             if len(training_data) % 100 == 0:
#                 print(len(training_data))
#
#                 if len(training_data) == 500:
#                     np.save(file_name, training_data)
#                     print('SAVED')
#                     training_data = []
#                     starting_value += 1
#                     file_name = 'D:/CarracingData/training_data-{}.npy'.format(starting_value)
#
#         keys = key_check()
#         if 'T' in keys:
#             if paused:
#                 paused = False
#                 print('unpaused!')
#                 time.sleep(1)
#             else:
#                 print('Pausing!')
#                 paused = True
#                 time.sleep(1)
#
#
# main(file_name, starting_value)