
from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import time
import myo
import numpy as np
import pandas as pd
import pickle


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):

      self.n = n
      self.lock = Lock()
      self.emg_data_queue = deque(maxlen=n)
      self.orientation = None

  def get_emg_data(self):
      with self.lock:
          return list(self.emg_data_queue)

  def on_connected(self, event):
      event.device.stream_emg(True)

  def on_emg(self, event):
      with self.lock:
          self.emg_data_queue.append((time.time(), event.emg))

  def on_orientation(self, event):
      self.orientation = event.orientation

  def on_orientation_data(self):
      result = list()
      if self.orientation:
          for i in self.orientation:
              result.append(i)
      return result






if __name__ == '__main__':
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(1)
  cont=-1
  cont1=-1
  start=time.time()
  tt=[]
  loaded_model = pickle.load(open('modelo_gbm_myo', 'rb'))
  while hub.run(listener.on_event, 60):
    cont=cont+1
    emg_data = listener.get_emg_data()
    if (int(time.time()) - int(start)) < 2:
      if (len(emg_data) > 0):
       tmp = []
       for v in listener.get_emg_data():
        tmp.append(v[1])
       tmp = list(np.stack(tmp).flatten())
       temp_arr = np.array(tmp)
       tt.append(temp_arr)
       #print(str(temp_arr))
    else:
      if len(tt)>0:
       cont1=cont1+1
       p3=pd.DataFrame(tt)
       f = p3.std()
       f1 = p3.mean()
       f2 = p3.median()
       f3 = p3.quantile(0.25)
       f4 = p3.quantile(0.75)
       d = np.hstack((f, f1, f2, f3, f4))
       d1 = np.zeros((1, 5 * 8))
       for i in range(0, len(d)):
         d1[0, i] = d[i]
       result = loaded_model.predict(d1)
       result = pd.DataFrame(result)
       result = result.replace([0, 1, 2, 3, 4, 5], ['Open Hands', 'Fist', 'Cool', 'Ok', 'PeaceandLove', 'Indicator'])
       print('Para o instante t= ' + str(cont1) + ', o valor previsto foi: ' + str(result))
       start = time.time()
       tt=[]
      # print(str(result))
      else:
       start = time.time()
       tt = []
      #v1=[]

