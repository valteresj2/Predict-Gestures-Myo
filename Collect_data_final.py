

from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import time
import myo
import numpy as np
import pandas as pd
from time import gmtime, strftime


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)
    self.orientation=None
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
      result=list()
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
  target=['Ok','Cool','Fist','Open Hands','PeaceandLove','Indicator']
  cont2=-1
  nobs=30
  while hub.run(listener.on_event, 60):
    emg_data = listener.get_emg_data()
    #emg_data1= listener.on_orientation_data()
    #print(str(emg_data1))
    if (int(time.time()) - int(start)) < 1:
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
       cont = cont + 1
       cont1=cont1+1
       p3=pd.DataFrame(tt)
       p3['index']=cont
       d1=p3

       if cont1 == 0:
         if cont2 == -1:
          print('Execute gesture ' + target[0])
         p4 = d1
       elif cont1 > 0:
         p4 = np.vstack((p4, d1))

       if cont1 == nobs and cont2 <= 4:
         cont2=cont2+1
         p5 = pd.DataFrame(p4)
         target1 = target[cont2]
         p5['target'] = target1
         cont1 = -1
         if cont2==0:
             p6=p5
         else:
             p6=np.vstack((p6, p5))

         if cont2 <= 4:
          print('Change gesture for ' + target[cont2+1])
         else:
          p6=pd.DataFrame(p6)
          p6.to_csv('Dataset_model.csv',sep=";", index=False, decimal=",")
          break
       print(str(cont1))
       #print(str(cont2))
       start = time.time()
       tt=[]
      # print(str(result))
      else:
       start = time.time()
       tt = []
      #v1=[]