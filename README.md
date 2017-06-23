# demo-python

There are 3 python demos in this folder, they all use the openBCIDataStream.js to get data from the board (you dont need to run it)
The first one is printData.py
  you run it with the command >python printData.py <freqRange>
  Choose the <freqRange> between : 'delta', 'theta', 'alpha', 'beta', 'gamma'

  delta : [0.4Hz, 4Hz]
  theta : [4Hz, 7Hz]
  alpha : [8Hz, 12Hz]
  beta : [12Hz, 25Hz]
  gamma : [25Hz, 35Hz]

This script will print in the terminal the mean valuer per second for each of the 4 channels of the frequency range you determined

_______________________________________________________________________________

The script sinToneDemo.py, is launched without argument >python sinToneDemo.py
and plays a sine tone whose frequency depends of the alpha amplitude, the freq increases from 500 to 1500Hz when alpha amplitude increases.  
TODO : smoothen the transition, for now it plays the tone every sec, with silence between two tones

_______________________________________________________________________________

The last python script soundDemo.py is also run without argument >python soundDemo.py
It will play a song from After Marianne band : https://www.facebook.com/AfterMarianneFR/
and will adjust the volume accordingly to the alpha amplitude. Volume is modified between 0 and 1,
1 corresponds to high alpha amplitude
0 to low alpha amplitude
