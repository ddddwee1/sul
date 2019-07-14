Convert weights from original [LSTMPM repo](https://github.com/lawy623/LSTM_Pose_Machines)

1. conda install caffe-cpu -c willyd
2. conda install tensorflow=1.8
3. mmtor -f caffe -n LSTM_deploy1.prototxt -w LSTM_PENN.caffemodel -o lstmpm_d1
4. mmtor -f caffe -n LSTM_deploy2.prototxt -w LSTM_PENN.caffemodel -o lstmpm_d2
5. python network_convert.py 
6. (optional) You can test whether the output is consistent by running 'python caffetest_deploy\*.py'
7. The deploy network is in 'network.py'

