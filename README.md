#CaptchaRecognation
Learn to use opencv and tensorflow to process and predict the number.
This is just a simple version that is only suitable on condition of four-number captcha.
## Basic info
I use cv2 to separate the four numbers and preprocess the image since mnist dataset is used to train the model.
Tensorflow is used to build a three-layer CNN to recognize the number. 
The CNN architecture is like conv-conv-pool-conv-pool-fc-softmax, which is really simple. You can modify it to whatever you like.

##The requirements:
funcsigs==1.0.2
mock==2.0.0
numpy==1.11.2
pbr==1.10.0
protobuf==3.2.0
six==1.10.0
TBB==0.1
tensorflow==0.12.1
opencv==2.4.11

##Training
run python captcha_recognition.py train
##Test
run python captcha_recognition.py test \<path to captcha.jpeg><br>

##Result
Since I have only thained the model for 2000*50/60000=1.66 epochs, in the current mnist_model.ckpt, test accuracy is
around 97.9% and training accuracy is around 98.4%.

Although this is not a powerful model in practice, but a good base to modify. 
Feel free to do whatever you like with this commit.
 

