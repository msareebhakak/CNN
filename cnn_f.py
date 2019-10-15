import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from google.colab import drive
import pandas as pd


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt 
image_index = 7777 #checking the image
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')

x_train[image_index].shape

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


learn_rate = 0.0005
batch = 180
drop_out = 0.440

model=Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28,28,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), input_shape=(14,14,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256))
model.add(Dropout(drop_out))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Dropout(drop_out))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train,y_train, batch_size=batch, epochs=1, validation_split=0.4)
print(model.evaluate(x_test, y_test))
pred = model.predict(x_test)

#np.savetxt('mnist.csv',pred)
#for google drive mounting as network trained with collab


drive.mount('/content/gdrive',force_remount=True)
root_dir="/content/gdrive/My Drive"
base_dir=root_dir + 'CNN/'
path = '/content/gdrive/My Drive/CNN/'

for i,d in enumerate(pred):
  pred[i] = pred[i] // max(d)

pred = np.array(pred, dtype=np.uint8)
save = pd.DataFrame(pred, columns=[0,1,2,3,4,5,6,7,8,9])
save.to_csv(path+'MNIST.csv',index=False, header=False)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


###Hyperparameter random search

def blackbox(lear_rate, drop, batch,x_train,y_train,x_test,y_test):
  
  
  model=Sequential()
  model.add(Conv2D(64, (5,5), input_shape=(28,28,1)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(128, (3,3), input_shape=(14,14,1)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())

  model.add(Dense(256))
  model.add(Dropout(drop))
  model.add(Activation('relu'))

  model.add(Dense(128))
  model.add(Dropout(drop))
  model.add(Activation('relu'))

  model.add(Dense(10))
  model.add(Activation('softmax'))
  
  adam = optimizers.Adam(lr=lear_rate, beta_1=0.9, beta_2=0.999)
  model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
  history = model.fit(x_train,y_train, batch_size=batch, epochs=10, validation_split=0.4)
  #print(model.evaluate(x_test, y_test))
  return history



ncom=4
np.random.seed(14)
lr=np.random.uniform(0.00001,0.01,ncom)

batch = np.random.randint(32,512,ncom)
drop =np.random.uniform(0.2,0.8,ncom)

lr=np.sort(lr)
batch=np.sort(batch)
drop=np.sort(drop)

print(drop)
print(lr)
print(batch)

hist =[]

for i in range(ncom):
  hist.append(blackbox(lr[i],drop[i],batch[i],x_train,y_train,x_test,y_test))
  print(hist[i].history.keys())
  # summarize history for accuracy
  plt.plot(hist[i].history['acc'])
  plt.plot(hist[i].history['val_acc'])
  plt.title('model accuracy'+str(i))
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(hist[i].history['loss'])
  plt.plot(hist[i].history['val_loss'])
  plt.title('model loss'+str(i))
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

val_acc=[]
for i in range(ncom):
  print(hist[i].history.values())
  val_acc.append(hist[i].history['val_acc'][-1])
  
  
print(val_acc)

plt.plot(batch,val_acc)
plt.title('batch vs. accuracy')
plt.ylabel('acc')
plt.xlabel('batch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(drop,val_acc)
plt.title('dropout vs. accuracy')
plt.ylabel('acc')
plt.xlabel('dropout')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(lr,val_acc)
plt.title('learning rate vs. accuracy')
plt.ylabel('acc')
plt.xlabel('learning rate')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()
