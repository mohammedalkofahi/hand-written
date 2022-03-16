import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import  Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split




# Dataset containing handwritten charachters 

X_train = pd.read_csv(r"csvTrainImages 13440x1024.csv").astype('float32')


y_train = pd.read_csv(r"csvTrainLabel 13440x1.csv").astype('float32')


X_test = pd.read_csv(r"csvTestImages 3360x1024.csv").astype('float32')


y_test = pd.read_csv(r"csvTestLabel 3360x1.csv").astype('float32')

# Reshaping the data to display it as an image


X_train = np.reshape(X_train.values, (X_train.shape[0], 32,32))
X_test = np.reshape(X_test.values, (X_test.shape[0], 32,32))


print("Training data shape: ", X_train.shape)
print("Testing  data shape: ", X_test.shape)


ar_char = {0:'أ',1:'ب',2:'ت',3:'ث',4:'ج',5:'ح',6:'خ',7:'د',8:'ذ',9:'ر',10:'ز',11:'س',12:'ش',13:'ص',14:'ض',15:'ط',16:'ظ',17:'ع',18:'غ',19:'ف',20:'ق',21:'ك',22:'ل',23:'م', 24:'ن',25:'ه',26:'و',27:'ي'}


# Plotting the number of alphabets in the dataset

y_axis= np.int0(y_train)


count = np.zeros(28, dtype='int')
for i in y_axis:
    count[i] +=1


alphabets = []
for i in ar_char.values():
    alphabets.append(i)

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.barh(alphabets, count)

plt.xlabel("Amount ")
plt.ylabel("Alphabets")
plt.grid()
plt.show()





#Reshaping the dataset to put in the model

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
print("New training data shape: ", X_train.shape)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
print("New Testing data shape: ", X_test.shape)


#Converting the labels to categorical values

ytrain = keras.utils.np_utils.to_categorical(y_train, num_classes = 28, dtype='int')
print("New training labels shape: ", ytrain.shape)

ytest = keras.utils.np_utils.to_categorical(y_test, num_classes = 28, dtype='int')
print("New Testing labels shape: ", ytest.shape)



model = Sequential()     #stack of layers in the NN

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(128,activation ="relu"))
model.add(Dense(64,activation ="relu"))

model.add(Dense(28,activation ="softmax"))

model.summary()

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


model_train = model.fit(X_train, ytrain, epochs=10, callbacks=[reduce_lr, early_stop],  validation_data = (X_test,ytest))



model.save('armodel.h5')




print("Validation accuracy :", model_train.history['val_accuracy'])
print("Training accuracy :", model_train.history['accuracy'])
print("Validation loss :", model_train.history['val_loss'])
print("Training loss :", model_train.history['loss'])








fig, axs = plt.subplots(2,2, figsize=(8,9))
axs = axs.flatten()

for i,ax in enumerate(axs):
    pic = np.reshape(X_test[i], (32,32))
    ax.imshow(pic, cmap="Greys")
    prediction = ar_char[np.argmax(ytest[i])]
    ax.set_title("Prediction: "+prediction)
    ax.grid()



img = cv2.imread(r'ar2.png')
plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


resized = cv2.resize(gray, (32,32))

newimg =np.reshape(resized, (1,32,32,1))


newimg = ar_char[np.argmax(model.predict(newimg))]


print("prediction: " + newimg)

