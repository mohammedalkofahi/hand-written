
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , Activation , Flatten , Conv2D , MaxPooling2D


mnist = tf.keras.datasets.mnist


(X_train, y_train), (X_test, y_test)=mnist.load_data()


X_train.shape


plt.imshow(X_train[0])
plt.show()
plt.imshow(X_train[0] , cmap= plt.cm.binary)






X_train = tf.keras.utils.normalize(X_train , axis=1)
X_test  = tf.keras.utils.normalize(X_test  , axis=1)
plt.imshow(X_train[0] , cmap= plt.cm.binary)





Img_Dim= 28
X_trainr= np.array(X_train).reshape(-1 , Img_Dim, Img_Dim,1)
X_testr = np.array(X_test).reshape(-1 , Img_Dim , Img_Dim,1)
print("Training samples dimension" , X_trainr.shape)
print("Training samples dimenson"  , X_testr.shape)




model= Sequential()




model.add(Conv2D(64, (3,3), input_shape= X_trainr.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))




model.summary()




print( "Total training samples= " , len(X_trainr))




model.compile(loss= "sparse_categorical_crossentropy" , optimizer="adam" , metrics=["accuracy"])




model.fit(X_trainr,y_train,epochs=5 , validation_split= 0.3) #training my model

model.save('mnist.h5')

# evaluating accuracy by testing dataset mnist
test_loss , test_acc = model.evaluate(X_testr, y_test)
print("Test loss :" , test_loss)
print("Validation accuracy :", test_acc)




predictions= model.predict([X_testr])
print(predictions)




print(np.argmax(predictions[0]))




plt.imshow(X_test[0])




img= cv2.imread("five.png")
plt.imshow(img)

#img.shape

gray= cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#gray.shape


resized = cv2.resize(gray,(28,28), interpolation= cv2.INTER_AREA)
#resized.shape


newimg = tf.keras.utils.normalize(resized, axis=1) 
newimg= np.array(newimg).reshape(-1, Img_Dim, Img_Dim,1) # kernal operation of convlution layenr
#newimg.shape



predictions= model.predict(newimg)
print(np.argmax(predictions))


    
font_scale= 1.5
font = cv2.FONT_HERSHEY_PLAIN

cap= cv2.VideoCapture("handwritten_digits.mp4")

if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
    
text= ""
#get the width and height of textbox

(text_width, text_height)= cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
#set the text start position
text_offset_x=10
text_offset_y=img.shape[0]- 25

box_coord=((text_offset_x, text_offset_y),(text_offset_x + text_width + 2 , text_offset_y - text_height - 2 ))

ctr=0;

while True:
    ret, frame = cap.read()
    ctr=ctr + 1;
    if((ctr%2)==0):
        gray= cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray,(28,28), interpolation= cv2.INTER_AREA)
        newimg = tf.keras.utils.normalize(resized, axis=1)
        newimg= np.array(newimg).reshape(-1, Img_Dim, Img_Dim,1)
        predictions= model.predict(newimg)
        status= np.argmax(predictions)
        print(status)
        print(type(status))
        
        x1,y1,w1,h1= 0,0,175,75
        cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)
        cv2.putText(frame, status.astype(str), (x1 + int(w1/5),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
        
        cv2.imshow("Handwritten Digits Recognition", frame)
        if cv2.waitKey(2) & 0xFF ==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


