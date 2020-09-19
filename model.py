i=2
import sys
import os
import argparse
#parser= argparse.ArgumentParser()
#parser.add_argument('i')
#args= parser.parse_args()
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
print("import successfully")
model=Sequential()
#i=int(sys.argv[1])
pool=(2,2)
model.add(Convolution2D(filters=32,
                            kernel_size=(3,3),
                            input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
    
    
for j in range(1,i+1):
        pool=(2,2)
        if i > 3:
            pool=(1,1)
        model.add(Convolution2D(filters=32,
                  kernel_size=(3,3),
                  activation='relu'))
        model.add(MaxPooling2D(pool_size=pool))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
    
for j in range(4,i):
           model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('Skin_cancer/Train',
target_size=(224, 224),
batch_size=1,
class_mode='binary')
test_set = test_datagen.flow_from_directory('Skin_cancer/Test',
target_size=(224, 224),
batch_size=1,
class_mode='binary')
    
result= model.fit(training_set,
steps_per_epoch=2000,
epochs=10,
validation_data=test_set,
validation_steps=100)        
model.summary()
x=result.history['accuracy'][0]*100
print("current accuracy=", x)
   

