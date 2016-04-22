
import os
from PIL import Image
from resizeimage import resizeimage
# from scipy.misc import imread, imresize, imshow

FILE = "changed_test_image" 



X=32
Y=32 

import numpy as np
import pickle, random
from scipy.misc import imread, imresize, imshow
from scipy.stats import logistic
import os,sys   
import PIL
# import Image 
import pickle, random



# following should add upto 1 
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

LAMBDA = 0.01
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 16

LOSS = []
X = 32
Y = 32




def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.01, reg_lambda=0.01):
        """
        Initialize the network with input, output sizes, weights, biases, learning_rate and regularization parameters
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :param reg_lambda: regularization rate lambda
        :return: None
        """
        # to measure accuracy while training
        self.correct = 0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01 # Weight matrix for input to hidden
        self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01 # Weight matrix for hidden to output
        
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.output_dim, 1)) # output bias
        
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # Add code to calculate h_a and probs        
        xs = X
        hs = np.tanh(np.dot(self.Wxh, xs) + self.bh) # hidden state
        ys = np.dot(self.Why, hs) + self.by 
        ps = np.exp(ys) / np.sum(np.exp(ys))
        
        return hs, ps

    def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
        """
        Add regularization terms to the weights
        :param dWhy: weight derivative from hidden to output
        :param dWxh: weight derivative from input to hidden
        :param Why: weights from hidden to output
        :param Wxh: weights from input to hidden
        :return: dWhy, dWxh
        """
        # Add code to calculate the regularized weight derivatives
        dWhy += self.reg_lambda * Why
        dWxh += self.reg_lambda * Wxh

        return dWhy, dWxh

    def _update_parameter(self, dWxh, dbh, dWhy, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        # Add code to update all the weights and biases here
        
        self.Wxh += -self.learning_rate * dWxh
        self.bh += -self.learning_rate * dbh
        self.Why += -self.learning_rate * dWhy
        self.by += -self.learning_rate * dby


    def _back_propagation(self, X, t, hs, ps):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        if np.argmax(ps) == t:
            self.correct += 1
         
        dWxh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
     
        dy = np.copy(ps)
        dy[t] -= 1 # backprop into y
 
        dWhy += np.dot(dy, hs.T)
        dby = dby + dy
        dh = np.dot(self.Why.T, dy)  # backprop into h
        dhraw = (1 - hs*hs ) * dh # backprop through tanh nonlinearity
        dbh = dbh + dhraw
        dWxh += np.dot(dhraw, X.T)
              
        return dWxh, dWhy, dbh, dby

    def _calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
        """
        Calculate the smoothened loss over the set of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: smooth loss
        """
        if regularizer_type == 'L2':
            # Add regulatization term to loss
            # loss += lamda * (Wxh^2 + Why^2)
            loss+= self.reg_lambda * (np.sum(self.Wxh ** 2) + np.sum(self.Why ** 2) )
            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    def train(self, inputs, targets, validation_data, num_epochs, regularizer_type=None):
        """
        Trains the network by performing forward pass followed by backpropagation
        :param inputs: list of training inputs
        :param targets: list of corresponding training targets
        :param validation_data: tuple of (X,y) where X and y are inputs and targets respectively
        :param num_epochs: number of epochs for training the model
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: None
        """
        print "training"

        for k in xrange(num_epochs):
            loss = 0
            self.correct = 0
            for i in xrange(len(inputs)):
                # Forward pass
                hs,ps = self._feed_forward(inputs[i])
                loss += -np.log(ps[targets[i]]) # softmax (cross-entropy loss)
                # LOSS.append(loss)

                # Backpropogation
                dWxh, dWhy, dbh, dby = self._back_propagation(inputs[i],targets[i],hs,ps)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh, dbh, dWhy, dby)

            # print("Accuracy [while training]", self.correct*100/len(inputs))
          
            # validation using the validation data
            validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            # print 'Validation'
            for i in xrange(len(validation_inputs)):
          
                # Forward pass
                hs,ps = self._feed_forward(validation_inputs[i])
                loss += -np.log(ps[validation_targets[i]]) # softmax (cross-entropy loss)
                # LOSS.append(loss)
                # Backpropogation
                dWxh, dWhy, dbh, dby = self._back_propagation(validation_inputs[i],validation_targets[i],hs,ps)
                
                if regularizer_type == 'L2':
                    dWhy, dWxh = self._regularize_weights( dWhy, dWxh, self.Why, self.Wxh)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh, dbh, dWhy, dby)

            smoothLoss = self._calc_smooth_loss(loss, len(inputs)+len(validation_inputs), regularizer_type)
            LOSS.append(smoothLoss)
            if k%5 == 0:    
                print "Epoch " + str(k) + " : Loss = " + str(smoothLoss)


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        # Implement the forward pass and return the output class (argmax of the softmax outputs)
        hs,ps = self._feed_forward(X)
        return np.argmax(ps)



    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

# arr = ['E']
import cv2,time
from PIL import Image
 
# Camera 0 is the integrated web cam on my netbook
camera_port = 0
TIMES = 5
 
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
 
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)
counter = 1 
# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im
 
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary

# Take the actual image we want to keep

for j in xrange(ramp_frames):
        # print rampls_frames - j,
        temp = get_image()


for i in range(TIMES):
    
    print("Taking image..." + str(counter))
    camera_capture = get_image()
    # print camera_capture[100:400,800:1200,:].shape
    time.sleep(1)
    file = "test_image"+str(counter)+".png"
    cv2.imwrite(file, camera_capture)
    im =Image.open(file)
    im.transpose(Image.FLIP_LEFT_RIGHT).crop((850,50,1200,450)).save("changed_"+file)

    # im.

    counter+=1
del(camera)


# arr = ['A','B','C','D','E']

# arr = []

nn = load('saveFile_NEWE.pkl')

file_name = FILE +str(1)+".png" 
    # for ch in arr:
    # img = Image.open(file_name)
    #changing the dimension of the image to 16*8
    # resize and cover as necessarry 
    # img = resizeimage.resize_cover(img, [X, Y])
    # img = imresize(img,(X,Y,))
im_back= cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # (thresh, im_bw

for i in range(2,6):
    file_name = FILE +str(i)+".png" 
    # for ch in arr:
    # img = Image.open(file_name)
    #changing the dimension of the image to 16*8
    # resize and cover as necessarry 
    # img = resizeimage.resize_cover(img, [X, Y])
    # img = imresize(img,(X,Y,))
    im_gray = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    res =  cv2.bitwise_and(im_gray,im_gray,mask=im_back)

    im_gray = cv2.resize(res,(X,Y))
                # img = resizeimage.resize_cover(img, [X, Y])
                # img = imresize(img,(X,Y,))
    # cv2.imwrite(datapath+"/"+file_name, im_gray)

    # (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite('pic1.png', im_bw)
    # (thresh, im_gray) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #changing the dimension of the image to 16*8
    # resize and cover as necessarry 
    # im_gray = cv2.resize(im_gray,(X,Y))
    # img = resizeimage.resize_cover(img, [X, Y])
    # img = imresize(img,(X,Y,))
    cv2.imwrite("small"+file_name, im_gray)

    # img.save("small"+file_name, img.format)
    # img.close()
    img = imread("small"+file_name)             
    #     # resize to 16x8x3
    #     img = imresize(img,(16,8,))
    linear_img = []
    linear_img = np.reshape(img, (X*Y*1, 1))
    #     # normalize 
    linear_img =np.true_divide(linear_img, 255.0)
    #     # add entry to inputs and target
    # inputs.append(linear_img)
    print int(nn.predict(linear_img)),chr(int(nn.predict(linear_img))+65),
        
        # os.system("echo '"+chr(int(nn.predict(linear_img))+65)+"\' | say ")
    print ""

# print datapath+"/"+file_name
           
