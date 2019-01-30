'''
Name: Matthew Persad
NetID: mp3685
Project 2
'''

#import the numpy module to use matrices and vectors
import numpy, datetime

#the Neural Network class
#which handles weight training and out
class NeuralNetwork():
    #initialize the Neural Network
    def __init__(self):
        #seeds the numpy random generator
        numpy.random.seed(1)
        
        #the network list will hold all the weights to be used
        self.network = []
        
        #the hidden list will hold the hidden layer outputs
        self.hidden = []
        
        #the errorT list will hold the average error for every sample
        self.errorT = []

    #method to initialize the weights
    #inputs = the size of the input layer
    #hidden = the size of the hidden layer
    #output = the size of the output layer
    def createNetwork(self, inputs, hidden, output):
        #generate a (hidden x inputs) size matrix, with every element as a randomly generated number
        #represents the weights between the hidden and input layer
        #Uses He-et-al initialization
        self.network.append(numpy.random.randn(hidden, inputs) * numpy.sqrt(2/hidden)) 
        
        #generate a (hidden x 1) size matrix, with every element as a randomly generated number between 0 and 1, inclusive
        #represents the bias weights for the hidden layer
        self.network.append(numpy.random.uniform(0.0, 1.0,(hidden, 1)))

        #generate a (output x hidden) size matrix, with every element as a randomly generated number
        #represents the weights between the output and hidden layer
        #Uses He-et-al initialization
        self.network.append(numpy.random.randn(output, hidden) * numpy.sqrt(2/output))
        
        #generate a (output x 1) size matrix, with every element as a randomly generated number between 0 and 1, inclusive
        #represents the bias weights for the output layer
        self.network.append(numpy.random.uniform(0.0, 1.0, (output, 1)))
        
    #method to calculate the sigmoid value of a given number (or every element in a matrix)
    def sigmoid(self, x):
        return 1.0 / (1.0 + numpy.exp(-x))

    #method to calculate the derivative of the sigmoid value of a given number (or every element in a matrix)
    def sigmoid_prime(self, x):
        return x * (1 - x)

    #method to calculate the output values of both the hidden and output layers 
    def calculate(self, array, test=False):
        #a for loop to calculate the hidden and output layer outputs
        for i in range(0, len(self.network), 2):
            
            #the variable "product" is a summation of every input multiplied by its respective weight
            #"product" is a matrix that will be passed into the "sigmoid" method
            product = numpy.dot(self.network[i], array) + (-1 * self.network[i+1])
            
            #the variable "array" represents the output values of a layer (stored as a ((layer size) x 1) matrix)
            array = self.sigmoid(product)
            
            #stores the hidden layer outputs
            #so it can be used during backpropagation
            if i==0:
                self.hidden = array
        
        #the variable "lst" is a list that will be used during testing
        #will be compared to the test labels
        lst=[]
        
        #conditional to determine whether the "calculate" was called during training or testing
        if test:
            
            #a for loop to determine which number is in the image
            for i in range(len(array)):
                
                #conditional to determine the maximum element out of the output layer outputs
                if array[i]==max(array):
                    
                    #the index of this element represents the number that was guessed by the neural network to represent the number in the image file
                    lst.append(1)
                else:
                    
                    #the index of this element represents the number that was not guessed by the neural network to represent the number in the image file
                    lst.append(0)
        
        #returns the output layer outputs and the "lst" variable
        # during training, the "lst" variable is always an empty list
        return [array, lst]

    #method to perform backpropagation on the weights of the neural network
    def error(self, true, array):
        
        #calculate the output layer outputs
        output = self.calculate(array)
        
        #the variable "output" now equals the output layer outputs
        output=output[0]
        
        #calculates the error for each output layer output
        errorArray = (true-output)
        
        #calcalates the average error for the current sample
        errorTotal = 0
        for i in range(len(errorArray)):
            errorTotal += (errorArray[i][0]**2)
        errorTotal *= 0.5
        errorTotal /= 5
        
        #appends the average error to a list so the average error of the epoch can be calculated
        self.errorT.append(errorTotal)

        #************************************************ - open
        #calulate the delta i for each output layer node
        deltaI = errorArray * self.sigmoid_prime(output) #comment this entire line
        
        #comment out the line above and  uncomment the line below for a better accuracy (at least 90-95%)
        #deltaI = errorArray * self.sigmoid_prime(self.sigmoid(output)) #uncomment this entire line
        #************************************************ - close

        #calculates the summation of the output-hidden weights times its respective delta i
        #will be sued to calculate delta j
        summation = numpy.dot(self.network[2].T, deltaI)

        #************************************************ - open
        #calculate the delta j for each hidden layer node
        deltaJ = self.sigmoid_prime(self.hidden) * summation #comment this entire line
        
        #comment out the line above and  uncomment the line below for a better accuracy (at least 90-95%)
        #deltaJ = self.sigmoid_prime(self.sigmoid(self.hidden)) * summation #uncomment this entire line
        #************************************************ - close
        
        #update the weights between the input and hidden layers
        self.network[0] = self.network[0] + (0.05 * array.T * deltaJ)
        
        #updates the bias weights for the hidden layer
        self.network[1] = self.network[1] + (0.05 * -1 * deltaJ)
        
        #update the weights between the hidden and ouput layers
        self.network[2] = self.network[2] + (0.05 * self.hidden.T * deltaI)
        
        #updates the bias weights for the output layer
        self.network[3] = self.network[3] + (0.05 * -1 * deltaI)

        return

#the main function of the program
def main():
    #create the neural network
    network = NeuralNetwork()
    
    #prompts the user to input the size of the hidden layer
    hiddenLayerSize = int(input("Please enter the size of the hidden layer: "))
    
    #initialize the neural network values
    network.createNetwork(784, hiddenLayerSize, 5)
    
    #a list to hold the values for every pixel in the training set
    #so the files only have to be read once
    numbers=[]
    
    #a counter for the number of epochs ran
    epoch=0
    
    #a variable to store the previous epoch's average error
    prev=0
    
    #a variable to store the current epoch's average error
    curr=100
    
    #a while loop to determine when to stop training the neural network:
    #train for at least 150 epochs
    #the pervious epoch's average error must be greater or equal to the current epoch's average error
    #the difference between the previous and current epoch's average errors is miniscule
    #and the current epoch's average is less than or equal to 2%
    while (epoch<150) or (prev < curr) or (abs(prev-curr) > 0.0002) or curr > 0.02:
        #test to determine whether to read from the training files or not
        if epoch==0:
            #open the training images file
            file1 = open("train_images.raw", "rb") #28038 images
            
            #open the training labels file
            file2 = open("train_labels.txt", "r")
            
            #a counter for the number of pixels in each image
            count=0
            
            #a list to hold all the pixels values for an image
            number=[]
            
            #read the first pixel value
            x=file1.read(1)
            
            #a while loop to read the entire training images file
            while x:
                #an if statement to determine whether all 784 pixels of an image have been read
                if count==784:
                    
                    #reset the pixel counter
                    count=0
                    
                    #read the label associated with the current image
                    line = file2.readline()
                    
                    #convert the label's values into a list of integers
                    line = line.split()
                    
                    #turn the list of integers into a numpy array
                    for i in range(len(line)):
                        line[i] = [int(line[i])]
                        
                    #perform training of the netural network
                    network.error(numpy.array(line), numpy.array(number))
                    
                    #append the image pixel list and respective label to the "numbers" so the files do not have to be read more than once
                    numbers.append((line, number))
                    
                    #reset the pixel list
                    number=[]
                    
                #convert the pixel value to an integer
                num=int.from_bytes(x, byteorder='big')
                
                #append the pixel value to the pixel list
                number.append([num])
                
                #increment the pixel counter
                count+=1
                
                #read the next pixel value from the file
                x=file1.read(1)
                
            #close both training files
            file1.close()
            file2.close()
        
        #if the current epoch is not the first epoch, perform training using the "numbers" list
        else:
            #a for loop to loop through the "numbers" to train the neural network
            for i in numbers:
                #train the neural network
                network.error(numpy.array(i[0]), numpy.array(i[1]))
        #store the previous epoch's average error
        if epoch!=0:
            prev=curr
            
        #calcalate the current epoch's average error
        curr=sum(network.errorT)/28038
        #reset the neural network's list to store every sample's average error
        network.errorT = []
        
        #increment the epoch counter
        epoch+=1
    
    #print the epochs that the neural network trained for
    print("EPOCH:", epoch)
    
    #open the files for testing the neural network
    file3 = open("test_images.raw", "rb") # 2561 images
    file4 = open("test_labels.txt", "r")
    
    #read the first pixel value from the test images file
    y=file3.read(1)
    
    #a list to hold onto the pixel values of an image
    test=[]
    
    #a variable to count the number of ixels per image
    count=0
    
    # variable to count the number of images in the file
    counter=0
    
    #a variable to count the number of images that the neural network guessed correctly
    correct = 0
    
    #creates the 5x5 confusion matrix as a numpy matrix
    list0=[0,0,0,0,0]
    list1=[0,0,0,0,0]
    list2=[0,0,0,0,0]
    list3=[0,0,0,0,0]
    list4=[0,0,0,0,0]
    confusion=numpy.array([list0,list1,list2,list3,list4])
    
    #a while loop to read every pixel value in the test images file
    while y:
        #an if statement to determine whether all 784 pixels of an image have been read
        if count==784:
            #reset the pixel counter
            count=0
            
            #use the neural network to guess which number is in the image
            array = network.calculate(numpy.array(test), True)
            
            #read the label associated with the current image
            line = file4.readline()
            
            #split the label line into a list of 5 numbers (as characters)
            line = line.split()
            
            #a for lopp to convert the list of characters into a list of integers
            for i in range(len(line)):
                line[i] = int(line[i])
                
            #a comparison to see the neural network correctly guessed the number in the image
            if array[1]==line:
                correct+=1
            
            #imcrement the appropriate element in the confusion matrix
            confusion[line.index(1)][array[1].index(1)]+=1
            
            #clear the test list
            test.clear()
            
            #increment the image counter
            counter+=1
        
        #convert the pixel value to an integer
        num=int.from_bytes(y, byteorder='big')
        
        #append the pixel value to the pixel value list
        test.append([num])
        
        #increment the pixel counter
        count+=1
        
        #read the next pixel value in the file
        y=file3.read(1)
        
    #print the accuracy of the neural network
    print("Accuracy:", correct/counter * 100)
    
    #print the confusion matrix
    print(confusion)
    
    #close the testing files
    file3.close()
    file4.close()

#call the main function to run the program
main()