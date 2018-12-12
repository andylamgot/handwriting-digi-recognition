'''
This dataset is made up of 1797 8x8 images. 
Each image is of a hand-written digit. 

In order to utilize an 8x8 figure, 
we have to first transform it 
into a feature vector with length 64.
'''

# Import dataset
from sklearn.datasets import load_digits

# Import the sklearn for SVM
from sklearn import svm

digits = load_digits()

# Each datapoint is a 8x8 image of digit
# Plot the image 
plt.gray()
plt.matshow(digits.images[20])
plt.show

print digits.images[20]
'''
Output:
[[  0.   0.   3.  13.  11.   7.   0.   0.]
 [  0.   0.  11.  16.  16.  16.   2.   0.]
 [  0.   4.  16.   9.   1.  14.   2.   0.]
 [  0.   4.  16.   0.   0.  16.   2.   0.]
 [  0.   0.  16.   1.   0.  12.   8.   0.]
 [  0.   0.  15.   9.   0.  13.   6.   0.]
 [  0.   0.   9.  14.   9.  14.   1.   0.]
 [  0.   0.   2.  12.  13.   4.   0.   0.]]
'''
clf = svm.SVC()

# Train the model 
clf.fit(digits.data[:-1], digits.target[:-1])  

# Test the model
prediction = clf.predict(digits.data[20:21])

print "Predicted Digit ->",prediction
'''
Output:
Predicted Digit -> [0]
'''
