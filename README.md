# Digit classifier
* Implemented CNN with 4 kernels + 2 max pooling layers using PyTorch to classify 10 MNIST digits 
* Augmented dataset with Gaussian noise, random affine transformations (restricted to rotation, scaling)
* Achieved test accuracy of 0.9958 (top 5% on Kaggle leaderboard ranking)
* Validation confusion matrix:  

![image](https://github.com/user-attachments/assets/7527c69b-9d9a-432a-bc3f-1fef17015699)
* Validation error samples (ambiguous digits are misclassified):

![image](https://github.com/user-attachments/assets/f50a5ad0-6d9d-4fdf-a9e5-0e5634b91b79)
* Top-k retrieval shows ambiguity in training data (validation sample followed by 5 closest training samples):

<img width="757" height="135" alt="image" src="https://github.com/user-attachments/assets/2428f685-ba52-4fe4-9773-67f08b3a6a46" />
