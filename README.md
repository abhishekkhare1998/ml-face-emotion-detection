# ml-face-emotion-detection

In this project I have implemented a binary classifier to classify the images into two "Facial Emotions".
and in the dataset, I have two sets of images - Happy face and Neutral Face.

Implementation:

1) Use the training images to formulate the "FaceSpace" of the two sets of images, So our aim is to create two subspaces - happy-face subspace and neutral-face subspace which spans the principle components of the respective image types. For doing this, we need to do the eigenvalue decomposition of our training images and compute the principle components having the maximum variance

2)Using the principle components formulate a subspace for respective image types. Compute the MSE (Mean Square Error) of images and display it. Also Analyse the variance of MSE as we change the number of principle components for construction of the subspace.

3) Test: 

    i) Take a test image - t, compute its orthogonal projection on both subspaces.
    ii) Let orthogonal projection of t on subspaces be t_happy and t_neutral respectively.
    iii) Compute L2 norm of (t-t_happy) and (t-t_neutral) and compare.
    iv) Classify the test based on the norm i.e, if ||(t-t_happy)|| < ||(t-t_neutral)|| then classify the image as happy.
    v) repeat steps i) to iv) for the rest of the images
