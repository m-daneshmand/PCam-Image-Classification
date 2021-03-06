# PCam-Image-Classification

PCam Image Classification
This project aims at comparing several techniques for detection of cancerous tissue patches. To reach this objective, two dataset types consisting of CIFAR-10 and PCam are used on VGG like network.  The primary step of the project is transfer learning by using a deep convolutional neural network used for CIFAR-10 classification to train the model on PCam dataset. Additionally, triplet loss is another approach to evaluate both scratch and pre-trained models.  After comparison between them, k-nearest neighbors (KNN) is fitted on train data and evaluate the performance of all the networks on PCam test set.

## Code structure
This project consists of five folders related to five different approaches mentioned above (*CIFAR-10 - 70 – KNN*, *PCam – Pretrained*, *PCam - Pretrained - TripletLoss – KNN*, *PCam - Scratch – KNN*, *PCam - Scratch - TripletLoss – KNN*). All of them have the same structure in which three folders and three python files are located to run. They are listed as below: 

* **Run-Train.py** run this file to start training the model.

* **Run-Test.py** run this file to start testing the model.

* **Run-KNN.py** run this file to start fitting KNN on train data. (The number of neighbors can be easily set to related argument in this file)

* **data/** Downloaded dataset should be placed in this directory.

* **model/** After training the model, pre-trained weights will be stored in this directory. 

* **log/** After testing the model, results will be generated in this directory. 

