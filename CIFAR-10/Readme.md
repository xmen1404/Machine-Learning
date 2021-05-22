# Train Neural Network for CIFAR-10 dataset
I developed a neural network model for CIFAR-10 dataset. This is an assignment from a course at my university.

# Usage
The source code is a .ipynb file which can be run on Google Colab and other websites that support this notebook document (e.g. Kaggle).
The weights of final model can be found [here](https://drive.google.com/file/d/1iXHMOqtOQWoHsn4XZUpo5coiPxRmXJl1/view?usp=sharing).

# Phases of development

Phase 1:

    - batch_size = 50
    - n_epoch = 60
    - learning_rate = 0.1
    - Optimzer = SGD
    - 3 convolution layer with kernel size = 3, maxpooling = 2x2 and stride = 2
    - 3 Fully connected layer
    - Validation accuracy increase and stop at 77%, training accuracy reach 100% at the end
    - Overfitting occured
    => We need to decreases the capacity of the model to prevent overfitting
    
Phase 2: 

    - 3 convolution layer with kernel size = 3, maxpooling = 2x2 and stride = 2: Image 3*32*32 -> 360*3*3
    - 2 Fully connected layer: 90 * 4 * 3 -> 1024 -> 10
    - Training accuracy reached over 90%, validation loss stopped at 71%
    - Overfitting occured again, the decrease of capacity did not work.
    => In next phase, we will try to increase the capacity and use dropout to prevent overfitting
    
Phase 3: 

    - 3 convolution layer with kernel size = 3, maxpooling = 2x2 and stride = 2: Image 3*32*32 -> 1920*3*3
    - 2 Fully connected layer: 1920 * 4 * 3 -> 10 (input - output)
    - Dropout with p=0.3 before each fully connected layer
    - Trainning accuracy eached nearly 100%, validation stopped at 79%
    - Result is improved but overfitting still occured 
    => In next phase, we will try increase dropout rate to 0.5, allow dropout between 2 convolution layer, and transform images to grayscale as well as normalizing them to have similar value range and see if they could reduce the overfitting effects.
        Also, the Adam optimizer will be deployed with learning rate 0.0003, this optimizer utilize adaptive learning rate for each parameters which is usually faster for convergence than SGD in most case, maybe also overfitting faster (higher learning rate for Adam prevent the model from learning as I observed from some trials).

Phase 4: 

    - 4 convolution layers: [Conv2d -> relu -> Conv2d -> relu -> maxpooling] * 2 
    - 3 fully connected layers: 270 * 2 * 6 * 6 -> 1024 -> 256 -> 10
    - Dropout with p=0.5 before each convolution layers and before each fully connected layers
    - Image transformation = Grayscale + Normalize(0.5, 0.5, 0.5)
    - The model converge to >70% faster (under 10 epochs from scratch).
    - I stopped the model at trainning accuracy = 94%, validation accuracy had stopped at 78% for a while and the validation loss started decreasing.
    - The overfitting kept occurring. The model converge faster.
    - I realized that this model with grayscale + Normalization transformation does not work well with image without same transformation. The image need to be transformed similarly before push into model for prediction. This is might due to the difference in range of value, as the trainning images have been normalized.
    => Next phase, we will try to use batch normalization, which normalize the range of value in each batch and may help increase the generalization of the model.
    
Phase 5: 

    - 4 convolution layers: [Conv2d -> batch_normalize -> relu -> maxpooling] * 4
    - 2 fully connected layers: 810*1*1 -> 256 -> 10
    - Dropout with p=0.5 only before each fully connected layers, I have read that dropout in convolution layers usually bring wierd result, and it has different effect as dropout in fully connected layers
    - Image transformation = Normalize(0.5, 0.5, 0.5)
    - The validation loss started decreases from epochs 20, trainning accuracy > 90% and validation accuracy kept at 77%, No noticable improvement.
    - Overfitting has not been solved.
    => We will try l2-regularization for the next phase with weight-decay=0.005 to see if it could effectively penalize the over-complexity of model and reduce the overfitting effects.

Phase 6: 

    - 3 convolution layers: [Conv2d -> batch_normalize -> relu -> maxpooling] * 3
    - 2 fully connected layers: 270*4*3*3 -> 512 -> 10
    - Dropout with p=0.5 only before each fully connected layers
    - Weight-decay factor = 0.005 for Adam Optimizer.
    - Image transformation = Normalize(0.5, 0.5, 0.5)
    - I stopped at epochs 103, as validation loss had increased for a while. Trainning accuacy = 88%, validation accuracy = 82%. 
    - Result is improved, and overfitting occur later than last phase.
    => We will try Horizontal flip and RandomCrop for training transformation to create more random image, help to generalize the training data more and check if it could reduce the overfitting effects. Also, the batch-normalization will be placed after each conv2d layer and the weight-decay will be decrease to 0.0008 (As I did try some trials beforehand with same configurations and higher weight-decay but the model could not learn after reaching traing accuracy = ~77%)

Phase 7: (Final)

    - 3 convolution layers: [Conv2d -> batch_normalize -> relu -> maxpooling] * 3
    - 2 fully connected layers: 270*4*3*3 -> 512 -> 10
    - Weight-decay factor = 0.0008 for Adam Optimizer
    - Dropout with p=0.5 only before each fully connected layers
    - Image transformation = Normalize(0.5, 0.5, 0.5), RandomHorizontalFlip, RandomCrop(32,padding=4).
    - The validation loss started increasing from ~epoch 60, the training accuracy ~85%, validation accuracy ~85%
    - The result is improved. The overfitting happened later than last phase.
    The result is finalized here!
