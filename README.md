# U-NET
- Used Pydicom, OpenCV for DICOM file conversion processing; Utilized MATLAB to preprocess the dataset, applying image enhancement techniques and random rotation to improve model generalization
- Constructed a U-NET neural network using PyTorch, fine-tuning hyperparameters, precisely the number of U-NET layers and channels for the task of lung CT images
- Improved the accuracy of conventional image segmentation to 91% segmentation accuracy, contributed to the foundation of subsequent lung image classification tasks through precise lung image segmentation
## ENVIRONMENT
pycharm+python3.6+pytorch1.3.1  
## HOW TO RUN:
Enter the dataset.py and correct the path of the datasets
example:
```
python main.py --action train&test --arch UNet --epoch 21 --batch_size 21 
```
## U-NET Model
![Unet_model](https://github.com/littlesheep12/U-Net/blob/main/Unet_model.png "Unet_model")
## RESULTS
after train and test,3 folders will be created,they are "result","saved_model","saved_predict".
### saved_model folder:
After training,the saved model is in this folder.
### result folder:
![result](https://github.com/littlesheep12/U-Net/blob/main/lung.png "result")
![result_1](https://github.com/littlesheep12/U-Net/blob/main/result_1.png "result_1")