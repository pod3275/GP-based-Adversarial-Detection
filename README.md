# GP-based-Adversarial-Detection

Official Keras implementation of paper:

[Adversarial example detection with Gaussian Process Regression-based Detection](http://www.itiis.org/digital-library/manuscript/2475) (TIIS).

## Description

  ![image](https://user-images.githubusercontent.com/26705935/47216775-0b91b180-d3e1-11e8-90d3-d015f70e02e0.png)

- Adversarial example detection with Gaussian Process Regression-based detector.

- Existing deep learning-based adversarial detection methods require numerous adversarial images for their training. 

- The proposed method overcomes this problem by performing classification based on the statistical features of adversarial images and clean images that are extracted by Gaussian process regression with a small number of images. 
  
## Requirements

- [Python 3.6](https://www.python.org/downloads/)
- [Tensorflow == 1.12.0](https://github.com/tensorflow/tensorflow)
- [Keras >= 2.2.4](https://github.com/keras-team/keras)
- [cleverhans >= 3.0.1](https://github.com/tensorflow/cleverhans)
- [GPy >= 1.9.6](https://github.com/SheffieldML/GPy)
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## How to run
**1. Git clone**

`$ git clone https://github.com/pod3275/GP-based-Adversarial-Detection.git`

**2. Training target model**

`$ python generate_model.py --dataset MNIST`

- Available dataset : [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

**3. Generate adversarial examples**

`$ python attack.py --dataset MNIST --attack FGSM`

- Available attack method : [FGSM](https://arxiv.org/pdf/1412.6572.pdf), [BIM](https://arxiv.org/pdf/1607.02533.pdf), [JSMA](https://arxiv.org/pdf/1511.07528.pdf), [DeepFool](https://arxiv.org/pdf/1511.04599.pdf), [CW](https://arxiv.org/pdf/1608.04644.pdf) 

**4. Detect with GP-based detector**

`$ python attack.py --dataset MNIST --attack FGSM --num_data_in_class 30`

- num_data_in_class : number of adversarial example in one class for training detector

## Results
- Attack accuracy

  ![image](https://user-images.githubusercontent.com/26705935/64116027-20deb700-cdcc-11e9-9187-f6528a5c990f.png)
  
- Detection accuracy

  ![image](https://user-images.githubusercontent.com/26705935/64116053-33f18700-cdcc-11e9-93a2-97296ab2232f.png)
  
- Number of adversarial examples for detector training

  ![image](https://user-images.githubusercontent.com/26705935/64116090-4a97de00-cdcc-11e9-8725-0f373dfa0e17.png)
  
  - Better performance with extremely small number of adversarial example.
  
## Citation
```
@inproceedings{BSSdistill,
	title = {Adversarial Detection with Gaussian Process Regression-based Detector},
	author = {Sangheon Lee, Noo-ri Kim, Youngwha Cho, Jae-Young Choi, Suntae Kim, Jeong-Ah Kim, Jee-Hyong Lee},
	booktitle = {KSII Transactions on Internet and Information Systems (TIIS)},
	year = {2019}
}
```
