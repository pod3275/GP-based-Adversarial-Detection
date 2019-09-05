# GP-based-Adversarial-Detection

Official Keras implementation of paper:

[Adversarial Detection with Gaussian Process Regression-based Detector](http://www.itiis.org/digital-library/manuscript/2475) (TIIS).

<br>

## Description

<div align="center">
  <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/model.png" width="754" height="374"><br>
</div>

- Adversarial example detection with Gaussian Process Regression-based detector.

- Existing deep learning-based adversarial detection methods require numerous adversarial images for their training. 

- The proposed method overcomes this problem by performing classification based on the statistical features of adversarial images and clean images that are extracted by Gaussian process regression **with a small number of images.**
  
<br>

## Requirements

- [Python 3.6](https://www.python.org/downloads/)
- [Tensorflow == 1.12.0](https://github.com/tensorflow/tensorflow)
- [Keras >= 2.2.4](https://github.com/keras-team/keras)
- [cleverhans >= 3.0.1](https://github.com/tensorflow/cleverhans)
- [GPy >= 1.9.6](https://github.com/SheffieldML/GPy)
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

<br>

## How to run
**1. Git clone**
```
$ git clone https://github.com/pod3275/GP-based-Adversarial-Detection.git
```

**2. Training target model**
```
$ python generate_model.py --dataset MNIST
```

- Available datasets : [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

**3. Generate adversarial examples**
```
$ python attack.py --dataset MNIST --attack JSMA
```

- Available attack methods : [FGSM](https://arxiv.org/pdf/1412.6572.pdf), [BIM](https://arxiv.org/pdf/1607.02533.pdf), [JSMA](https://arxiv.org/pdf/1511.07528.pdf), [DeepFool](https://arxiv.org/pdf/1511.04599.pdf), [CW](https://arxiv.org/pdf/1608.04644.pdf) 

  - For FGSM and BIM, you should add epsilon at the end of the attack name (*ex. CIFAR10: "--attack FGSM_e9"*)
  - Same as the **name of directory** where the adversarial data saved

**4. Detect with GP-based detector**
```
$ python attack.py --dataset MNIST --attack DeepFool --num_data_in_class 30
```

- *num_data_in_class* : number of adversarial example in one class for training detector

<br>

## Results
**- Attack accuracy**

<div align="center">
 <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/Table%201.png" width="700" height="136"><br><br>
</div>
  
**- Detection accuracy**

<div align="center">
 <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/Table%202%2C3.png" width="700" height="202"><br><br>
</div>
 
**- Number of adversarial examples for detector training**

<div align="center">
 <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/graph.png" width="828" height="279">
</div>

  - Better performance with extremely small number of adversarial example.
  
<br>

## Other operations
**1. Check adversarial image & model prediction results**
```
$ python check_label.py --dataset CIFAR10 --attack DeepFool
```

  - Check clean and adversarial images (included in *check_label.py*)
  
<div align="center">
 <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/check_image.png"><br><br>
</div>

  - Check clean and adversarial images' labels
 
<div align="center">
 <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/check_label.png" width="50%"><br><br>
</div>
  
**2. Calculate L2 perturbations of adversarial examples**
```
$ python calculate_L2_perturb.py --dataset CIFAR10 --attack BIM_e9
```

<div align="center">
 <img src="https://github.com/pod3275/GP-based-Adversarial-Detection/blob/master/assets/check_L2_perturbations.png" width="40%"><br><br>
</div>

## Citation
```
@proceedings{GP-basedAdvDetect,
	title = {Adversarial Detection with Gaussian Process Regression-based Detector},
	author = {Sangheon Lee, Noo-ri Kim, Youngwha Cho, Jae-Young Choi, Suntae Kim, Jeong-Ah Kim, Jee-Hyong Lee},
	booktitle = {KSII Transactions on Internet and Information Systems (TIIS)},
	year = {2019}
}
```
