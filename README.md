# DeepSIM and Thin Plate Spline

DeepSIM is a generative adversarial network (GAN) trained using only one pair of images, (input, label) that are augmented using Thin Plate Spline (TPS).

![Normal Images](https://res.cloudinary.com/vincent1bt/image/upload/c_scale,w_490/v1637620616/example1_paper_1_oxxb1g.jpg)
> Normal input and label images

![Transformed images](https://res.cloudinary.com/vincent1bt/image/upload/v1637620326/example_paper_1_kfocfw.jpg)
> Label and input images after the TPS transformation

You can read more about TPS and the DeepSIM Gan in [this blog post](https://vincentblog.link/posts/thin-plate-splines-and-its-implementation-as-data-augmentation-technique).

The TPS augmentation implementation is inside *data_generator/data_utils.py*.

This network is implemented using *TensorFlow*, the original code uses *PyTorch* and you can find it [here](https://github.com/eliahuhorwitz/DeepSIM).

The project is based on the [DeepSIM: Image Shape Manipulation from a Single Augmented Training Sample](https://arxiv.org/abs/2109.06151) paper.

## Training

First you need to install **tensorflow_addons**:

```
pip install tfa-nightly
```

You can train this network using:

```
python train.py --save_generator
```

You can also train for some epochs first:

```
python train.py --epochs=8000
```

and then continue training:

```
python train.py --epochs=4000 --start-epoch=8000 --continue_training --save_generator
```

## Generate Images

Inside the *data/car_test* folder you can find the images to test the results of the model using:

```
python use_model.py
```
