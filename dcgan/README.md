# Deep Convolution Generative Adversarial Networks

This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

The implementation is very close to the Torch implementation [dcgan.torch](https://github.com/soumith/dcgan.torch)

After every 100 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

## Downloading the dataset

You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) and running

```
python download.py -c bedroom
```

## Usage

```
usage: main.py [-h] --dataset DATASET [--dataroot DATAROOT]
               [--workers WORKERS] [--batchSize BATCHSIZE]
               [--imageSize IMAGESIZE] [--nz NZ] [--ngf NGF] [--ndf NDF]
               [--niter NITER] [--lr LR] [--beta1 BETA1] [--dry-run]
               [--ngpu NGPU] [--netG NETG] [--netD NETD] [--outf OUTF]
               [--manualSeed MANUALSEED] [--classes CLASSES] [--accel]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | mnist |imagenet | folder | lfw | fake
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF             number of generator filters
  --ndf NDF             number of discriminator filters
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --dry-run             check a single training cycle works
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
  --manualSeed MANUALSEED
                        manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
  --accel               enables accelerator
```
