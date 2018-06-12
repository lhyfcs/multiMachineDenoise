
[Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](http://www4.comp.polyu.edu.hk/~cslzhang/paper/DnCNN.pdf)

## Train
```
$ python generate_patches.py
$ python main.py
(note: You can add command line arguments according to the source code, for example
    $ python main.py --batch_size 64 )
```

For the provided model, it took about 4 hours in GTX 1080TI.

Here is my training loss:

**Note**: This loss figure isn't suitable for this trained model any more, but I don't want to update the figure :new_moon_with_face:


![loss](./img/loss.png)

## Test
```
$ python main.py --phase test
```
## Run
```
for i in `cat machine.txt`;do echo $i;ssh  -i ./key.pem root@$i "cd /mnt/jinfan/tensorflow/multiMachineDenoise/ && python3 main.py --epoch=10 &";done

j=0;for i in `cat machine.txt`;do echo $i $j;ssh -i ./key-jr-nonprod-huabei2.pem root@$i "cd /mnt/jinfan/tensorflow/multiMachineDenoise/ && python3 main.py --task_index=$j --job_name='worker' &";((j+=1));done

```
## TODO
- [x] Write code to support multi machine to learning
- [x] Implement server to support restful API denoise
- [x] Compare with original DnCNN.
- [x] Replace tf.nn with tf.layer.
- [ ] Replace PIL with OpenCV.
- [ ] Try tf.dataset API to speed up training process.
- [ ] Train a noise level blind model.






