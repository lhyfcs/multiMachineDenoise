import time

from utils import *


def convmodel(inputs_):
    # encode
    conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

    #decode
    conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
    conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

    conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
    conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

    conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
    conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

    return tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)


class convdenoiser(object):
    def __init__(self, sees=None, input_c_dim=3, sigma=2.5, batch_size=128):
        if sees != None:
            self.sess = sees
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # build model
        self.inputs_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='noise_image')
        self.target_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
        logits_ = convmodel(self.inputs_)
        self.outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_, logits=logits_)
        self.cost = tf.reduce_mean(loss)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        print("[*] Initialize model successfully...")

    def train(self, server, data, eval_data, batch_size, ckpt_dir, epoch, lr, sample_dir, task_index=0, eval_every_epoch=2):
        init = tf.global_variables_initializer()
        # assert data range is between 0 and 1
        numBatch = int(data.shape[0] / batch_size)
        # load pretrained model
        # make summary
        
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        np.random.shuffle(data)
        start_time = time.time()
        
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index == 0),
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=100,
            save_summaries_steps=100) as sess:
            sess.run(init)

            step = 0
            batch_id = 0
            epo = 0
            count = 0
            while not sess.should_stop() and step < epoch * numBatch:
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                noise_images = batch_images + self.sigma * np.random.randn(*batch_images.shape)
                noisy_imgs = np.clip(noise_images, 0., 1.)
                _, cost, step = sess.run([self.train_op, self.cost, self.global_step],
                                                    feed_dict={self.inputs_: noisy_imgs, self.target_: batch_images, self.lr: epoch[epo]})
                step += 1
                epo = step // numBatch
                batch_id = step % numBatch
                print("Epoch: [%4d] Global step: [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epo, batch_id, numBatch, time.time() - start_time, cost))
                if (step % numBatch == 0):
                    np.random.shuffle(data)
                count += 1
            sess.stop()

    def load(self, sess, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            if full_path:
                global_step = int(full_path.split('/')[-1].split('-')[-1].split('.')[0])
                saver.restore(sess, full_path)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status = self.load(self.sess, ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in range(len(test_files)):
            nosie_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            start_time = time.time()
            output_clean_image = self.sess.run([self.outputs_], feed_dict={self.inputs_: nosie_image})
            groundtruth = np.clip(255 * nosie_image, 0, 255).astype('uint8')
            #noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f-- time:4.4%f" % (idx, psnr, time.time() - start_time))
            psnr_sum += psnr
            #save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)            
            save_images(os.path.join(save_dir, 'denoised%d.jpg' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
