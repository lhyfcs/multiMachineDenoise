import time

from utils import *


def dncnn(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output


class denoiser(object):
    def __init__(self, sees=None, input_c_dim=3, batch_size=128, num_workers = 1):
        if sees != None:
            self.sess = sees
        self.input_c_dim = input_c_dim
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.X_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        decay_steps = tf.Variable(tf.constant(100), name='decay_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.Y = dncnn(self.X_, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.learning_rate = tf.train.exponential_decay(0.2,
                                           global_step=self.global_step,
                                           decay_steps=decay_steps,decay_rate=0.9)

        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        print("[*] Initialize model successfully...")


    def train(self, server, data_denoise, data_noise, batch_size, ckpt_dir, epoch, task_index=0, eval_every_epoch=2):
        init = tf.global_variables_initializer()
        # assert data range is between 0 and 1
        numBatch = int(data_denoise.shape[0] / batch_size)
        # load pretrained model
        # make summary        
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        start_time = time.time()
        # this seesion will read save and restore data automatic from check point dir
        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index == 0),
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=100,
            save_summaries_steps=100) as sess:
            sess.run(init)
            load_model_status, global_step = self.load(saver, sess, ckpt_dir)
            if load_model_status:
                #iter_num = global_step
                start_epoch = global_step // numBatch
                #start_step = global_step % numBatch
                print("[*] Model restore success!")
            else:
                #iter_num = 0
                start_epoch = 0
                #start_step = 0
                print("[*] Not find pretrained model!")
            print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, global_step))
            step = 0
            batch_id = 0
            epo = 0
            count = 0
            while not sess.should_stop() and step < epoch * numBatch:
                denoise_images = data_denoise[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                noise_images = data_noise[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss, step = sess.run([self.train_op, self.loss, self.global_step],
                                                    feed_dict={self.Y_: denoise_images, self.X_: noise_images, self.is_training: True})
                step += 1
                epo = step // numBatch
                batch_id = step % numBatch
                print("Epoch: [%4d] Global step: [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epo, batch_id, numBatch, time.time() - start_time, loss))
                count += 1
            sess.close()

    def load(self, saver, sess, checkpoint_dir):
        print("[*] Reading checkpoint...")
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
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, _ = self.load(saver, self.sess, ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        for idx in range(len(test_files)):
            start_time = time.time()
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image = self.sess.run([self.Y],feed_dict={self.Y_: clean_image, self.is_training: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f --- time:%4.4f" % (idx, psnr, time.time() - start_time))
            psnr_sum += psnr           
            save_images(os.path.join(save_dir, 'denoised%d.jpg' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
