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
    def __init__(self, input_c_dim=3, sigma=25, batch_size=128, num_workers = 1):
        #self.sess = sess
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = self.Y_ + tf.random_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        # self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        self.Y = dncnn(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        # opt = tf.train.SyncReplicasOptimizer(
        #     optimizer, use_locking=False,
        #     replicas_to_aggregate=0,
        #     total_num_replicas=num_workers,
        #     name="mnist_sync_replicas")        
        #self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, test_data, sample_dir, summary_merged, summary_writer):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(test_data)):
            clean_image = test_data[idx].astype(np.float32) / 255.0
            print ("%d -- %d" % (clean_image.shape[0], clean_image.shape[1]))
            output_clean_image, noisy_image, psnr_summary = self.sess.run(
                [self.Y, self.X, summary_merged],
                feed_dict={self.Y_: clean_image,
                           self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, noisyimage, outputimage)
        avg_psnr = psnr_sum / len(test_data)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def denoise(self, data):
        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
                                                              feed_dict={self.Y_: data, self.is_training: False})
        return output_clean_image, noisy_image, psnr

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
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        merged = tf.summary.merge_all()
        sv = tf.train.Supervisor(is_chief=(task_index == 0),
            logdir="./checkpoint/", 
            init_op=init,
            summary_op=None,
            global_step=self.global_step)
        
        start_time = time.time()
        # self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
        #               summary_writer=writer)  # eval_data value range is 0-255
        #writer = tf.summary.FileWriter('./logs', self.sess.graph)
        save_path = os.path.join(checkpoint_dir, 'DnCNN-tensorflow')
        with sv.managed_session(server.target) as sess:              
            saver=sv.saver
            load_model_status, global_step = self.load(saver,sess, ckpt_dir)
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
            while not sv.should_stop() and step < epoch * numBatch:
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss, step, summary = sess.run([self.train_op, self.loss, self.global_step, merged],
                                                    feed_dict={self.Y_: batch_images, self.lr: lr[epo],
                                                                self.is_training: True})
                step += 1
                epo = step // numBatch
                batch_id = step % numBatch
                if (count % 1000 == 0 and task_index == 0):
                    sv.summary_computed(sess, summary,global_step=step)
                    saver.save(sess,save_path,global_step=step)
                print("Epoch: [%4d] Global step: [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epo, batch_id, numBatch, time.time() - start_time, loss))
                if (step % numBatch == 0):
                    np.random.shuffle(data)
                count += 1
                
            if (task_index == 0):
                sv.summary_computed(sess, summary,global_step=step)
                saver.save(sess,save_path,global_step=step)
            sv.stop()
        # for epoch in range(start_epoch, epoch):
        #     np.random.shuffle(data)
        #     for batch_id in range(start_step, numBatch):
        #         batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
        #         # batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
        #         _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
        #                                          feed_dict={self.Y_: batch_images, self.lr: lr[epoch],
        #                                                     self.is_training: True})
        #         print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
        #               % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        #         iter_num += 1
        #         #writer.add_summary(summary, iter_num)
        #     if np.mod(epoch + 1, eval_every_epoch) == 0:
        #         # self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
        #         #               summary_writer=writer)  # eval_data value range is 0-255
        #         self.save(iter_num, ckpt_dir)
        # print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, saver, sess, checkpoint_dir):
        print("[*] Reading checkpoint...")
        #saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in range(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)            
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
