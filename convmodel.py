import time
import math
import random 

from utils import *


def convmodel(inputs_, width, height):
    # encode
    conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

    #decode
    conv4 = tf.image.resize_nearest_neighbor(conv3, (height // 4, width // 4))
    conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

    conv5 = tf.image.resize_nearest_neighbor(conv4, (height // 2, width // 2))
    conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

    conv6 = tf.image.resize_nearest_neighbor(conv5, (height, width))
    conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

    return tf.layers.conv2d(conv6, 3, (3,3), padding='same', activation=None)


class convdenoiser(object):
    def __init__(self, sees=None, input_c_dim=3, batch_size=128):
        if sees != None:
            self.sess = sees
        self.input_c_dim = input_c_dim
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # build model
        self.width = tf.placeholder(tf.int32, None, name='image_width')
        self.height = tf.placeholder(tf.int32, None, name='image_height')
        self.decay_steps = tf.placeholder(tf.int32, None, name='decay_steps')
        self.inputs_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='noise_image')
        self.target_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
        logits_ = convmodel(self.inputs_, self.width, self.height)
        self.outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_, logits=logits_)
        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.train.exponential_decay(0.2,
                                           global_step=self.global_step,
                                           decay_steps=self.decay_steps,decay_rate=0.9)
        #self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step=self.global_step)
        print("[*] Initialize model successfully...")


    def train(self, server, denoise_files, noise_files, ckpt_dir, epoch, lr, width=3200, height=1800, batch_file=1, task_index=0, eval_every_epoch=2):
        init = tf.global_variables_initializer()
        # assert data range is between 0 and 1
        # load pretrained model
        # make summary
        
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        start_time = time.time()
        file_len = len(denoise_files)
        step_len = math.ceil(file_len / batch_file)
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index == 0),
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=100,
            save_summaries_steps=100) as sess:
            sess.run(init)
            step = 0
            batch_id = 0
            epo = 0
            count = 0
            random.shuffle(denoise_files) 
            while not sess.should_stop() and step < epoch * step_len:
                denoises = denoise_files[batch_id:batch_id+batch_file] if batch_id < file_len - 1 else denoise_files[batch_id:]
                noises = find_match_file(denoises, noise_files)
                denoise_images = load_conv_images(denoises, width, height)
                for img in denoise_images:
                    img.astype(np.float32) / 255.0
                noise_images = load_conv_images(noises, width, height)
                for img in noise_images:
                    img.astype(np.float32) / 255.0
                _, cost, step, lr = sess.run([self.train_op, self.cost, self.global_step, self.learning_rate],
                                                    feed_dict={self.inputs_: noise_images, self.target_: denoise_images, self.decay_steps: step_len, self.width: width, self.height: height})
                step += 1
                epo = step // step_len
                batch_id = step % step_len
                print("Epoch: [%4d] Global step: [%4d/%4d] time: %4.4f, loss: %.6f, learing_rate: %.6f"
                      % (epo, batch_id, step_len, time.time() - start_time, cost, lr))
                if (step % step_len == 0):
                    random.shuffle(denoise_files) 
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
        for idx in range(len(test_files)):
            nosie_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            start_time = time.time()
            output_clean_image = self.sess.run([self.outputs_], feed_dict={self.inputs_: nosie_image})
            groundtruth = np.clip(255 * nosie_image, 0, 255).astype('uint8')
            #noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f-- time:%4.4f" % (idx, psnr, time.time() - start_time))
            psnr_sum += psnr
            #save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)            
            save_images(os.path.join(save_dir, 'denoised%d.jpg' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
