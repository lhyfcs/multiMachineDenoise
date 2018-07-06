import time
import math
import random 

from utils import *
from glob import glob


def convmodel(inputs_, size):
    # encode
    conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

    #decode
    conv4 = tf.image.resize_nearest_neighbor(conv3, (size // 4, size // 4))
    conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

    conv5 = tf.image.resize_nearest_neighbor(conv4, (size // 2, size // 2))
    conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

    conv6 = tf.image.resize_nearest_neighbor(conv5, (size, size))
    conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

    return tf.layers.conv2d(conv6, 3, (3,3), padding='same', activation=None)


class convpatchdenoiser(object):
    def __init__(self, sees=None, input_c_dim=3, batch_size=128, img_size=40, num_workers = 7, is_chief = False):
        if sees != None:
            self.sess = sees
        self.input_c_dim = input_c_dim
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # build model
        decay_steps = tf.Variable(tf.constant(500), name='decay_step', trainable=False)
        self.inputs_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='noise_image')
        self.target_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
        logits_ = convmodel(self.inputs_, img_size)
        self.outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_, logits=logits_)
        self.cost = tf.reduce_mean(self.loss)
        self.learning_rate = tf.train.exponential_decay(0.2,
                                           global_step=self.global_step,
                                           decay_steps=decay_steps,decay_rate=0.9)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.SyncReplicasOptimizer(optimizer,
        #     replicas_to_aggregate=num_workers,
        #     total_num_replicas=num_workers)
        # self.hook=optimizer.make_session_run_hook(is_chief, num_tokens=0)
        self.train_op = optimizer.minimize(self.cost, global_step=self.global_step)
        print("[*] Initialize model successfully...")

    def evalute_test(self, sess, test_dir, save_dir, step):
        test_files = glob(test_dir)        
        for file in test_files:
            nosie_image, width, height = load_image_patches([file], patch_size = 100, stride = 100)
            nosie_image = nosie_image.astype(np.float32) / 255.0
            for idx in range(0, nosie_image.shape[0]):
                noise = nosie_image[idx: idx + 1]
                start_time = time.time()
                output, loss = sess.run([self.outputs_, self.loss],feed_dict={self.inputs_: noise})
                outputimage = np.clip(255 * output, 0, 255).astype('uint8')
                source = np.clip(255 * noise, 0, 255).astype('uint8')
                psnr = cal_psnr(source, outputimage)
                print ('test cost: %.6f, time: time:%4.4f, psnr: %.6f' %(loss, time.time() - start_time, psnr))
                save_images(os.path.join(save_dir, 'denoised%d--%d.jpg' % (step, idx)), outputimage)
            
    def train(self, server, data_denoise, data_noise, test_dir, save_dir, batch_size, ckpt_dir, epoch, task_index=0, eval_every_epoch=2):
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
        # with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index == 0),
        #     checkpoint_dir=checkpoint_dir,
        #     save_checkpoint_steps=500,
        #     save_summaries_steps=500) as sess:
        with tf.Session() as sess:
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
            place = [n for n in range(0, numBatch)]
            np.random.shuffle(place)
            # while not sess.should_stop() and step < epoch * numBatch:
            while step < epoch * numBatch:
                pos = place[batch_id]
                print ('batch id: %d' % pos)
                denoise_images = data_denoise[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                noise_images = data_noise[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss, step, lr = sess.run([self.train_op, self.cost, self.global_step, self.learning_rate],
                                                    feed_dict={self.inputs_: noise_images, self.target_: denoise_images})
                step += 1
                epo = step // numBatch
                batch_id = step % numBatch
                if batch_id == 0:
                    print ('random data')
                    np.random.shuffle(place)
                if step % 500 ==0:
                    self.evalute_test(sess, test_dir, save_dir, step)
                    self.save(sess, saver, step, ckpt_dir)
                print("Epoch: [%4d] Global step: [%4d/%4d] time: %4.4f, loss: %.6f, learning_rate: %.6f"
                      % (epo, batch_id, numBatch, time.time() - start_time, loss, lr))
                count += 1
            #sess.close()

    def load(self, saver, sess, checkpoint_dir):
        print("[*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            print ('full_path: %s' % full_path)
            if full_path:
                global_step = int(full_path.split('/')[-1].split('-')[-1].split('.')[0])
                saver.restore(sess, full_path)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0
    def save(self, sess, saver, global_step, check_dir):
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
        print ('Start saving model, step=%d' % global_step)
        saver.save(sess, check_dir, global_step)

    def test(self, test_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files) != 0, 'No testing data!'
        saver = tf.train.Saver()
        load_model_status = self.load(saver, self.sess, ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0        
        for idx in range(len(test_files)):
            nosie_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            start_time = time.time()
            output_clean_image = self.sess.run(self.outputs_, feed_dict={self.inputs_: nosie_image})
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
