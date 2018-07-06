import argparse
from glob import glob

import tensorflow as tf
import random 

from model import denoiser
from convmodel import convdenoiser
from model_cmp import cmpdenoiser
from model_cmp_patch import convpatchdenoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=2.5, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='BSD68', help='dataset for testing')
parser.add_argument('--task_index', dest='task_index', default=0, help='set machine task index')
parser.add_argument('--job_name', dest='job_name', default='ps', help='set machine job name')
parser.add_argument('--denoise_set', dest='denoise_set', default='ultraHDImage', help='folder for denoised images')
parser.add_argument('--img_width', dest='img_width', default=3200, help='width for images')
parser.add_argument('--img_height', dest='img_height', default=1800, help='height for images')
args = parser.parse_args()


# work flag settings
flags = tf.app.flags
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_string("ps_hosts", "10.80.54.230:2222",
                    "Comma-separated list of hostname:port pairs")
#flags.DEFINE_string("worker_hosts", "10.80.51.58:2228,10.80.51.60:2229,10.80.51.49:2223,10.80.51.53:2224,10.80.51.52:2225,10.80.51.55:2226,10.80.51.48:2227",
flags.DEFINE_string("worker_hosts", "10.80.51.58:2228",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", args.job_name, "job name: worker or ps")
flags.DEFINE_integer("task_index", args.task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
FLAGS = flags.FLAGS

def denoiser_train(denoiser, server, task_index, lr):
    with load_data(filepath='./data/img_clean_pats.npy') as data:
        # if there is a small memory, please comment this line and uncomment the line99 in model.py
        print (len(data))
        data = data.astype(np.float32) / 255.0  # normalize the data to 0-1
        eval_files = glob('./data/test/{}/*.jpg'.format(args.eval_set))
        eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
        denoiser.train(server, data, eval_data, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr,
                       sample_dir=args.sample_dir, task_index=task_index)

def conv_denoise_train(denoiser, server, task_index, lr):
    denoise_files = glob('./data/test/{}/*.jpg'.format(args.denoise_set))
    noise_files = glob('./data/test/{}/*.jpg'.format(args.denoise_set+'_nodenoise'))
    denoiser.train(server, denoise_files, noise_files, width=args.img_width, height=args.img_height, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr, task_index=task_index)

def conv_patch_denoise_train(denoiser, server=None, task_index=0):
    with load_data(filepath='./data/img_denoise_pats.npy', rand=False) as data_denoise:
        with load_data(filepath='./data/img_noise_pats.npy', rand=False) as data_noise:
            data_denoise = data_denoise.astype(np.float32) / 255.0  # normalize the data to 0-1
            data_noise = data_noise.astype(np.float32) / 255.0  # normalize the data to 0-1
            denoiser.train(server, data_denoise, data_noise, './data/test/{}/*.jpg'.format(args.test_set), args.test_dir, 128, ckpt_dir=args.ckpt_dir, epoch=args.epoch, task_index=task_index)

def cmp_denoise_train(server, task_index, num_worker=7):
    with load_data(filepath='./data/img_denoise_pats.npy', rand=False) as data_denoise:
        with load_data(filepath='./data/img_noise_pats.npy', rand=False) as data_noise:
            denoiser = cmpdenoiser(num_workers=num_worker, is_chief=FLAGS.task_index==0)
            print (len(data_denoise))
            data_denoise = data_denoise.astype(np.float32) / 255.0  # normalize the data to 0-1
            data_noise = data_noise.astype(np.float32) / 255.0  # normalize the data to 0-1
            denoiser.train(server, data_denoise, data_noise, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, task_index=task_index)

def denoiser_test(denoiser):
    test_files = glob('./data/test/{}/*.jpg'.format(args.test_set))
    denoiser.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

def create_done_queue(num_workers):
    with tf.device("/job:ps/task:0"):
        return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")

def createServer():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index =="":
        raise ValueError("Must specify an explicit `task_index`")
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    #num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})        
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    return server, cluster

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    config = tf.ConfigProto(device_count={"CPU": 32}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 16,   
                intra_op_parallelism_threads = 32) 

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0
    if args.phase == 'test':
        with tf.device("/cpu:0"):
            with tf.Session(config=config) as sess:
                model = denoiser(sess, sigma=args.sigma, isDenoise=True)
                denoiser_test(model)
    elif args.phase == 'testcmp':
        with tf.device("/cpu:0"):
            with tf.Session(config=config) as sess:
                model = cmpdenoiser(sess)
                denoiser_test(model)
    elif args.phase == 'train':
        # distribution check
        server, cluster = createServer()
        if FLAGS.job_name == "ps":
            server.join() 
        elif FLAGS.job_name == "worker":
            worker_device = "/job:worker/task:%d" % FLAGS.task_index
            with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):
                model = denoiser(sigma=args.sigma)
                denoiser_train(model, server, FLAGS.task_index, lr=lr)
    elif args.phase == 'trainconv':
        server, cluster = createServer()
        if FLAGS.job_name == "ps":
            server.join() 
        elif FLAGS.job_name == "worker":
            worker_device = "/job:worker/task:%d" % FLAGS.task_index
            with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):
                model = convdenoiser()
                conv_denoise_train(model, server, FLAGS.task_index, lr=lr)
    elif args.phase == 'trainconvpatch':
        server, cluster = createServer()
        if FLAGS.job_name == "ps":
            server.join() 
        elif FLAGS.job_name == "worker":
            worker_device = "/job:worker/task:%d" % FLAGS.task_index
            with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):
                model = convpatchdenoiser(num_workers=len(FLAGS.worker_hosts.split(",")), is_chief=FLAGS.task_index==0)
                conv_patch_denoise_train(model, server, FLAGS.task_index)
    elif args.phase == 'traincmp':
        server, cluster = createServer()
        if FLAGS.job_name == "ps":
            server.join() 
        elif FLAGS.job_name == "worker":
            worker_device = "/job:worker/task:%d" % FLAGS.task_index
            with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):                
                cmp_denoise_train(server, FLAGS.task_index, num_worker=FLAGS.worker_hosts.split(","))
    elif args.phase == 'compare':
        denoise_files = glob('./data/test/{}/*.jpg'.format(args.denoise_set))
        noise_files = glob('./data/test/{}/*.jpg'.format(args.denoise_set+'_nodenoise'))
        diffs = 0.0
        for file in denoise_files:
            noise_file = find_match_file([file], noise_files)
            noise_image = load_images(noise_file[0]).astype(np.float32) / 255.0
            denoise_image = load_images(file).astype(np.float32) / 255.0
            save_img = np.clip(255 * noise_image, 0, 255).astype('uint8')
            
            save_images(os.path.join(args.test_dir, file.split("/")[-1].split("_")[0] + ".jpg"), save_img)
            diff = cal_psnr(denoise_image, noise_image)
            diffs += diff
        diffs = diffs / len(denoise_files)
    elif args.phase == 'alone_conv_patch':
        with tf.device("/cpu:0"):
            model = convpatchdenoiser()
            conv_patch_denoise_train(model)
    elif args.phase == 'batch_cmp':
        place = [n for n in range(0, 20)]
        print (place)
        np.random.shuffle(place)
        print (place)
        # with load_data(filepath='./data/img_denoise_pats.npy', rand=False) as data_denoise:
        #     with load_data(filepath='./data/img_noise_pats.npy', rand=False) as data_noise:
        #         numBatch = int(data_denoise.shape[0] / 128)
        #         noise_num = int(data_denoise.shape[0] / 128)
        #         print ('%d---%d' % (numBatch, noise_num))
        #         for i in range(0, 10):
        #             num = random.randint(0,numBatch)
        #             print (num)
        #             denoise = data_denoise[num:num + 1, :, :, :]
        #             noise = data_noise[num:num + 1, :, :, :]
        #             save_images(os.path.join(args.test_dir, 'denoised%d.jpg' % i), denoise)
        #             save_images(os.path.join(args.test_dir, 'noise%d.jpg' % i), noise)
    else:
        print('[!]Unknown phase')
        exit(0)
    

    
    # if args.use_gpu:
    #     # added to control the gpu memory
    #     print("GPU\n")
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #         model = denoiser(sess, sigma=args.sigma)
    #         if args.phase == 'train':
    #             denoiser_train(model, lr=lr)
    #         elif args.phase == 'test':
    #             denoiser_test(model)
    #         else:
    #             print('[!]Unknown phase')
    #             exit(0)
    # else:
    #     print("CPU\n")
    #     with tf.device("/cpu:0"):
    #         with tf.Session(config=config) as sess:
    #             model = denoiser(sess, sigma=args.sigma)
    #             if args.phase == 'train':
    #                 denoiser_train(model, lr=lr)
    #             elif args.phase == 'test':
    #                 denoiser_test(model)
    #             else:
    #                 print('[!]Unknown phase')
    #                 exit(0)


if __name__ == '__main__':
    tf.app.run()
