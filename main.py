import argparse
from glob import glob

import tensorflow as tf

from model import denoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='BSD68', help='dataset for testing')
parser.add_argument('--task_index', dest='task_index', default=0, help='set machine task index')
parser.add_argument('--job_name', dest='job_name', default='ps', help='set machine job name')
args = parser.parse_args()


# work flag settings
flags = tf.app.flags
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_string("ps_hosts", "10.80.54.230:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "10.80.51.58:2228,10.80.51.60:2229,10.80.51.49:2223,10.80.51.53:2224,10.80.51.52:2225,10.80.51.55:2226,10.80.51.48:2227",
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


def denoiser_test(denoiser):
    test_files = glob('./data/test/{}/*.jpg'.format(args.test_set))
    denoiser.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

def create_done_queue(num_workers):
    with tf.device("/job:ps/task:0"):
        return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")

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
                model = denoiser(sess, sigma=args.sigma)
                denoiser_test(model)
    elif args.phase == 'train':
        # distribution check
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

        if FLAGS.job_name == "ps":
            server.join() 
        elif FLAGS.job_name == "worker": 
            worker_device = "/job:worker/task:%d" % FLAGS.task_index
            with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):
                model = denoiser(sigma=args.sigma)
                denoiser_train(model, server, FLAGS.task_index, lr=lr)
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
