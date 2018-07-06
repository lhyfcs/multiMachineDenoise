from utils import *
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--f', dest='f', default='', help='string of file')
args = parser.parse_args()


patches, width, height = load_image_patches(args.f, patch_size = 200)
print ("width: %d --- height: %d" % (width, height))
print (patches.shape)
save_patches_to_image(patches, 200, width, height, './test.jpg')





