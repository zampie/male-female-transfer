"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
#import cv2
#from model import CycleGAN
import utils
import glob

# FLAGS = tf.flags.FLAGS
#
# name = '202580.jpg'
#
#
# tf.flags.DEFINE_string('model', 'pretrained/female2anime.pb', 'model path (.pb)')
# tf.flags.DEFINE_string('input', 'C:/Users/Zampie/Desktop/inf/' + name, 'input image path (.jpg)')
# tf.flags.DEFINE_string('output', 'C:/Users/Zampie/Desktop/inf/a_' + name, 'output image path (.jpg)')
# tf.flags.DEFINE_integer('image_size', '128', 'image size, default: 256')

path = './dataM/'
imgs = glob.glob(path + '*.jpg')
#model = 'pretrained/male2female_5d_2.pb'
model = './pretrained/female2male_2d.pb'
image_size = 128


def inference():
    graph = tf.Graph()

    with tf.Session(graph=graph) as sess:
        with graph.as_default():
            '''
            for input in imgs:
                output = input[0:-4] + '_f.jpg'

                with tf.gfile.FastGFile(input, 'rb') as f:
                    image_data = f.read()
                    input_image = tf.image.decode_jpeg(image_data, channels=3)
                    input_image = tf.image.resize_images(input_image, size=(image_size, image_size))
                    input_image = utils.convert2float(input_image)
                    input_image.set_shape([image_size, image_size, 3])

                with tf.gfile.FastGFile(model, 'rb') as model_file:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(model_file.read())

                [output_image] = tf.import_graph_def(graph_def,
                                                     input_map={'input_image': input_image},
                                                     return_elements=['output_image:0'],
                                                     name='output')

                generated = output_image.eval()
		#cv2.imshow('frame', output_image)

                with open(output, 'wb') as f:
                    f.write(generated)
                    #cv2.imshow('frame', generated)
            '''
            for i in range(3):
                inputdata = "./dataM/img_{}.jpg".format(i)
                output = "./dataM/img_{}.jpg".format(i+1)

                with tf.gfile.FastGFile(inputdata, 'rb') as f:
                    image_data = f.read()
                    input_image = tf.image.decode_jpeg(image_data, channels=3)
                    input_image = tf.image.resize_images(input_image, size=(image_size, image_size))
                    input_image = utils.convert2float(input_image)
                    input_image.set_shape([image_size, image_size, 3])

                with tf.gfile.FastGFile(model, 'rb') as model_file:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(model_file.read())

                [output_image] = tf.import_graph_def(graph_def,
                                                     input_map={'input_image': input_image},
                                                     return_elements=['output_image:0'],
                                                     name='output')

                generated = output_image.eval()
		#cv2.imshow('frame', output_image)

                with open(output, 'wb') as f:
                    f.write(generated)
                    #cv2.imshow('frame', generated)

def main(unused_argv):
    inference()


if __name__ == '__main__':
    tf.app.run()
