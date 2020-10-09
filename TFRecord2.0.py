import tensorflow as tf
import os

input_dir = "S:/face/faces_64/"
output_dir = "S:/face/faces2.tfreocrds"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


image_file = os.listdir(input_dir)
image_file.sort(key=lambda x:int(x[:-4]))

writer = tf.io.TFRecordWriter(output_dir)

for image_name in image_file:
    print(image_name)
    image_path = input_dir + image_name

    image_raw = open(image_path, 'rb').read()

    example = tf.train.Example(features = tf.train.Features(feature = {'image_raw':_bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()