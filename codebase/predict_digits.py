import tensorflow as tf
import numpy as np


n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

def load_digits():
    frozen_graph_filename = '../models/model.pb'
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    print('start')
    graph = load_digits()
    x = graph.get_tensor_by_name('prefix/dx_hold:0')
    keep_prob = graph.get_tensor_by_name('prefix/ddrop:0')
    out = graph.get_tensor_by_name('prefix/output:0')
    print('end')

    img = np.ones((1, 784), dtype = float)
    with tf.Session(graph=graph) as sess:
    # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        y_out = sess.run([out], feed_dict={ x: img, keep_prob: 1. })
    print(y_out) # [[ False ]] Yay, it works!
    # print(temp)



