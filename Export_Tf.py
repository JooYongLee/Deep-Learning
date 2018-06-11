import tensorflow as tf
import os,sys
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from google.protobuf import text_format
import argparse
import struct
import numpy as np
import cv2
CKPT_DIR = './checkmodel'
MODEL_NAME = 'CNN'
CHECK_SAVE_PATH = CKPT_DIR + '/' + MODEL_NAME + '_check.ckpt'
OUTPUT_PB_NAME = "frozen_" + MODEL_NAME
OUTPUTNODE_NAME = "pred_model"
INPUTNODE_NAME = "input"

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=CKPT_DIR, help="Model folder to export")
parser.add_argument("--output_node_names", type=str, default=OUTPUTNODE_NAME, help="The name of the output nodes, comma separated.")
parser.add_argument("--frozen_model_filename", default="frozen_model.pb", type=str,
                    help="Frozen model file to import")
parser.add_argument('--input', default = OUTPUT_PB_NAME + ".pb", help='Path to frozen graph.')
parser.add_argument('--output', default = OUTPUT_PB_NAME + "_final.pb", help='Path to output graph.')
parser.add_argument('--mode', type=str, default='train',help = 'select mode to be processed')

args = parser.parse_args()

OUTPUT_PB_NAME_TXT = OUTPUT_PB_NAME + '.pbtxt'
OUTPUT_PB_NAME_FINAL = args.output
OUTPUT_PB_NAME_OPT = 'optimized_' + MODEL_NAME + '.pb'

"""
[ref]
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
https://dato.ml/drop-dropout-from-frozen-model/
https://gist.github.com/omimo/5d393ed5b64d2ca0c591e4da04af6009
"""
def display_nodes( nodes ):
    for i, node in enumerate( nodes ):
        print('%d %s %s' % (i, node.name, node.op ))
        [print(u'└─── %d ─ %s' %(i,n)) for i,n in enumerate( node.input) ]
def my_freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/" + OUTPUT_PB_NAME + ".pbtxt"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        # with tf.gfile.GFile(output_graph, "w") as f:
        #     f.write(output_graph_def.SerializeToString())
        tf.train.write_graph(sess.graph_def,'.', output_graph)

        # output_graph_def = sess.graph_def
        print("%d ops in the final graph." % len(output_graph_def.node))
        for i in range(len(output_graph_def.node)):
            print("[%03d]"%i, output_graph_def.node[i].name )

    return output_graph_def
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    print('load......%s' % frozen_graph_filename)
    if( os.path.isfile( frozen_graph_filename )) :
        print('file exist..')
    else:
        print('file dosent exist.')
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # text_format.Merge(f.read(), graph_def)


    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        pass
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def ,name="")
    return graph
def MakeUseOfPb(loadpbname=None):
    # Let's allow the user to pass the filename as an argument

    # We use our "load_graph" function
    if loadpbname :
        graph_name = loadpbname
    else:
        graph_name = OUTPUT_PB_NAME_FINAL
    # graph = load_graph(args.output)
    if not os.path.isfile( graph_name ):
        raise ValueError('invalid protocol buf : %s '% graph_name )


    graph = load_graph(graph_name)

    print(graph_name)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes

    x = graph.get_tensor_by_name('input:0')
    y = graph.get_tensor_by_name('pred_model:0')
    print('intput tensor',x)
    print('output', y)
    Xinput = np.float32(np.random.randn(1,28,28,1))
    print('input type', Xinput.shape, Xinput.dtype)

    testcnt = 10
    for id in range(testcnt):
        Testinput, Testlabel = mnist.test.next_batch(1)
        # print(Testinput)
        disp   =   Testinput.reshape(28,28,1)

        testsrc = Testinput.reshape(1, 28, 28, 1)


    # We launch a Session
        with tf.Session(graph=graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            y_out = sess.run(y, feed_dict={x: testsrc   })
            # I taught a neural net to recognise when a sum of numbers is bigger than 45
            # it should return False in this case
            # print(y_out)  # [[ False ]] Yay, it works!
            print(y_out)

        cv2.imshow('%d'% np.argmax( y_out), disp)
        print('---------%d------------'% np.argmax(y_out))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def printalltensor():
    alltensors = [ tensor for tensor in tf.get_default_graph().get_operations()]
    for tnsr in alltensors:
        print(tnsr.name)
#def freeze_graph(model_dir, output_node_names):
def SavePb(model_dir, out_node_name):

    input_graph_path = OUTPUT_PB_NAME_TXT
    checkpoint_path = CHECK_SAVE_PATH #CKPT_DIR + '/' + MODEL_NAME + '.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = OUTPUTNODE_NAME
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = OUTPUT_PB_NAME_FINAL #'frozen_' + MODEL_NAME + '.pb'
    output_optimized_graph_name = OUTPUT_PB_NAME_OPT
    clear_devices = True

    ckpt = tf.train.checkpoint_exists(checkpoint_path)
    if ckpt:
        print('ckpt is yes')
    else:
        print('ckpt is not')
    if os.path.isdir(CKPT_DIR):
        print(checkpoint_path, 'is eixist')
    else:
        print(checkpoint_path, 'dosent eixist')

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    # Optimize for inference
    print(output_frozen_graph_name)
    input_graph_def = tf.GraphDef()
    print(input_graph_def)
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    display_nodes(input_graph_def.node)



    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        [INPUTNODE_NAME],  # an array of the input node(s)
        [OUTPUTNODE_NAME],  # an array of output nodes
        tf.float32.as_datatype_enum)

    # Save the optimized graph
    print('------------------------------------')
    print('optimize_for_inference_lib.........')
    display_nodes( output_graph_def.node )


    print(output_optimized_graph_name,'saved.....')
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

def CnnTrain():
    #########
    # 신경망 모델 구성
    ######
    # 기존 모델에서는 입력 값을 28x28 하나의 차원으로 구성하였으나,
    # CNN 모델을 사용하기 위해 2차원 평면과 특성치의 형태를 갖는 구조로 만듭니다.
    X = tf.placeholder(tf.float32, [None, 28, 28, 1],name=INPUTNODE_NAME)
    Y = tf.placeholder(tf.float32, [None, 10],name='output')
    # keep_prob = tf.placeholder(tf.float32,name='drop_prob')

    # 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
    # W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
    # L1 Conv shape=(?, 28, 28, 32)
    #    Pool     ->(?, 14, 14, 32)
    with tf.name_scope('Conv1'):
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        # tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
        # padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
        L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        # Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # L1 = tf.nn.dropout(L1, keep_prob)

    # L2 Conv shape=(?, 14, 14, 64)
    #    Pool     ->(?, 7, 7, 64)
    # W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.
    with tf.name_scope('Conv2'):
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # L2 = tf.nn.dropout(L2, keep_prob)

    # FC 레이어: 입력값 7x7x64 -> 출력값 256
    # Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줍니다.
    #    Reshape  ->(?, 256)
    with tf.name_scope('Conv3'):
        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
        L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
        L3 = tf.matmul(L3, W3,name='matmul')
        L3 = tf.nn.relu(L3)
        # L3 = tf.nn.dropout(L3, keep_prob)

    # 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
    with tf.name_scope('FC1'):
        W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
        model = tf.matmul(L3, W4)


    final_tensor = tf.nn.softmax(model,name=OUTPUTNODE_NAME)


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    # 최적화 함수를 RMSPropOptimizer 로 바꿔서 결과를 확인해봅시다.
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    printalltensor()
    #########
    # 신경망 모델 학습
    ######
    init = tf.global_variables_initializer()
    sess = tf.Session()

    batch_size = 20
    total_batch = 100


    saver = tf.train.Saver()
    ckpt    =   tf.train.get_checkpoint_state(CKPT_DIR)
    print(ckpt)
    if ckpt and tf.train.checkpoint_exists( ckpt.model_checkpoint_path ):
        print('restore complete')
        saver.restore( sess, ckpt.model_checkpoint_path )
    else:
        print('initializeing!!')
        sess.run(init)

    # tf.train.write_graph(sess.graph_def, '.', OUTPUT_PB_NAME + ".pbtxt")
    if not os.path.isdir( CKPT_DIR ):
        os.mkdir( CKPT_DIR )

    for epoch in range(10):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # batch_xs = np.random.randn(total_batch,28,28,1)
            # batch_ys = np.random.randn(total_batch,10)
            # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)

            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: batch_xs,
                                              Y: batch_ys})
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

    print('최적화 완료!')

    saver.save(sess, CHECK_SAVE_PATH )
    tf.train.write_graph(sess.graph.as_graph_def(), "", OUTPUT_PB_NAME_TXT)

if __name__ == "__main__":

    if args.mode == 'train':
        # generate tensorflow model(cnn), save check points
        CnnTrain()
        # save graph
        my_freeze_graph( args.model_dir, args.output_node_names)

        # freeze graph
        SavePb(args.model_dir, args.output_node_names)

        print('training complete\n')
    elif args.mode== 'test':
        # save
        MakeUseOfPb()