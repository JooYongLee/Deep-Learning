# from tensorflow.contrib.slim.nets import alexnet
import alexnettune
import  tensorflow as tf
import os
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from tensorflow.contrib.slim.python.slim.nets import alexnet

def load_image(filename):
    if( os.path.isfile( filename )):
        img = imread(filename)

    if( img.ndim == 2 ):
        img = np.expand_dims(img,axis=2)

    img = np.float32( img )
    return img


def _change_one_hot(x):
    # print( x )
    t = np.zeros( (x.shape[0], (np.max(x) + 1)) )
    for i in range(x.shape[0]):
        t[i, x[i]] = 1
    return t
def match(x,y):
    for i, ix in enumerate(x):
        if(  ix ==y ):
            return i
    return -1
def LoadDataSet( dataset_dir , onehot = True ):
    """
    :rtype: object
    :param dataset_dir:
    :return:   train_input_name, train_output_labels, train_labels_name
    """
    train_input_name = []
    train_output_labels = []
    train_labels_name = []

    table = []

    for dirname, dirnames, filenames in os.walk( dataset_dir ):
        # print( dataset_dir + "/" + files)
        # for subdirname in dirnames:
        #     train_labels_name.append( subdirname )



        basedirname, basedir = os.path.split(dirname)

        if not basedir in table and len( filenames) > 0:
            table.append( basedir )

        tableindex = match( table, basedir )
        for filename in filenames:
            train_input_name.append(dirname + "/" + filename)
            train_output_labels.append( tableindex)

    if( len(train_input_name) == 0 or len( train_output_labels ) == 0):
        ValueError('Cannot load data')
        return []

    # print( train_input_name)
    # print(train_output_labels)

    train_output_labels = np.array( train_output_labels )
    if( onehot == True ):
        train_output_labels = _change_one_hot( train_output_labels )

    train_labels_name= table
    return train_input_name, train_output_labels, train_labels_name


def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print('----------', variable, '-------------')
        print('shape', shape, 'shape len', len(shape))

        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print('total parameter',total_parameters)

load_image("BREAST_RCC_1160.png")



# train_input_name, train_output_labels, train_labels_name =LoadDataSet("F:/Image/flower_photos/flower_photos" )
# train_inputs = []
# for i,files in enumerate(train_input_name):
#     img = load_image(files)
#     if( i >= 100):
#         break
#     train_inputs.append(img)
#
# train_inputs = np.array(train_inputs)
# print(train_inputs.shape)

# print( "total image #%d"%(len(train_input_name)))
# for i, labelnaem in enumerate(train_labels_name) :
#     print("[%d]%s"%(i,labelnaem))
def prepare_data():
    """

    :return:
    """

    testpath = './Xray-data/test'
    trainpath = './Xray-data/train'

    test_data_name, test_label_data ,_= LoadDataSet( testpath)
    train_data_name, train_labels_data,_ = LoadDataSet( trainpath )

    test_label_data = np.float32( test_label_data )
    train_labels_data = np.float32(train_labels_data)



    return train_data_name,train_labels_data, test_data_name,test_label_data

import time
def AlexnetTest():

    batch_size = 4
    input_data_channels = 1
    maxepoch = 100
    numiter = 50  # int(total_img_size / batch_size)
    output_classes = 4
    train_input_name, train_output_labels, test_input_name, test_output_labels = prepare_data()

    outlabels = []
    for i in range(test_output_labels.shape[0]):
        outlabels.append(np.argmax(test_output_labels[i]))
    np.array(outlabels)

    total_img_size = len(train_input_name)

    net_inputs = tf.placeholder(tf.float32, [None, None, None, input_data_channels])
    net_outputs = tf.placeholder(tf.float32, [None, output_classes])

    imgsize = 224
    alexnettune.default_image_size = imgsize
    print(alexnettune.default_image_size)
    alexnetmodle, end_points = alexnettune.alexnet_v2(net_inputs, num_classes=output_classes,
                                                      spatial_squeeze=False)  # spatial_squeeze=False )

    # count_params()
    checkpointpath = './checkpoints'
    checkpointfile = 'testchek.ckpt'
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=100, var_list=[var for var in tf.trainable_variables()])
    ckpt = tf.train.get_checkpoint_state(checkpointpath )

    print(ckpt.model_checkpoint_path)
    # print(ckpt)
    print(tf.train.checkpoint_exists(ckpt.model_checkpoint_path))



    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('restore checkpoints')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('initialize')
        sess.run(tf.global_variables_initializer())





    test_acc = 0.0

    t1 = time.time()

    for i, testfile in enumerate(test_input_name):
        # print(testfile)
        testimg = load_image(testfile)
        testimg = np.expand_dims(testimg, axis=0)
        pred = sess.run(alexnetmodle, feed_dict={net_inputs: testimg})
        # print(pred)
        pred = np.argmax(pred)
        # print('prediction',pred)
        # print('true_labels',test_output_labels[i], test_output_labels.shape)
        # np.argmax()
        # print(pred,np.argmax(test_output_labels[i]))
        if ((3-pred) == np.argmax(test_output_labels[i])):
            test_acc += 1.0
    test_acc /= len(test_input_name)
    t2 = time.time()

    print('accuracy %.5f\n'%test_acc)
    print('total %.6f sec...average %.6f sec '%(t2-t1,(t2-t1)/len(test_input_name)))


def AlexnetTraining():
    batch_size = 4
    input_data_channels = 1

    maxepoch = 100
    numiter = 50  #int(total_img_size / batch_size)
    output_classes = 4


    train_input_name, train_output_labels, test_input_name,test_output_labels  = prepare_data()

    total_img_size = len(train_input_name)


    net_inputs = tf.placeholder(tf.float32, [None, None, None, input_data_channels])
    net_outputs = tf.placeholder(tf.float32, [None, output_classes])

    print(train_output_labels.shape)

    print('train data %d searched..!!'% total_img_size)
    # train_input = np.float32(
    #     np.random.randn(total_img_size, alexnettune.default_image_size, alexnettune.default_image_size, 3))
    #
    # train_output = np.float32(np.random.randn(total_img_size, output_classes))


    imgsize = 224
    alexnet.default_image_size = imgsize
    print(alexnet.default_image_size)
    alexnetmodle, end_points = alexnet.alexnet_v2(net_inputs, num_classes=output_classes, spatial_squeeze=False)  # spatial_squeeze=False )
    print('alexmode',alexnetmodle)
    print('endpoints', end_points )
    # for var in tf.trainable_variables() :
    #     print( var )
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=alexnetmodle, labels=net_outputs))

    globalstep = tf.Variable(0, trainable=False, name='globstep')
    varlist = [var for var in tf.trainable_variables()]
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=[var for var in tf.trainable_variables()])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss,
                                                                     var_list=[var for var in tf.trainable_variables()],
                                                                     global_step = globalstep)

    # count_params()
    checkpointpath = './checkpoints'
    checkpointfile = 'testchek.ckpt'
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=100, var_list= [var for var in tf.trainable_variables()])
    ckpt = tf.train.get_checkpoint_state(checkpointpath + '/' + checkpointfile)

    # count_params()

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('restore checkpoints')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('initialize')
        sess.run(tf.global_variables_initializer())

    print( 'total image size',total_img_size)
    for nepoch in range(maxepoch):






        for n in range(numiter):
            batch_mask = np.random.choice(total_img_size, batch_size)

            bacth_train_input = []
            batch_train_output = []
            # print(batch_mask)
            for idx in batch_mask:
                bacth_train_input.append(load_image(train_input_name[idx]))
                batch_train_output.append(train_output_labels[idx])

            bacth_train_input = np.array(bacth_train_input)
            batch_train_output = np.array(batch_train_output)


            # print(bacth_train_input.shape)
            # print(batch_train_output.shape)
            # for key, pointlayer in end_points.items():
            #     print(key, sess.run(pointlayer, feed_dict={net_inputs: bacth_train_input}).shape)

            _, calc_loss = sess.run([optimizer, loss],
                                    feed_dict={net_inputs: bacth_train_input, net_outputs: batch_train_output})
            # print( sess.run( alexnetmodle, feed_dict={net_inputs:np.expand_dims(bacth_train_input[0],axis=0)}))

            # for var in tf.trainable_variables() :
            #     print( var )
            if( n % 10 == 0):
                print('#%d loss %.7f'%(n,calc_loss))

        test_acc = 0
        for i,testfile in enumerate( test_input_name ):
            # print(testfile)
            testimg = load_image(testfile)
            testimg = np.expand_dims( testimg, axis = 0)
            pred = sess.run(alexnetmodle, feed_dict={net_inputs:testimg })
            pred = np.argmax(pred)
            # print('prediction',pred)
            # print('true_labels',test_output_labels[i], test_output_labels.shape)
            #np.argmax()
            if( pred == np.argmax(test_output_labels[i])):
                test_acc += 1.0
        test_acc /= len( test_input_name )

        print( "%d epochs complete, save globa step #%d check points\n" % (nepoch, sess.run(globalstep)))
        print( "test accuracy %.5f\n"% test_acc)
        saver.save( sess, checkpointpath + '/' + checkpointfile ,global_step  = globalstep)

import csv
def csvread(filepath):

    if not os.path.isfile(filepath):
        ValueError('file is not exist\n')

    filename, exten = os.path.splitext( filepath )

    if exten != '.csv':
        ValueError('not CSV file!!')

    coloridx  = []
    with open(filepath,'r') as files:
        file_reader = csv.reader(files, delimiter=',')
        for rows in file_reader:
            for i,str in enumerate(rows):
                if( i == 0):
                    pass
                else:
                    coloridx.append( str )

                # print(i,str,'aa')

    # print(coloridx)
    colornum=np.array(coloridx).astype(np.int)
    print(colornum.shape)
    print(colornum)





def csvwritetest():
    savepath = './dataset'
    if not os.path.isdir(savepath):
        os.makedirs( savepath )
    dicstfile = open(savepath + "/testdict.csv","w")
    dicstfile.write("name ,%d,%d\n"%(3,4))
    dicstfile.write("yosi, file,maelong,nice,fuck\n")
    dicstfile.close()



if __name__ == "__main__":
    """
    """
    # filepath = './dataset/testdict.csv'
    # csvread(filepath)
    # AlexnetTraining()

    # AlexnetTraining()
    AlexnetTest()
    # train_input_name, train_output_labels, train_labels_name = LoadDataSet(dirs)
    #
    # print(len(train_input_name))
    # print(train_output_labels)
    # print(train_labels_name)
    #
    # dsdd = 1










# print( alexnetmodle )
# print( type(alexnetmodle ))
# # print( end_points )
# for key, value in end_points.items():
#     print(key, value)