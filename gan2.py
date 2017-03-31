import tensorflow as tf
import model as M
import numpy as np 
import cv2
import random
from datetime import datetime
import time
import imagelib as IL

ZDIM = 20
IMGPIX = 128
CHN = 1
TXTFILE = 'avcout.txt'
BETA = 0.5
LR = 0.0002
BSIZE = 64
MAXITER = 300000

IMGSIZE = [None,IMGPIX,IMGPIX,CHN]
VARS = {}

with tf.name_scope('vecInput'):
    z = tf.placeholder(tf.float32,[None,ZDIM],name='InputVec')
with tf.name_scope('imgInput'):
    imgholder = tf.placeholder(tf.float32,IMGSIZE,name='TrainImg')

def gen(inp,inpshape,reuse=False):
    with tf.variable_scope('Generator',reuse=reuse):
        mod = M.Model(inp,inpshape)
        mod.fcLayer(4*4*1024,activation=M.PARAM_RELU,batch_norm=True)
        mod.construct([4,4,1024])
        mod.deconvLayer(5,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)
        mod.deconvLayer(5,256,stride=2,activation=M.PARAM_RELU,batch_norm=True)
        mod.deconvLayer(5,128,stride=2,activation=M.PARAM_RELU,batch_norm=True)
        mod.deconvLayer(5,64,stride=2,activation=M.PARAM_RELU,batch_norm=True)
        mod.deconvLayer(5,1,stride=2,activation=M.PARAM_TANH)
        VARS['g'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')
        print(len(VARS['g']))
        return mod.get_current_layer()

def dis(inp,inpshape,reuse=False):
    with tf.variable_scope('Discriminator',reuse=reuse):
        mod = M.Model(inp,inpshape)
        mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
        mod.convLayer(5,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
        mod.convLayer(5,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
        mod.convLayer(5,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
        mod.convLayer(5,1024,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
        mod.flatten()
        mod.fcLayer(2)
        VARS['d'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')
        print (len(VARS['d']))
        return mod.get_current_layer()

generated = gen(z,[None,ZDIM])
disfalse = dis(generated,IMGSIZE)
distrue = dis(imgholder,IMGSIZE,reuse=True)

with tf.name_scope('lossG'):
    lossG = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([BSIZE],dtype=tf.int64),logits=disfalse))
    tf.summary.scalar('lossG',lossG)
with tf.name_scope('lossD'):
    lossD1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([BSIZE],dtype=tf.int64),logits=distrue))
    lossD2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([BSIZE],dtype=tf.int64),logits=disfalse))
    lossD = 0.5*(lossD1+lossD2)
    tf.summary.scalar('lossD',lossD)

with tf.name_scope('opti'):
    with tf.name_scope('optiG'):
        trainG = tf.train.AdamOptimizer(learning_rate=LR,beta1=BETA).minimize(lossG,var_list=VARS['g'])
    with tf.name_scope('optiD'):
        trainD = tf.train.AdamOptimizer(learning_rate=LR,beta1=BETA).minimize(lossD,var_list=VARS['d'])
    with tf.control_dependencies([trainG, trainD]):
        trainAll = tf.no_op(name='train')

def getGeneratedImg(sess,it):
    a = np.random.uniform(size=[4,ZDIM],low=-1.0,high=1.0)
    img = sess.run(generated,feed_dict={z:a})
    img = IL.originalImgs(img).reshape([-1,128,128])
    for i in range(4):
        cv2.imwrite('res/iter'+str(it)+'img'+str(i)+'.jpg',img[i])

def getImgs():
    print('reading image data...')
    pics = IL.fromListGetImages(TXTFILE,gray=IL.PARAM_GRAY,shape=[-1,128,128,1],resize=IMGPIX)
    pics = IL.normalizeImgs(pics)
    return pics

modelpath = 'model/'
def training():
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        print('creating log file...')
        writer = tf.summary.FileWriter('./logs/',sess.graph)
        saver = tf.train.Saver()
        M.loadSess(modelpath,sess=sess)
        imgs = list(getImgs())
        print('start training...')
        for i in range(MAXITER):
            a = np.random.uniform(size=[BSIZE,ZDIM],low=-1.0,high=1.0)
            # for _ in range(3):
                # sess.run(trainG,feed_dict={z:a})
            _,mg,lsd,lsg = sess.run([trainAll,merged,lossD,lossG],feed_dict={z:a,imgholder:random.sample(imgs,BSIZE)})
            if (i)%1 == 0:
                writer.add_summary(mg,i)
                print('iter:',i)
                print('lsd:',lsd)
                print('lsg:',lsg)
            if (i+1)%100==0:
                getGeneratedImg(sess,i+1)   
            if (i+1)%1000==0:
                saver.save(sess,modelpath+'Model_epoc'+str(i+1)+'Time'+datetime.now().strftime('%Y%m%d%H%M%S')+'.ckpt')

def generateSample():
    with tf.Session() as sess:
        loadSess(sess=sess)
        for _ in range(100):
            a = np.random.uniform(size=[1,ZDIM],low=-1.0,high=1.0)
            img,res = sess.run([generated,tf.nn.softmax(disfalse)],feed_dict={z:a})
            resimg = IL.originalImgs(res).reshape([128,128])
            cv2.imwrite('samples/'+str(cnt)+'.jpg',resimg)

training()
# generateSample()