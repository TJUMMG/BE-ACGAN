import os
import numpy as np
import tensorflow as tf
import cv2
from acgan_test import ACGAN
import glob


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement=True) 
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True 


def test():

    x = tf.placeholder(tf.float32, [None, None, None, 3])
    downscaled = tf.placeholder(tf.float32, [None, None, None, 3])
    is_training = tf.placeholder(tf.bool, [])

    b = tf.placeholder(tf.int32)
    h = tf.placeholder(tf.int32)
    w = tf.placeholder(tf.int32)

    model = ACGAN(x, downscaled, is_training, b, h, w)
    sess = tf.Session()
    init = tf.global_variables_initializer() 
    sess.run(init)

    # Restore the ACGAN network
    saver = tf.train.Saver()
    saver.restore(sess, './model/latest')

    testdata = glob.glob('./testdata/*.png')

    for pic_fname in testdata:
        pic_name = pic_fname.split('/')[-1]
        pure_name = pic_name.split('.')[0]
        
        print pure_name

        pic = cv2.imread(pic_fname)
        x_batch = pic[np.newaxis,:,:,:]

        B, H, W, C = x_batch.shape
        downs = downscale(x_batch, H, W, C)
        res = x_batch - downs

        res_f = normalize(res)
        downs_f = normalize(downs)

        fake = sess.run(model.imitation,
            feed_dict={x: res_f, downscaled: downs_f, is_training: False, b:B, h:H, w:W})

        adda = fake + downs_f
        
        a = np.clip(adda[0], 0, 1)
        im = np.uint8(a * 255)

        path = './results_48/'
        if os.path.exists(path) == False:
            os.mkdir(path)

        cv2.imwrite(path + pure_name + '_rlt.png', im)
        

def normalize(images):

    return np.array([image/255.0 for image in images])

    
    
def downscale(images, H, W, C):

    # generate 4-bit LBD input
    downs = [[[[0 for p in range(C)] for k in range(W)] for j in range(H)] for i in range(len(images))]
    for ii in range(len(images)):
        for j in range(len(images[ii])):
            for k in range(len(images[ii][j])):
                for p in range(len(images[ii][j][k])):
                    tmp = bin(images[ii][j][k][p])
                    tmp_quan = tmp[:-4]+ '0000'
                    downs[ii][j][k][p] = int(tmp_quan,2)
    downs_np = np.array(downs, dtype=np.uint8)

    return downs_np
    
    
if __name__ == '__main__':

    test()

