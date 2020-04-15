# # Run and Evaluate the CNN model
import keras
import matplotlib.pyplot as plt
import time

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model,Model

from CNN_keras import CNN
from set_params import set_params
from perf_plot import perf_plot
from conf_matrix import *

## define where your data and test file is
data_path = './data/training_data.npy'
test_path = './data/test_data.npy'
psf_path = './data/psf_library.npy'
#checkpt_dir = '.output/test'

# ## Prepare dataset and initialise model.

# may run this cell multiple times
keras.backend.clear_session()
c_ratio, epoch, lr, batch_size, checkpt_dir, DL_model = set_params()

#start_time = time.time()
# preprocess method will take your negative exampels and return the training data and test data to you (postive+ negative)
train_data_shu, train_label_shu,test,test_label,num_planet,actual_loc = DL_model.preprocess(data_path=data_path, psf_path=psf_path, test_path=test_path,c_ratio=c_ratio)

# after everything is set, you can run your training, and the checkpoint file will appear under your checkpoint folder. 
DL_model.train(training_data = train_data_shu, training_label = train_label_shu,epoch=epoch,lr=lr,batch_size= batch_size,checkpoint_dir=checkpt_dir)

# print to file how long the computation took
# end_time = time.time() - start_time
# f = open('double_conv_time.txt','a')
# strin = str(epoch) + ' ' + str(end_time) + '\n'
# f.write(strin)
# f.close()

## evaluate the performance of the trained model
DL_model.predict_result(checkpt_dir,test,test_label)

#Performance plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.clf()
plot_path = checkpt_dir + "history/training_0.log"
plot_image = checkpt_dir + "plots/"
perf_plot(plot_path=plot_path,plot_img=plot_image)

# Print to file the structure of the CNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DL_model.print_summary(checkpt_dir, output_dir = checkpt_dir)

#confusion matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TP_idx, FP_idx = conf_matrix(checkpt_dir,test, test_label)

# ## Heatmap implementation  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Show heatmap function. 
## actual_loc is a collection of actual planet lcoation, it can be found from preprocessing up there^
## k controls the number of images , or test data for producing heatmap.
DL_model.show_heatmap(checkpt_dir,test,actual_loc,k = 5, TP_idx = TP_idx)