#Script for setting up the hyperparameters of the CNN; outputs information file info_params

def set_params():
    import os
    import keras
    from CNN_keras import CNN
    
    print_info = True
    
    #output folder
    dirName = 'comp_heat_single/'
    checkpt_dir = './final_report/results/' + dirName
    
    #training parameters
    c_ratio = [0.05, 0.10]                                      #[0.01, 0.02]
    epoch = 60                                               #1
    lr = 0.0001                                                  #0.001
    batch_size = 64                                             #32
    
    DL_model = CNN()
    
    #                                                           Initial values for hyperparameters
    DL_model.gen_param['cross_valid']= 1                        #1
    cross_valid = str(DL_model.gen_param['cross_valid'])
    
    DL_model.gen_param['num_class']= 2                          #2   
    num_class = str(DL_model.gen_param['num_class'])
    
    DL_model.gen_param['image_size']= 64                        #64
    image_size = str(DL_model.gen_param['image_size'])
    
    DL_model.gen_param['channel']= 1                            #1
    channel = str(DL_model.gen_param['channel'])
    
    DL_model.gen_param['valid_size']= 0.2                       #0.2
    valid_size = str(DL_model.gen_param['valid_size'])

    DL_model.net_param['droprate'] = 0.5                        #0.5
    droprate = str(DL_model.net_param['droprate'])
    
    DL_model.net_param['dense_unit'] = 128                      #256
    dense_unit = str(DL_model.net_param['dense_unit'])
    
    DL_model.net_param['model_type'] = 'vgg'                    #'vgg'
    model_type = str(DL_model.net_param['model_type'])
    
    DL_model.net_param['num_features'] = 4                      #8
    num_features = str(DL_model.net_param['num_features'])
    
    DL_model.net_param['num_layers'] = 3                        #3
    num_layers = str(DL_model.net_param['num_layers'])
    
    DL_model.net_param['filter_size'] = 3                       #3
    filter_size = str(DL_model.net_param['filter_size'])
    
    DL_model.net_param['num_dense'] = 1                         #1
    num_dense = str(DL_model.net_param['num_dense'])
    
    DL_model.net_param['double_conv'] = False              #True
    double_conv = str(DL_model.net_param['double_conv'])
    
    DL_model.net_param['display_net'] = False                 #False
    display_net = str(DL_model.net_param['display_net'])

    DL_model.train_param['decay'] = 0.0                         #0.0
    decay = str(DL_model.train_param['decay'])
    
    #create information file
    os.makedirs(checkpt_dir)
    
    if print_info:
        fileName = checkpt_dir + 'params_info.txt'  
        file = open(fileName, "w")
    
        str1 ='CNN hyperparams used for computation:\n\ncross_valid =' + cross_valid+'\nnum_class ='+num_class+'\nimage_size ='+image_size+'\nchannel ='+channel+'\nvalid_size ='+valid_size
        str2 ='\n\ndroprate ='+droprate+'\ndense_unit ='+dense_unit+'\nmodel_type ='+model_type+'\nnum_features ='+num_features+'\nnum_layers ='+num_layers+'\nfilter_size ='+filter_size
        str3 ='\nnum_dense ='+num_dense+'\ndouble_conv ='+double_conv+'\ndisplay_net ='+display_net+'\n\ndecay ='+decay+'\n\nc_ratio ='+str(c_ratio)+'\nepoch ='+str(epoch)+'\nlr ='+str(lr)+'\nbatch_size ='+str(batch_size)
        strings = str1 + str2 + str3
        file.write(strings)
        file.close()
    
    
    return c_ratio, epoch, lr, batch_size, checkpt_dir, DL_model