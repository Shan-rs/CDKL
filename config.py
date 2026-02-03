# -*- coding: utf-8 -*-


class Config(object):
    debug_file = 'debug_file'
    
    train_data_root = ''
    val_data_root = ''
 
    output_sample = 'output_sample'
    dehazing_result = 'dehazing_result'
    load_Tmodel_path = None
    load_model_path =  None #Smodel
    train_num = 43140
    batch_size = 8
    val_batch_size = 1
    num_workers = 4
    
    lr = 0.00001# initial learning rate

    new_lr = 0.00001

    weight_decay = 0.0001 

    
    max_epoch = 45
    display_iter = 50
    sample_iter = 100
    result_sample_iter = 10
    
    lr_decay = 0.70
    

