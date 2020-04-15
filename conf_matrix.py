import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model,Model

import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def conf_matrix(checkpoint_dir,test,test_label,plot = True,threshold = 0.5):
    """
    Calculate and optionally plot/save the confusion matrix for last checkpoint
    :param checkpoint_dir: folder storing your checkpoints
    :param test: test data
    :param test_label: test data associated labels
    :param plot_cm: choose if to plot and save cm
    :param treshold
    
    :return TP_idx: indexes of true positives
    """
        
    #get last checkpoint and load model
    load_checkpoint = checkpoint_dir + "ckt/*"
    ckts = glob.glob(load_checkpoint)
    last_ckt = ckts[-1]
         
    keras.backend.clear_session()
    model = load_model(last_ckt)
        
    #get predictions
    pred = model.predict(test)
        
    pos1 = []
    pos2 = []
    #normalize
    for i in pred:
        pos1.append(i[0])
        pos2.append(i[1])
        
    for i in range(len(pos1)):
        sumation = pos1[i] + pos2[i]
        pos1[i] /= sumation
        pos2[i] /= sumation
        
    pos_pred_percent = pos2[:]
    
    #convert to 0 and 1
    for idx in range(len(pos2)):
        if((pos2[idx] > 0.5)):  #treshold ~0.8-0.9 for visualization purposes only
            pos2[idx] = 1
        else:
            pos2[idx] = 0
            
    pos_pred = pos2[:] #predictions for the positive class
         
    test_label_cut = []
    for idx in test_label:
        test_label_cut.append(idx[1])
        
    # f = open(os.path.join(checkpoint_dir, 'test_print.txt'), "w+")
    # f.write(str(pos_pred_percent))
    # f.write('\n\n')
    # f.write(str(pos_pred))
    # f.write('\n\n')
    # f.write(str(test_label))
    # f.close()
        
    TP_idx = []
    FP_idx = []
    TN_idx = []
    FN_idx = []
    
    predicted_yes = []
    actual_yes = []
    
    for idx in range(len(pos_pred)):
        if(pos_pred[idx] == 1 and test_label_cut[idx] == 1):  
            TP_idx.append(idx)
        elif(pos_pred[idx] == 1 and test_label_cut[idx] == 0):
            FP_idx.append(idx)
        elif(pos_pred[idx] == 0 and test_label_cut[idx] == 0):
            TN_idx.append(idx)
        else:
            FN_idx.append(idx)
            
    for idx in range(len(pos_pred)):
        if(pos_pred[idx] == 1):
            predicted_yes.append(pos_pred[idx])
            
    for idx in range(len(test_label_cut)):
        if(test_label_cut[idx] == 1):                  
            actual_yes.append(test_label_cut[idx])
            
            
    #calculate accuracy, precision, recall, Fscore
    total = len(pos_pred)
    accuracy = (len(TP_idx) + len(TN_idx)) / total
    precision = len(TP_idx) / len(predicted_yes)
    recall = len(TP_idx) / len(actual_yes)
    f1_score = 2 * (precision * recall) / (precision + recall)
    brier = brier_score_loss(test_label_cut, pos_pred_percent)
            
    #print to file
    f = open(os.path.join(checkpoint_dir, 'cm_rates.txt'), "w+")
    f.write("Accuracy : " + f"{accuracy:.2f}" + "\n")
    f.write("Precision: " + f"{precision:.2f}" + "\n")
    f.write("Recall   : " + f"{recall:.2f}" + "\n")
    f.write("f1_score : " + f"{f1_score:.2f}" + "\n")
    f.write("Brier    : " + f"{brier:.2f}" + "\n")
    f.close()

    # create matrix and plot
    cm = confusion_matrix(test_label_cut,pos_pred)
    
    #reliability graph
    prob_true_reli, prob_pred_reli = calibration_curve(test_label_cut, pos_pred_percent, normalize=False, n_bins=10, strategy='uniform')
    
    if plot:
        plot_cm(cm, output_dir = checkpoint_dir)
        plot_hist(pos_pred_percent, output_dir = checkpoint_dir)
        plot_reli(prob_true_reli, prob_pred_reli, output_dir = checkpoint_dir)
        
    return TP_idx, FP_idx
        
        
        
def plot_cm(cm, output_dir, title = None):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    classNames = ['Negative','Positive']
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation = 0)
    plt.yticks(tick_marks, classNames, rotation = 90)
    plt.autoscale(enable=True, axis='both')
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()
            
    fig_name = output_dir + "cm.png"
    plt.savefig(fig_name, dpi = 400)
    plt.clf()
    
def plot_hist(prob_pos_class, output_dir):
    """plot the histogram of probabilities for the positive class""" 
    plt.hist(prob_pos_class, bins=20, alpha = 0.75)
    plt.title('Histogram of predicted scores')
    plt.xlabel('Probability')
    plt.show()
    
    output_hist = output_dir + 'pos_pred_hist.png'
    plt.savefig(output_hist, dpi = 400)
    plt.clf()
    
def plot_reli(prob_true_reli,prob_pred_reli,output_dir):
    """diplays probability diagram of the CNN"""
    x = np.linspace(0,1,100) #diagonal line
    plt.plot(x,x,'k:', label = "Perfectly calibrated", alpha = 0.25)
    
    plt.plot(prob_true_reli,prob_pred_reli,'x')
    plt.xlabel('Predicted probabilities')
    plt.ylabel('Positives')
    plt.title('CNN reliability plot')
    plt.legend()
    plt.show()
    
    output_reli = output_dir + 'reliability_plot.png'
    plt.savefig(output_reli, dpi = 400)
    plt.clf()