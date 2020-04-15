#Script providing visual representation of the performance of the CNN
#plot_path: location of data file
#plot_img: output location of the resutled plots

def perf_plot(plot_path, plot_img):
    import matplotlib.pyplot as plt
    import os

    file = open(plot_path, "r")
    
    os.mkdir(plot_img)
     
    #extract data from log file
    if (file.mode == 'r'):
        lines = file.readlines()
        epoch = []
        acc = []
        loss = []
        val_acc = []
        val_loss = []
    
        for x in lines:
            epoch.append(x.split(',')[0])
            acc.append(x.split(',')[1])
            loss.append(x.split(',')[2])
            val_acc.append(x.split(',')[3])
            val_loss.append(x.split(',')[4])

    file.close()

    #remove the label at beginning of each array
    epoch.pop(0)
    acc.pop(0)
    loss.pop(0)
    val_acc.pop(0)
    val_loss.pop(0)

    #convert to float
    for x in range (0, len(epoch)):
        epoch[x] = float(epoch[x])
        acc[x] = float(acc[x])
        loss[x] = float(loss[x])
        val_acc[x] = float(val_acc[x])
        val_loss[x] = float(val_loss[x])
    
    #accuracy vs number of epochs
    '''plt.plot(epoch,acc, 'k--x', lw = 0.75)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    figname = plot_img + 'acc_vs_epochs.png'
    plt.savefig(figname, dpi = 600)
    plt.clf()

    #val_acc vs number of epochs
    plt.plot(epoch, val_acc, 'k--x', lw = 0.75)
    plt.xlabel("Epochs")
    plt.ylabel("val_acc")
    figname = plot_img + 'valacc_vs_epochs.png'
    plt.savefig(figname, dpi = 600)
    plt.clf()

    #loss vs number of epochs
    plt.plot(epoch, loss, 'k--x', lw = 0.75)
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    figname = plot_img + 'loss_vs_epochs.png'
    plt.savefig(figname, dpi = 600)
    plt.clf()

    #val_loss vs number of epochs
    plt.plot(epoch, val_loss, 'k--x', lw = 0.75)
    plt.xlabel("Epochs")
    plt.ylabel("val_loss")
    figname = plot_img + 'valloss_vs_epochs.png'
    plt.savefig(figname, dpi = 600, bbox_inches = "tight")
    plt.clf()'''
    
    #loss and val_loss vs epochs
    plt.plot(epoch, val_loss, 'k--x', lw = 0.75, label = 'val_loss')
    plt.plot(epoch, loss, '-.', lw = 0.75, label = 'loss')
    plt.xlabel("Epochs")
    plt.ylabel("Training and Validation loss")
    figname = plot_img + 'both_loss_vs_epochs.png'
    plt.legend(loc = 'best')
    plt.savefig(figname, dpi = 400, bbox_inches = "tight")
    plt.clf()
    
    #accuracy and validation accuracy vs epochs
    plt.plot(epoch,val_acc, 'k--x', lw = 0.75, label = 'val_acc')
    plt.plot(epoch,acc, '-.', lw = 0.75, label = 'acc')
    plt.xlabel("Epochs")
    plt.ylabel("Training and Validation Accuracy")
    figname = plot_img + 'both_acc_vs_epochs.png'
    plt.legend(loc = 'best')
    plt.savefig(figname, dpi = 400)
    plt.clf()

