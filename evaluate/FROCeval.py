import numpy as np
from noduleCADEvaluationLUNA16 import noduleCADEvaluation
import os 
import csv 
from multiprocessing import Pool
import functools
import SimpleITK as sitk
#from config_training import config
from layers import nms
fold = 9
trainnum = 9
annotations_filename          = './10csv/'+str(fold)+'/annos'    +str(fold) + '.csv'
annotations_excluded_filename = './10csv/'+str(fold)+'/excluded' +str(fold) + '.csv'# path for excluded annotations for the fold
seriesuids_filename           = './10csv/'+str(fold)+'/subset'   +str(fold) + '.csv'# path for seriesuid for the fold
datapath = '../$DOWNLOADLUNA16PATH/LUNA16/subset'+str(fold)+'/' #原数据
sideinfopath = '../$LUNA16PROPOCESSPATHnew！/subset'+str(fold)+'/' #预处理后的数据
nmsthresh = 0.1  
 
bboxpath = '../traindeeplung/results/inresnet2!?-'+str(fold)+'-pbb/bbox/'  #  pbb在的地方
frocpath = './inresnet2!?/bbox/nms-'+str(fold)+'-' + str(nmsthresh) + '/'  # _focal
outputdir = './bboxoutput0/' 

# detp = [0.3, 0.4, 0.5, 0.6, 0.7]
detp = [0.3]

# nprocess = 38  # 4
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm','probability']


def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def convertcsv(bboxfname, bboxpath, detp):
    resolution = np.array([1, 1, 1])
    
    origin = np.load(sideinfopath+bboxfname[:-8]+'_origin.npy', mmap_mode='r')
    spacing = np.load(sideinfopath+bboxfname[:-8]+'_spacing.npy', mmap_mode='r')
    extendbox = np.load(sideinfopath+bboxfname[:-8]+'_extendbox.npy', mmap_mode='r')
    
    pbb = np.load(bboxpath+bboxfname, mmap_mode='r')
    diam = pbb[:, -1]
    
    check = sigmoid(pbb[:, 0]) > detp
    pbbold = np.array(pbb[check])
#    pbbold = np.array(pbb[pbb[:,0] > detp])
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15
    pbbold = pbbold[np.argsort(-pbbold[:, 0])][:1000]
    pbb = nms(pbbold, nmsthresh)
    pbb = np.array(pbb[:, :-1])
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)
    
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []
    
    for nk in range(pos.shape[0]):  # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        rowlist.append([bboxfname[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], diam[nk], 1/(1+np.exp(-pbb[nk, 0]))])
        
    return rowlist


def getfrocvalue(results_filename, outputdir):
    return noduleCADEvaluation(annotations_filename, annotations_excluded_filename,
                               seriesuids_filename, results_filename, outputdir)


def getcsv(detp):
    if not os.path.exists(frocpath):
        os.makedirs(frocpath)
        
    for detpthresh in detp:
        print('detp', detpthresh)
        # f = open(frocpath + 'predanno' + str(detpthresh) + '.csv', 'w')
        f = open(frocpath + 'predanno' + str(detpthresh) + '.csv', 'w', newline='')
        fwriter = csv.writer(f)
        fwriter.writerow(firstline)
        fnamelist = []
        for fname in os.listdir(bboxpath):
            if fname.endswith('_pbb.npy'):
                fnamelist.append(fname)
                # print fname
                # for row in convertcsv(fname, bboxpath, k):
                #     fwriter.writerow(row)
        # # return
        fnamelen = len(fnamelist)
        print('fnamelen',fnamelen)
        # predannolist = p.map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthres
        # predannolist = map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthresh), fnamelist)
        predannolist = []
        for flen in fnamelist:
            out = convertcsv(flen, bboxpath=bboxpath, detp=detpthresh)
            predannolist.append(out)
        print(len(predannolist))
        # print len(predannolist), len(predannolist[0])
        for predanno in predannolist:
            # print predanno
            for row in predanno:
                # print row
                fwriter.writerow(row)
        f.close()


def getfroc(detp):
    
    predannofnamalist = []
    outputdirlist = []
    for detpthresh in detp:
        predannofnamalist.append(outputdir + 'predanno' + str(detpthresh) + '.csv')
        outputpath = outputdir + 'predanno' + str(detpthresh) + '/'
        outputdirlist.append(outputpath)
        
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
#    froclist = p.map(getfrocvalue, predannofnamalist, outputdirlist)
        
    froclist = []
    for i in range(len(predannofnamalist)):
        froclist.append(getfrocvalue(predannofnamalist[i], outputdirlist[i]))
    
    np.save(outputdir+'froclist.npy', froclist)
           
        
if __name__ == '__main__':
    # p = Pool(processes=1)
    getcsv(detp)
#    getfroc(detp)
#     p.close()
    print('finished!')
