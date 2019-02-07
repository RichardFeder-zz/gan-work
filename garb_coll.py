from helpers import *

basedir = '/Users/richardfeder/Documents/caltech/gan_work/results/'

def garb_coll():
    n=0
    mv_bool = False
    for file in os.listdir(basedir):
        if '2019' in file:
            timedir = basedir+str(file)
            if 'params.txt' in os.listdir(timedir):
                p = open(timedir+'/params.txt')
                pdict = pickle.load(p)
                if pdict['n_iterations'] < 100:
                    mv_bool = True
                    print pdict['n_iterations']
                    os.rename(timedir, basedir+'garb/'+str(file))
                    n+=1
            else:
                os.rename(timedir, basedir+'garb/'+str(file))
                mv_bool = True
                n+=1
    
    if n==0:
        print 'Nothing to move!'
    else:
        print 'Moved', n, 'folders to garbage'

garb_coll()