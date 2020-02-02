from helpers import *

if sys.platform=='darwin':
    basedir = '/Users/richardfeder/Documents/caltech/gan_work/results/'
elif sys.platform=='linux2':
    basedir = '/work/06147/pberger/maverick2/results/'

def garb_coll():
    n=0
    mv_bool = False
    for file in os.listdir(basedir):
        if '2019' in file:
            timedir = basedir+str(file)
            if 'params.txt' in os.listdir(timedir):
                p = open(timedir+'/params.txt')
                pdict = pickle.load(p)
                try:
                    n_epochs = pdict['n_iterations']
                except:
                    try:
                        n_epochs = pdict['n_epochs']
                    except:
                        continue
                if n_epochs < 100:
                    mv_bool = True
                    print n_epochs
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
