import os
import sys

def grf_hparam_tests(param_name, vals, imagesize=128, nepochs=20000, trainsize=0):
    command_list = []
    base_command = 'python dcgan_powerspec.py --ndf=32 --ngf=32 --cuda --imageSize='+str(imagesize)
    base_command += ' --n_epochs='+str(nepochs)+' --trainSize='+str(trainsize)
    for val in vals:
        command_list.append(base_command + ' --'+param_name+'='+str(val))
        
    for command in command_list:
        print(command)
        os.system(command)
    return command_list

grf_hparam_tests('testname', [0, 1, 2, 4])
