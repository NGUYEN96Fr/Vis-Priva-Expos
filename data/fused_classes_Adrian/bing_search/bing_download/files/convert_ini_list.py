"""
This file convert an initial searching list to a pre-defined type list

"""

import os

ini_file =  'task_7_class_searching_bing.txt'
new_file = 'bing_slist.txt'
writer = open(new_file, 'w')

with open(ini_file) as fp:
    lines = fp.readlines()

    for line in lines:
        for category in line.split('\n')[0].split(','):
            save_name =  category.replace(' ','_')
            writer.write('%s;%s\n'%(save_name,category))

writer.close()