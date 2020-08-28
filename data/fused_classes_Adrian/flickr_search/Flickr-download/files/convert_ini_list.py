"""
This file convert an initial searching list to a pre-defined type list

"""

import os

ini_file =  'task_7_class_searching_flickr.txt'
new_file = 'flickr_slist.txt'
writer = open(new_file, 'w')

with open(ini_file) as fp:
    lines = fp.readlines()

    for line in lines:
        print('**')
        print(line.split(','))
        for category in line.split('\n')[0].split(','):
            save_name =  category.replace(' ','_')
            writer.write('%s\t%s\n'%(category,save_name))

writer.close()