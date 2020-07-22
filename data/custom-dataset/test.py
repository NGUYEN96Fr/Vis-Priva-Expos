import os

with open('./nicu1857.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        parts  = line.split('\t')
        if parts[1] != 'skipped':
            x1,y1,x2,y2 = [int(coord) if int(coord) > 0 else 0 for coord in parts[3].split(' ')]
            w_bbox = x2 - x1
            h_bbox = y2 - y1
            if w_bbox <= 0 or h_bbox <= 0:
                print(parts[0])
                print(parts[3])
                print('**********')
                #raise ValueError('Bounding box with negative size')