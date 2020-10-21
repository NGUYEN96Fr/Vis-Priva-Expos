"""
This file generates .slurm files to fine-tune models

Usage:

    python gen_slurm.py -f bank_mobi.slurm -p gpu-test -n node21 -e 1 -s BANK -d MOBI
"""
import os
import argparse


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="bank_mobi.slurm", help="saved slurm file name")
    parser.add_argument("-dir", "--directory", default='/home/users/apopescu/van-khoa/saved_models', help='saved directory')
    parser.add_argument("-p", "--partition", required=True, help="partition of node")
    parser.add_argument("-n", "--node", default="", required=True, help="node where to train models")
    parser.add_argument("-e", "--exclusive", required=True, help="1 to turn on the exclusive mode")
    parser.add_argument("-s", "--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument("-d", "--detector", required=True, help="MOBI, RCNN")

    return parser

def main():
    """

    Returns
    -------

    """

    Ks = [10, 15, 20]
    GAMMAs = [0, 1, 2, 3, 4]
    args = argument_parser().parse_args()
    writer = open(args.file,'w')
    writer.write('#!/bin/bash\n# SBATCH -N 1\n#SBATCH -n 1\n')
    writer.write('#SBATCH --partition %s\n#SBATCH -w %s\n'%(args.partition, args.node))

    if args.exclusive == '1':
        writer.write('#SBATCH --exclusive\n')
    writer.write('#SBATCH -J FT\n')

    writer.write('# SBATCH --output=%s\n'%(args.file.replace('slurm','out')))
    writer.write('# SBATCH --error=%s\n'%(args.file.replace('slurm', 'errs')))
    writer.write('\n')

    common_part = 'python3 main.py --config_file ../configs/'
    save_model = args.file.split('.slurm')[0]
    out_dir = os.path.join(args.directory, args.file.split('.slurm')[0])
    counter = 0

    for gamma in GAMMAs:
        for k in Ks:
            if args.detector == 'MOBI':
                model_part = common_part+'mobi_BL.yaml --model_name '+save_model+'_'+str(counter)+'.pkl '+'--situation '+args.situation
            elif args.detector == 'RCNN':
                model_part = common_part+'rcnn_BL.yaml --model_name '+save_model+'_'+str(counter)+'.pkl '+'--situation '+args.situation
            param_part = model_part+' --opts'+' FE.K '+str(k)+' FE.GAMMA '+str(gamma)+' OUTPUT.DIR '+out_dir
            writer.write(param_part+' &\n')
            counter += 1

    writer.write('wait\n')

if __name__ == '__main__':
    main()