"""
The module resolves the problem A

Usage:
    - Use default setting:
        python problem_B.py --m 5

    - Manual setting:change weights of q1, m
        python problem_B.py --q1 60 --m 5

"""
import argparse
from libs import  problemB

def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1", default=45, type= float, help="angle q1 in degree")
    parser.add_argument("--m1", default=1, type= float, help="segment 1 weight in kg")
    parser.add_argument("--m", required= True, type= float, help=" variable weight in kg")
    parser.add_argument("--l1", default=1, type= float, help="segment 1 length in meter")
    parser.add_argument("--g", default=9.8, type= float, help="gravity in m/s^2")

    return parser

if __name__ == '__main__':

    args = argument_parser().parse_args()
    solver = problemB(args.m1, args.m, args.l1, \
                    args.q1, args.g)
    solver.solve()