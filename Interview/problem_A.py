"""
The module resolves the problem A

Usage:
    - Use default setting:
        python problem_A.py --q1 30 --q2 30

    - Manual setting:change weights of m1, m2
        python problem_A.py --q1 60 --q2 30 --m1 5 --m2 1

"""
import argparse
from libs import  problemA

def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1", required= True, type= float, help="angle q1 in degree")
    parser.add_argument("--q2", required= True, type= float, help= "angle q2 in degree")
    parser.add_argument("--m1", default=1, type= float, help="segment 1 weight in kg")
    parser.add_argument("--m2", default=1, type= float, help="segment 2 weight in kg")
    parser.add_argument("--l1", default=1, type= float, help="segment 1 length in meter")
    parser.add_argument("--l2", default=1, type= float, help="segment 2 length in meter")
    parser.add_argument("--g", default=9.8, type= float, help="gravity in m/s^2")

    return parser

if __name__ == '__main__':

    args = argument_parser().parse_args()
    solver = problemA(args.m1, args.m2, args.l1, args.l2,\
                    args.q1, args.q2, args.g)
    solver.solve()