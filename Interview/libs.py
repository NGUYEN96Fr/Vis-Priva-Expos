import math

class problemA(object):
    """
    The module calculates the end-effector position, and the joint torque
    presented in the problem A.

    """

    def __init__(self, m1, m2, l1, l2, q1, q2, g):
        """

        :param m1: float
            weight of the segment 1 in kg
        :param m2: float
            weight of the segment 2 in kg
        :param l1: float
            length of the segment 1 in meter
        :param l2: float
            length of the segment 1 in meter
        :param q1: float
            angle between the segment 1 and the x-axis in degree
        :param q2: float
            angle between the segment 1 and the segment 2 in degree
        :param g: float
            gravity in m/s^2
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.q1_rad = q1/180*math.pi
        self.q2_rad = q2/180*math.pi
        self.g = g

    def position(self):
        """
        Estimate the position of the end-effector

        :return:

        """

        x_coord = self.l1*math.cos(self.q1_rad) +\
                  self.l2*math.cos(self.q1_rad-self.q2_rad)

        y_coord = self.l1*math.sin(self.q1_rad) + self.l2*math.sin(self.q1_rad - self.q2_rad)
        print("#-------------------------------#")
        print('#End-effector coordinates         ')
        print("#-------------------------------#")
        print('x coordinate: ', "{:.2f}".format(x_coord), ' m')
        print('y coordinate: ', "{:.2f}".format(y_coord), ' m')


    def tau(self):
        """
        Estimate the joint torque (tau 1)

        :return:

        """
        tau1 = (1/2*self.m1 + self.m2)*self.l1*math.cos(self.q1_rad)*self.g\
               + 1/2*self.m2*self.l2*math.cos(self.q1_rad-self.q2_rad)*self.g

        print("#------------------------------#")
        print('#Joint torque                    ')
        print("#------------------------------#")
        print('tau1: ', "{:.2f}".format(tau1), ' N.m')

    def solve(self):
        """
        Solve the problem A

        :return:
        """
        print('SOLVING THE PROBLEM A ...')
        self.position()
        self.tau()
        print('FINISH !')


class problemB(object):
    """
    The module calculates the joint torque
    presented in the problem B.

    """
    def __init__(self, m1, m, l1, q1, g):
        """

        :param m1: float
            weight of the segment 1 in kg
        :param l1: float
            length of the segment 1 in meter
        :param q1: float
            angle between the segment 1 and the x-axis in degree
        :param m: float
            variable weight in kg
        :param g: float
            gravity in m/s^2
        """
        self.m1 = m1
        self.l1 = l1
        self.q1_rad = q1/180*math.pi
        self.m = m
        self.g = g

    def tau(self):
        """
        Estimate the joint torque (tau 1)

        :return:

        """
        tau1 = (1/2*self.m1 + self.m)*self.g*self.l1*math.cos(self.q1_rad)
        print("#------------------------------#")
        print('#Joint torque                    ')
        print("#------------------------------#")
        print('tau1: ', "{:.2f}".format(tau1), ' N.m')

    def solve(self):
        """
        Solve the problem B

        :return:
        """
        print('SOLVING THE PROBLEM B ...')
        self.tau()
        print('FINISH !')