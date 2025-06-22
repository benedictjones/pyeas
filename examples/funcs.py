import numpy as np 
# from typing import Annotated
# from dataclasses import dataclass

# @dataclass
# class ValueRange:
#     """ Used to specify [custom float range typing int](https://stackoverflow.com/questions/66451253/is-there-a-way-to-specify-a-range-of-valid-values-for-a-function-argument-with-t)"""
#     min: float
#     max: float



def quadratic_order_2(
        x1:float, 
        x2:float,
    ) -> float:
    """
    Second Order Quadratic to solve

    Args:
        x1 (float): first coef
        x2 (float): second coef

    Returns:
        float: y
    """
    return (x1 + 3) + (10 * (x2 + 2)) ** 2

def quadratic_order_3(
        x1:float, 
        x2:float,
        x3:float,
    ) -> float:
    """
    Third Order Quadratic to solve

    Args:
        x1 (float): first coef
        x2 (float): second coef
        x3 (float): third coef

    Returns:
        float: y
    """
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2 + (x3**3)


def matyas(
        x1:float, 
        x2:float,
    ) -> float:
    """
    [Matyas function](https://www.sfu.ca/~ssurjano/matya.html)

    Domian: [[-10,10],[-10,10]]
    """
    for i, x in enumerate([x1, x2]):
        if x < -10 or x > 10:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [-10, 10]")
        
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

def bohach(
        x1:float, 
        x2:float,
    ) -> float:
    """
    [Bohachevsky Function](https://www.indusmic.com/post/bohachevsky-function)
    
    Domian: [[-100,100],[-100,100]]
    """
    for i, x in enumerate([x1, x2]):
        if x < -100 or x > 100:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [-100, 100]")
        
    return x1**2 +2*(x2**2)-0.3*np.cos(3*np.pi*x1)-0.4*np.cos(4*np.pi*x2)+0.7

def hc3(
        x1:float, 
        x2:float,
    ) -> float:
    """" 
    [Three hump camel functions.](https://www.indusmic.com/post/three-hump-camel-function)
    
    Domain: [-5,5],[-5,5] 
    """
    for i, x in enumerate([x1, x2]):
        if x < -5 or x > 5:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [-5, 5]")
        
    return 2*(x1**2)-1.05*(x1**4)+((x1**6)/6)+(x1*x2)+(x2**2)


def hc6(
        x1:float, 
        x2:float
    ) -> float:
    """ 
    [Six hump camel functions.](https://www.indusmic.com/post/six-hump-camel-function)
    
    The function has global minimum f (x*) = -1.0316, at x*= (0.0898,-0.7126) and (-0.0898, 0.7126).
    
    Domain: [-3,3],[-2,2]             
    """
    if x1 < -3 or x1 > 3:
        raise ValueError(f"x1={x1} exceeds limits of [-3, 3]")
    if x2 < -2 or x2 > 2:
        raise ValueError(f"x2={x2} exceeds limits of [-2, 2]")
    
    return 4*x1**2-2.1*x1**4+(x1**6)/3+x1*x2-4*x2**2+4*x2**4

def kean(
        x:float, 
        y:float
    ) -> float:
    """ 
    [Keane Function](https://www.indusmic.com/post/python-implementation-of-keane-function)

    Input Domain:
        The Keane Function is defined on input range x  [0,10] and y [0,10].

    Global Minima :
        The Keane Function has two global minimum f(x*) = 0.673667521146855 at
            x* = (1.393249070031784, 0)
            x* = (0, 1.393249070031784)
    """
    for i, x in enumerate([x, y]):
        if x < 0 or x > 10:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [0, 10]")
        
    a = -np.sin(x-y)**2*np.sin(x+y)**2
    b = np.sqrt(x*x+y*y)   
    return a/b

def ackley(
        x:float, 
        y:float
    ) -> float:
    """ 
    Ackley function
    
    Domain: [-5,5],[-5,5]
    """
    for i, x in enumerate([x, y]):
        if x < -5 or x > 5:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [-5, 5]")
        
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
 
def rose(
        x1:float, 
        x2:float
    ) -> float:
    """ 
    [Rosenbrock function.](https://www.indusmic.com/post/rosenbrock-function)
    
    Domain: [-5,10],[-5,10] 
    """
    for i, x in enumerate([x1, x2]):
        if x < -5 or x > 10:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [-5, 10]")
        
    return 100*(x2-x1**2)**2+(x1-1)**2


def beale(
        x:float, 
        y:float
    ) -> float:
    """ 
    [Beale function.](https://www.sfu.ca/~ssurjano/beale.html)
    
    Domain: [-4.5,4.5],[-4.5,4.5] 
    """
    for i, x in enumerate([x, y]):
        if x < -4.5 or x > 4.5:
            raise ValueError(f"{i}'th argument (value={x}) exceeds limits of [-4.5, 4.5]")
        
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

