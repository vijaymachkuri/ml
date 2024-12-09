import math
print("pi value is:", math.pi)
print("tau value is:",math.tau)
print("Eulers number is",math.e)
print(math.nan)
print(type(math.nan))
print(math.inf)
print(-math.inf)
p = 10.1
print(math.ceil(p))
q = 9.99
print(math.floor(q))
n = 5
print("{}! = {}".format(n, math.factorial(n)))
print(math.gcd(25,120))
print(math.fabs(-25))
deposit = 10000
interest_rate = 0.04
number_years = 5
final_amount = deposit * math.pow(1 + interest_rate, number_years)
print("The final amount after five years is", final_amount)
print(type(math.pow(2, 4)))
print(math.exp(3))
print(math.pow(math.e, 3))
print(math.sqrt(16))
print(math.log10(1000))
print(math.sin(math.pi/4))
print(math.cos(math.pi/2))
print(math.tan(math.pi/4))
print(math.tan(math.pi/6))
print(math.tan(0))
x = 0.5
print(math.cosh(x))
print(math.sinh(x))
print(math.tanh(x))