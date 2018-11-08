import sys

#NOTE: You must replace 'C:/Users/erina/cs207-FinalProject/' below with the path to the cs207-FinalProject repo on your machine
sys.path.insert(0, "C:/Users/erina/cs207-FinalProject/")
import autodiff as ad

def f(a,b):
    return 3*a/b*ad.sin(a*b+2)

out = f(ad.DualNumber('x',2),ad.DualNumber('y',3))

print(out.value)
print("expected:", 1.978716)

print(out.derivatives['x'])
print("expected:", 0.116358)


print(out.derivatives['y'])
print("expected:", -1.24157)
