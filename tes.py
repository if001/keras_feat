import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a.shape)
print(a[0:2].mean(axis = (0)))

dic ={"a":1,"b":2,"c":3}
print(len(dic))

for value in dic.keys():
    dic[value] = 2

print(dic)
