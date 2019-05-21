print(abs(10))
print(abs(-10))

print(bool(0))
print(bool(1))
print(bool(None))
print(bool('a'))
print(bool(' '))
print(bool(''))

my_list = []
print(bool(my_list))
my_list.append("s")
print(bool(my_list))

help(my_list.clear)

eval('10*5')