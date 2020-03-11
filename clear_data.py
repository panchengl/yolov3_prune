import os
f = open('./data/my_data/dianli_test.txt')
a = open('./data/my_data/dianli_test_r.txt', 'w')
list_file = f.readlines()
for line in list_file:
    # print(line)
    print(len(line.split(' ')))
    if len(line.split(' ')) == 4:
        print(line)
        continue
    else:
        a.write(line )
f.close()
a.close()