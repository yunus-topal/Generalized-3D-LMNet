# read a.txt file line by line


objectlist = []
print("reading")
with open("a.txt", "r") as f:
    # get next line
    line = f.readline().strip()
    # if line is not empty
    while line:
        #print(line)
        objectlist.append(line)
        # get next line
        line = f.readline().strip()


print(objectlist)