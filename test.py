import ast
list_link =[]
list_link= [ast.literal_eval(line.strip('\n'))[1] for line in open("./image_link.txt",'r')]
print(list_link)
