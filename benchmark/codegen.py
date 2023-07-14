import csv
import os
import math

fft_par = 4

mode = ["LINEAR", "NEAREST"]
data = ["HMC", "FMC"]

def read_config(csv_file):
    data_dict = {}
    with open(csv_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines[1:]:
            if len(line) <= 1:
                continue
            contents = line.split(',')
            key = contents[1]
            key = key.replace(" ", "_").replace("(", "-").replace(")", "-")
            data_dict[key] = {}
            data_dict[key]["times"] = int(contents[2])
            data_dict[key]["dims"]  = int(contents[3])
            data_dict[key]["rows"]  = int(contents[4])
            data_dict[key]["cols"]  = int(contents[5])
            data_dict[key]["depths"]= int(contents[6])
    print(data_dict)
    return data_dict

para_dict = read_config("parameters.csv")

print(para_dict)

cnt = 0

for m in mode:
    os.system("mkdir -p ./" + m)
    for d in data:
        os.system("mkdir -p ./" + m + "/" + d)
        prefix = "./" + m + "/" + d
        for k in para_dict.keys():
            os.system("cp -r ./template" + " " + prefix + "/" + k)
            cu_file = "./template/main.cu"
            with open(cu_file, 'r') as fp:
                contents = fp.readlines()
            cu2_file = prefix + "/" + k + "/" + "main.cu"
            with open(cu2_file, 'w') as fp:
                cu2_str = ""
                status = 0
                inserted = False
                for l in contents:
                    if l.find("parameter-hook") >= 0:
                        cu2_str += "#define" + " " + m + "\n"
                        cu2_str += "#define" + " " + d + "\n"
                        cu2_str += ""
                        for par in para_dict[k].keys():
                            cu2_str += "const int " + par + " = " + str(para_dict[k][par]) + ";\n"
                        if d == "FMC":
                            comb_num = int(para_dict[k]["dims"] ** 2)
                        else:
                            comb_num = int(para_dict[k]["dims"] * (para_dict[k]["dims"] + 1) // 2)
                        cu2_str += "const int " + "combs" + " = " + str(comb_num) + ";\n"
                        cu2_str += "const int fft_combs = " + str(2 ** math.ceil(math.log2(comb_num))) + ";\n"
                        cu2_str += "const int fft_parallel = " + str(fft_par) + ";\n"
                    else:
                        cu2_str += l
                print(cu2_str, file=fp)