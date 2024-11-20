# import sys 
# sys.path.append('../')
# import data_loader
# import preprocessing
# import json

# def main():
#     data = data_loader.SpeechPairLoader('output.csv')
#     with open(r'../config.json','r') as fl:
#         config = json.load(fl)
#     config = config["preprocess"]
#     wrapper = preprocessing.preprocess.piplineV1(data,config)
#     for i in wrapper:
#         print(i)
#         break

# if __name__ == "__main__":
#     main()