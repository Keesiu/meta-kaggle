from prediction import *

infoList = getInfoList()

print("infoList -- done")

f = open("train.csv", 'r')
trainList = f.read().split('\n')[1:]
f.close()

g = open('pred.csv', 'w')
g.write("id,start.1,start.2,start.3,start.4,start.5,start.6,start.7,start.8,start.9,start.10,start.11,start.12,start.13,start.14,start.15,start.16,start.17,start.18,start.19,start.20,start.21,start.22,start.23,start.24,start.25,start.26,start.27,start.28,start.29,start.30,start.31,start.32,start.33,start.34,start.35,start.36,start.37,start.38,start.39,start.40,start.41,start.42,start.43,start.44,start.45,start.46,start.47,start.48,start.49,start.50,start.51,start.52,start.53,start.54,start.55,start.56,start.57,start.58,start.59,start.60,start.61,start.62,start.63,start.64,start.65,start.66,start.67,start.68,start.69,start.70,start.71,start.72,start.73,start.74,start.75,start.76,start.77,start.78,start.79,start.80,start.81,start.82,start.83,start.84,start.85,start.86,start.87,start.88,start.89,start.90,start.91,start.92,start.93,start.94,start.95,start.96,start.97,start.98,start.99,start.100,start.101,start.102,start.103,start.104,start.105,start.106,start.107,start.108,start.109,start.110,start.111,start.112,start.113,start.114,start.115,start.116,start.117,start.118,start.119,start.120,start.121,start.122,start.123,start.124,start.125,start.126,start.127,start.128,start.129,start.130,start.131,start.132,start.133,start.134,start.135,start.136,start.137,start.138,start.139,start.140,start.141,start.142,start.143,start.144,start.145,start.146,start.147,start.148,start.149,start.150,start.151,start.152,start.153,start.154,start.155,start.156,start.157,start.158,start.159,start.160,start.161,start.162,start.163,start.164,start.165,start.166,start.167,start.168,start.169,start.170,start.171,start.172,start.173,start.174,start.175,start.176,start.177,start.178,start.179,start.180,start.181,start.182,start.183,start.184,start.185,start.186,start.187,start.188,start.189,start.190,start.191,start.192,start.193,start.194,start.195,start.196,start.197,start.198,start.199,start.200,start.201,start.202,start.203,start.204,start.205,start.206,start.207,start.208,start.209,start.210,start.211,start.212,start.213,start.214,start.215,start.216,start.217,start.218,start.219,start.220,start.221,start.222,start.223,start.224,start.225,start.226,start.227,start.228,start.229,start.230,start.231,start.232,start.233,start.234,start.235,start.236,start.237,start.238,start.239,start.240,start.241,start.242,start.243,start.244,start.245,start.246,start.247,start.248,start.249,start.250,start.251,start.252,start.253,start.254,start.255,start.256,start.257,start.258,start.259,start.260,start.261,start.262,start.263,start.264,start.265,start.266,start.267,start.268,start.269,start.270,start.271,start.272,start.273,start.274,start.275,start.276,start.277,start.278,start.279,start.280,start.281,start.282,start.283,start.284,start.285,start.286,start.287,start.288,start.289,start.290,start.291,start.292,start.293,start.294,start.295,start.296,start.297,start.298,start.299,start.300,start.301,start.302,start.303,start.304,start.305,start.306,start.307,start.308,start.309,start.310,start.311,start.312,start.313,start.314,start.315,start.316,start.317,start.318,start.319,start.320,start.321,start.322,start.323,start.324,start.325,start.326,start.327,start.328,start.329,start.330,start.331,start.332,start.333,start.334,start.335,start.336,start.337,start.338,start.339,start.340,start.341,start.342,start.343,start.344,start.345,start.346,start.347,start.348,start.349,start.350,start.351,start.352,start.353,start.354,start.355,start.356,start.357,start.358,start.359,start.360,start.361,start.362,start.363,start.364,start.365,start.366,start.367,start.368,start.369,start.370,start.371,start.372,start.373,start.374,start.375,start.376,start.377,start.378,start.379,start.380,start.381,start.382,start.383,start.384,start.385,start.386,start.387,start.388,start.389,start.390,start.391,start.392,start.393,start.394,start.395,start.396,start.397,start.398,start.399,start.400,diff.1,diff.2,diff.3,diff.4,diff.5,diff.6,diff.7,diff.8,diff.9,diff.10,diff.11,diff.12,diff.13,diff.14,diff.15,diff.16,diff.17,diff.18,diff.19,diff.20,diff.21,diff.22,diff.23,diff.24,diff.25,diff.26,diff.27,diff.28,diff.29,diff.30,diff.31,diff.32,diff.33,diff.34,diff.35,diff.36,diff.37,diff.38,diff.39,diff.40,diff.41,diff.42,diff.43,diff.44,diff.45,diff.46,diff.47,diff.48,diff.49,diff.50,diff.51,diff.52,diff.53,diff.54,diff.55,diff.56,diff.57,diff.58,diff.59,diff.60,diff.61,diff.62,diff.63,diff.64,diff.65,diff.66,diff.67,diff.68,diff.69,diff.70,diff.71,diff.72,diff.73,diff.74,diff.75,diff.76,diff.77,diff.78,diff.79,diff.80,diff.81,diff.82,diff.83,diff.84,diff.85,diff.86,diff.87,diff.88,diff.89,diff.90,diff.91,diff.92,diff.93,diff.94,diff.95,diff.96,diff.97,diff.98,diff.99,diff.100,diff.101,diff.102,diff.103,diff.104,diff.105,diff.106,diff.107,diff.108,diff.109,diff.110,diff.111,diff.112,diff.113,diff.114,diff.115,diff.116,diff.117,diff.118,diff.119,diff.120,diff.121,diff.122,diff.123,diff.124,diff.125,diff.126,diff.127,diff.128,diff.129,diff.130,diff.131,diff.132,diff.133,diff.134,diff.135,diff.136,diff.137,diff.138,diff.139,diff.140,diff.141,diff.142,diff.143,diff.144,diff.145,diff.146,diff.147,diff.148,diff.149,diff.150,diff.151,diff.152,diff.153,diff.154,diff.155,diff.156,diff.157,diff.158,diff.159,diff.160,diff.161,diff.162,diff.163,diff.164,diff.165,diff.166,diff.167,diff.168,diff.169,diff.170,diff.171,diff.172,diff.173,diff.174,diff.175,diff.176,diff.177,diff.178,diff.179,diff.180,diff.181,diff.182,diff.183,diff.184,diff.185,diff.186,diff.187,diff.188,diff.189,diff.190,diff.191,diff.192,diff.193,diff.194,diff.195,diff.196,diff.197,diff.198,diff.199,diff.200,diff.201,diff.202,diff.203,diff.204,diff.205,diff.206,diff.207,diff.208,diff.209,diff.210,diff.211,diff.212,diff.213,diff.214,diff.215,diff.216,diff.217,diff.218,diff.219,diff.220,diff.221,diff.222,diff.223,diff.224,diff.225,diff.226,diff.227,diff.228,diff.229,diff.230,diff.231,diff.232,diff.233,diff.234,diff.235,diff.236,diff.237,diff.238,diff.239,diff.240,diff.241,diff.242,diff.243,diff.244,diff.245,diff.246,diff.247,diff.248,diff.249,diff.250,diff.251,diff.252,diff.253,diff.254,diff.255,diff.256,diff.257,diff.258,diff.259,diff.260,diff.261,diff.262,diff.263,diff.264,diff.265,diff.266,diff.267,diff.268,diff.269,diff.270,diff.271,diff.272,diff.273,diff.274,diff.275,diff.276,diff.277,diff.278,diff.279,diff.280,diff.281,diff.282,diff.283,diff.284,diff.285,diff.286,diff.287,diff.288,diff.289,diff.290,diff.291,diff.292,diff.293,diff.294,diff.295,diff.296,diff.297,diff.298,diff.299,diff.300,diff.301,diff.302,diff.303,diff.304,diff.305,diff.306,diff.307,diff.308,diff.309,diff.310,diff.311,diff.312,diff.313,diff.314,diff.315,diff.316,diff.317,diff.318,diff.319,diff.320,diff.321,diff.322,diff.323,diff.324,diff.325,diff.326,diff.327,diff.328,diff.329,diff.330,diff.331,diff.332,diff.333,diff.334,diff.335,diff.336,diff.337,diff.338,diff.339,diff.340,diff.341,diff.342,diff.343,diff.344,diff.345,diff.346,diff.347,diff.348,diff.349,diff.350,diff.351,diff.352,diff.353,diff.354,diff.355,diff.356,diff.357,diff.358,diff.359,diff.360,diff.361,diff.362,diff.363,diff.364,diff.365,diff.366,diff.367,diff.368,diff.369,diff.370,diff.371,diff.372,diff.373,diff.374,diff.375,diff.376,diff.377,diff.378,diff.379,diff.380,diff.381,diff.382,diff.383,diff.384,diff.385,diff.386,diff.387,diff.388,diff.389,diff.390,diff.391,diff.392,diff.393,diff.394,diff.395,diff.396,diff.397,diff.398,diff.399,diff.400\n")

print("trainList -- done")

score = []

for trainStr in trainList:
    if (trainStr):
        trainData = list(map(int,trainStr.split(',')))
        gridStart = list2grid(trainData[2:402], edge)
        gridStop = list2grid(trainData[402:802], edge)
        gridPred = deltaBack(trainData[1], gridStop, infoList)
        gridDiff = [[(gridPred[i][j] - gridStart[i][j]) for j in range(edge)] for i in range(edge)]
        score.append([difference(gridStart, gridPred),
                      difference(gridStop, deltaStep(trainData[1], gridPred))])
        g.write(",".join(map(str, trainData[:1]+grid2list(gridPred)+grid2list(gridDiff))))
        g.write('\n')

from numpy import mean
print((mean(list(zip(*score))[0][:50]), mean(list(zip(*score))[0][51:])))
print((mean(list(zip(*score))[1][:50]), mean(list(zip(*score))[1][51:])))

g.close()

