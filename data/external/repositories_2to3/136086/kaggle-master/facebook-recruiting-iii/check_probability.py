from common_functions import Fmeasure

tagCount = [17337, 33227, 35732, 23913, 1476]

same1 = [[Fmeasure(real, predicted, 1) for real in range(1,6)] for predicted in range(1,6)]
same2 = [[Fmeasure(real, predicted, 2) for real in range(2,6)] for predicted in range(2,6)]
same3 = [[Fmeasure(real, predicted, 3) for real in range(3,6)] for predicted in range(3,6)]
same4 = [[Fmeasure(real, predicted, 4) for real in range(4,6)] for predicted in range(4,6)]
same5 = [[Fmeasure(real, predicted, 5) for real in range(5,6)] for predicted in range(5,6)]

######################################################

# In here, we calculate the decrease fraction of F-measure
# if we further add a wrong predicted tag.

# If the number of same tags is 1
print((sum([same1[1][i]/same1[0][i]*tagCount[i]/sum(tagCount) for i in range(5)]))) # 1->2
print((sum([same1[2][i]/same1[1][i]*tagCount[i]/sum(tagCount) for i in range(5)]))) # 2->3
print((sum([same1[3][i]/same1[2][i]*tagCount[i]/sum(tagCount) for i in range(5)]))) # 3->4
print((sum([same1[4][i]/same1[3][i]*tagCount[i]/sum(tagCount) for i in range(5)]))) # 4->5
#0.772319405982
#0.816129057703
#0.845431975987
#0.866523577153

# If the number of same tags is 2
print((sum([same2[1][i]/same2[0][i]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 2->3
print((sum([same2[2][i]/same2[1][i]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 3->4
print((sum([same2[3][i]/same2[2][i]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 4->5
#0.828280661058
#0.853780368827
#0.872622479695

# If the number of same tags is 3
print((sum([same3[1][i]/same3[0][i]*tagCount[i+2] for i in range(3)])/sum(tagCount[2:]))) # 3->4
print((sum([same3[2][i]/same3[1][i]*tagCount[i+2] for i in range(3)])/sum(tagCount[2:]))) # 4->5
#0.864895920738
#0.881037613913

# If the number of same tags is 4
print((sum([same4[1][i]/same4[0][i]*tagCount[i+3] for i in range(2)])/sum(tagCount[3:]))) # 4->5
#0.889534837922

print("---------------")

######################################################

# Then we calculate the increase fraction of F-measure
# if we further add a correct predicted tag.

# If the number of same tags is 1->2
print((sum([same2[0][i]/same1[0][i+1]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 1->2
print((sum([same2[1][i]/same1[1][i+1]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 2->3
print((sum([same2[2][i]/same1[2][i+1]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 3->4
print((sum([same2[3][i]/same1[3][i+1]*tagCount[i+1] for i in range(4)])/sum(tagCount[1:]))) # 4->5
#1.58346743666
#1.65656132212
#1.70756073765
#1.74524495939

# If the number of same tags is 2->3
print((sum([same3[0][i]/same2[0][i+1]*tagCount[i+2] for i in range(3)])/sum(tagCount[2:]))) # 2->3
print((sum([same3[1][i]/same2[1][i+1]*tagCount[i+2] for i in range(3)])/sum(tagCount[2:]))) # 3->4
print((sum([same3[2][i]/same2[2][i+1]*tagCount[i+2] for i in range(3)])/sum(tagCount[2:]))) # 4->5
#1.26548217003
#1.29734388111
#1.32155642087

# If the number of same tags is 3->4
print((sum([same4[0][i]/same3[0][i+1]*tagCount[i+3] for i in range(2)])/sum(tagCount[3:]))) # 3->4
print((sum([same4[1][i]/same3[1][i+1]*tagCount[i+3] for i in range(2)])/sum(tagCount[3:]))) # 4->5
#1.16774324839
#1.18604645056

# If the number of same tags is 4->5
print((same5[0][0]/same4[0][1])) # 4->5
#1.125

print("---------------")

######################################################

# To calculate the confidence to add another predicted tag,
# we need conf*Fpp + (1-conf)*Fmm > 1, where Fpp and Fmm are the F-measure
# increase/decrease when add a further correct/wrong tag,
# i.e., we assume the F-measure increase/decrease rate can be treated
# As a weight average of its different contributions.
def confidence(Fpp, Fmm): return (1-Fmm)/(Fpp-Fmm)

# Unfortunately, We don't know the number of same tags of the predicted/real case.
# What is more, the numbers are not consistant the the number of same tags are
# not the same. Probably the most we can do is to use the average number

from numpy import mean

# Predicted tag 1->2
print((confidence(1.58346743666, 0.772319405982))) # same tag 1
#0.280689326001

# Predicted tag 2->3
print((confidence(1.65656132212, 0.816129057703))) # same tag 1
print((confidence(1.26548217003, 0.828280661058))) # same tags 2
print(("mean = "+str(mean([0.218781393911, 0.392769318994]))))
#0.218781393911
#0.392769318994
#mean = 0.305775356452

# Predicted tag 3->4
print((confidence(1.70756073765, 0.845431975987))) # same tag 1
print((confidence(1.29734388111, 0.853780368827))) # same tags 2
print((confidence(1.16774324839, 0.864895920738))) # same tags 3
print(("mean = "+str(mean([0.179286471913, 0.329647563706, 0.446112832857]))))
#0.179286471913
#0.329647563706
#0.446112832857
#mean = 0.318348956159

# Predicted tag 4->5
print((confidence(1.74524495939, 0.866523577153))) # same tag 1
print((confidence(1.32155642087, 0.872622479695))) # same tags 2
print((confidence(1.18604645056, 0.881037613913))) # same tags 3
print((confidence(1.125, 0.889534837922)))         # same tags 4
print(("mean = "+str(mean([0.151898457856, 0.283733326047, 0.390029309953, 0.469135905724]))))
#0.151898457856
#0.283733326047
#0.390029309953
#0.469135905724
#mean = 0.323699249895


