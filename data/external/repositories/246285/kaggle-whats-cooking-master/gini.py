import json
import csv
from pprint import pprint
with open('train.json') as data_file:    
    data = json.load(data_file)

with open('test.json') as data_file:    
    testData = json.load(data_file)

partition = []
#copies data to partition, will use this partition during gini calculation
partition.extend(data);

def GetCuisines(partition):
  cuisines = []
  for recipe in partition:
    cuisines.append(recipe["cuisine"])
  cuisines = list(set(cuisines))
  return cuisines;

def GetIngredients(partition):
  ingredients = []
  for recipe in partition:
    for ingredient in recipe["ingredients"]:
      ingredients.append(ingredient)
  ingredients = list(set(ingredients))
  return ingredients;

#yes partition function
def YesPartition(e_ingredient, source):
  "takes an ingredient, produces a partition that includes all those ingredients from the source partition"
  newPartition = []
  for recipe in source:
    if any(e_ingredient in ingredients for ingredients in recipe["ingredients"]):
      newPartition.append(recipe)
  return newPartition;
#no partition function
def NoPartition(e_ingredient, source):
  "takes an ingredient, produces a partition that do not include all the ingredient from the source partition"
  newPartition = []
  for recipe in source:
    if not any(e_ingredient in ingredients for ingredients in recipe["ingredients"]):
      newPartition.append(recipe)
  return newPartition;

#tree node
class TreeNode:
  "these are individual nodes in a tree, should contain an ingredient or decision and yes or no functions to return the next node"
  def __init__(self, item):
    self.content = item
    self.yes = 0
    self.no = 0
  def SetYes(self, node):
    self.yes = node
  def SetNo(self, node):
    self.no = node
  def GetYes(self):
    return self.yes
  def GetNo(self):
    return self.no
  def GetContent(self):
    return self.content

#GiniFunction
def Gini(partition, ingredient):
  "takes a partition and an ingredient, and provides the gini index of that ingredient"
  partitionTotal = len(partition)
  cuisines = GetCuisines(partition)
  yeses = []
  for recipe in partition:
    if ingredient in recipe["ingredients"]:
      yeses.extend(recipe)
  yeses = [recipe for recipe in partition if ingredient in recipe["ingredients"]] #find recipes with ingredient
  yesCount = len(yeses) #count how many recipes in the partition contain matching | yescount
  nos = [recipe for recipe in partition if ingredient not in recipe["ingredients"]] #find recipes without ingredient
  noCount = len(nos) #nocount
  #count how many recipes that is cuisine1 that has ingredient, repeat for all cuisines | yescuisine1...
  yesCuisineCount = {}
  for cuisine in cuisines:
    yesCuisineTotal = 0
    for recipe in partition:
      if ingredient in recipe["ingredients"] and cuisine in recipe["cuisine"]:
        yesCuisineTotal = yesCuisineTotal + 1
    if yesCuisineTotal > 0:
      yesCuisineCount[cuisine] = yesCuisineTotal
  #count how many recipes that is cuisine1 that does not have ingredient, repeat for all cuisines | nocuisine1...
  noCuisineCount = {}
  for cuisine in cuisines:
    noCuisineTotal = 0
    for recipe in partition:
      if ingredient not in recipe["ingredients"] and cuisine in recipe["cuisine"]:
        noCuisineTotal = noCuisineTotal + 1
    if noCuisineTotal > 0:
      noCuisineCount[cuisine] = noCuisineTotal

  #Gini Formula
  yesSubtractTotal = 0
  for yesCuisine in yesCuisineCount:
    yesSubtractTotal = yesSubtractTotal + (float(yesCuisineCount[yesCuisine])/yesCount)**2
  yesIndex = yesCount/float(partitionTotal) * (1 - yesSubtractTotal)
  noSubtractTotal = 0
  for noCuisine in noCuisineCount:
    noSubtractTotal = noSubtractTotal + (float(noCuisineCount[noCuisine])/noCount)**2
  noIndex = noCount/float(partitionTotal) * (1 - noSubtractTotal)
  giniIndex = yesIndex + noIndex
  #pprint(str(ingredient) + " " + str(giniIndex))
  return giniIndex;

#traverse the tree, given recipe, return the cuisine
def FindCuisine(subTree, recipe):
  if subTree.yes == 0 and subTree.no == 0: #base case: if arrived at leaf, return leaf value
    return subTree.content;
  #test to see if traverse right or left
  if subTree.content in recipe["ingredients"]:
    return FindCuisine(subTree.yes, recipe)
  return FindCuisine(subTree.no, recipe)

#find lowest gini index
def TreeBuilder(partition, treeNode):
  "use partition for gini index, use treeNode to set next node. Call treeNode by treeNode->Yes or treeNode->No"
  #basecase: when all recipes in partition have the same cuisine, add cuisine to the tree and return
  cuisines = GetCuisines(partition)
  if len(cuisines) < 2:
    treeNode = TreeNode(cuisines[0])
    #print(treeNode.content)
    return treeNode; #what should be returned?
  ingredients = GetIngredients(partition)
  giniIngredient = {} #dictionary of ingredients and their corresponding gini index
  for ingredient in ingredients:
    giniIngredient[ingredient] = Gini(partition, ingredient)
  selectCuisine = min(giniIngredient, key=giniIngredient.get)
  
  #add to the tree
  treeNode = TreeNode(selectCuisine)
  #print(treeNode.content)
  
  #partition then call gini again
  yesPartition = YesPartition(selectCuisine, partition) #partition that follows yes path
  treeNode.yes = TreeBuilder(yesPartition, treeNode.yes)
  #print(treeNode.yes.content)
  noPartition = NoPartition(selectCuisine, partition) #partition that follows no path
  treeNode.no = TreeBuilder(noPartition, treeNode.no)
  #print(treeNode.no.content)
  return treeNode;

#tree = DecisionTree(0) #create empty tree
tree = TreeBuilder(partition, TreeNode(0)) #build tree

print(FindCuisine(tree, partition[0]))

#csv builder, for each recipe, call tree traversal
resultDict = {}
for recipe in testData:
  resultDict[recipe["id"]] = FindCuisine(tree, recipe)
writer = csv.writer(open('result.csv', 'wb'))
for key, value in resultDict.items():
   writer.writerow([key, value])


#Tests
#pprint(Partition("cooking cream", partition)) #test for partition function
#pprint(ingredients); #test for creating all nonduplicating ingredients list
#End Tests


#Misc Notes
#print data[0]["ingredients"][1] #prints first ingredient of first food
#print data[0]["cuisine"] #prints cuisine of the first food

#Algorithm Notes
#Gini(total) = 1 - (cuisine1/totalcount)^2 - ...

#Gini(ingredient:yes or no)(partition): yescount/partitiontotal ( 1 - (yescuisine1/yescount)^2 - (yescuisine2/yescount)^2)...) + nocount/partitiontotal ( 1 - (nocuisine1/nocount)^2...)
