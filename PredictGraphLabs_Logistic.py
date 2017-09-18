# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:26:37 2016

"""

import graphlab
import pandas as pd

print 'Reading Data'

questionInfo = pd.read_csv('QuestionInfo.csv')

#print questionInfo
# questionInfo = pd.read_csv('question_info.txt', header = None, delimiter = '\t')
# questionInfo.columns = ['QID', 'QuestionTags', 'WordIDSeq', 'CharIDSeq', 'NoUpvotes', 'NoAnswers', 'NoTopQAns']

questionInfoFrame = graphlab.SFrame(questionInfo)
# del questionInfo


userInfo = pd.read_csv('UserInfo.csv')

# userInfo = pd.read_csv('user_info.txt', header = None, delimiter = '\t')
# userInfo.columns = ['UID', 'UserTags', 'WordIDSeq', 'CharIDSeq']

userInfoFrame = graphlab.SFrame(userInfo)
# del userInfo

trainingInfo = pd.read_csv('invited_info_train.txt', header = None, delimiter ='\t')
trainingInfo.columns = ['QID','UID', 'Status']

# trainingInfoFrame = graphlab.SFrame(trainingInfo)
# del trainingInfo

#print userInfo.iloc[0]['WordIDSeq'].split('/')
testInfo = pd.read_csv('test_nolabel.txt', header = None, delimiter =',')
testInfo.columns = ['QID','UID']

# testInfoFrame = graphlab.SFrame(testInfo)
# del testInfo

#
#a = graphlab.SFrame({'ID':[1,2,3,4,5], 'Age':[25,35,24,26,25], 'Sex':['M', 'F', 'F', 'M', 'F'],'Rating':[0,1,0,1,0]})
#
#print a
trainingInfo['Sort'] = range(len(trainingInfo))

print 'Merging Train Data'

trainingInfo = pd.merge(trainingInfo, userInfo, how='inner', on='UID', left_on=None, right_on=None,left_index=True, right_index=False, sort=False,
          suffixes=('', '_User'), copy=True, indicator=False).reset_index()

# print trainingInfo

trainingInfo = pd.merge(trainingInfo, questionInfo, how='inner', on='QID', left_on=None, right_on=None,left_index=True, right_index=False, sort=False,
          suffixes=('', '_Ques'), copy=True, indicator=False).reset_index()

# print trainingInfo

trainingInfo = trainingInfo.sort_values(by=['Sort'])

trainingInfo = trainingInfo.drop(['QID', 'UID','level_0', 'index', 'Sort'],1)

trainingInfoFrame = graphlab.SFrame(trainingInfo)

# for c in trainingInfoBrandNew.columns:
# 	if(trainingInfoBrandNew[c].max() == 0):
# 		trainingInfoBrandNew = trainingInfoBrandNew.drop([c],1)


#testInfo = testInfo.copy(deep = True)
testInfo['Sort'] = range(len(testInfo))


print 'Merging Test Data'
testInfo = pd.merge(testInfo, userInfo, how='inner', on='UID', left_on=None, right_on=None,left_index=True, right_index=False, sort=False,
          suffixes=('', '_User'), copy=True, indicator=False).reset_index()

#print len(trainingInfo)

testInfo = pd.merge(testInfo, questionInfo, how='inner', on='QID', left_on=None, right_on=None,left_index=True, right_index=False, sort=False,
          suffixes=('', '_Ques'), copy=True, indicator=False).reset_index()

testInfo = testInfo.sort_values(by=['Sort'])

testInfonew = testInfo.drop(['QID', 'UID', 'level_0', 'index', 'Sort'],1)

testInfoFrame = graphlab.SFrame(testInfonew)

del questionInfo
del userInfo
del trainingInfo
#del testInfo

print 'Train Model'
m=graphlab.logistic_classifier.create(trainingInfoFrame,target='Status',convergence_threshold = 0.001, max_iterations=200)
#m = graphlab.factorization_recommender.create(trainingInfoFrame, user_id = 'UID',item_id = 'QID', user_data = userInfoFrame, item_data = questionInfoFrame, target="Status",max_iterations =200, binary_target=True,side_data_factorization=True)


#b = graphlab.SFrame({'ID':[1,2], 'Age':[23,36], 'Sex':['M','F']})

print 'Predict Labels'
testInfo['label'] = m.predict(testInfoFrame,output_type='probability')
#print m.recommend()
testInfo = testInfo[['QID', 'UID', 'label']]
testInfo.to_csv('Validate_Prediction.txt', index = False, mode = 'a')

print 'Complete!'

