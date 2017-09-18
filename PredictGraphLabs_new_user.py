# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:26:37 2016

"""

import graphlab
import pandas as pd

print 'Reading Data'

questionInfo = pd.read_csv('question_word.txt')

#print questionInfo
# questionInfo = pd.read_csv('question_info.txt', header = None, delimiter = '\t')
# questionInfo.columns = ['QID', 'QuestionTags', 'WordIDSeq', 'CharIDSeq', 'NoUpvotes', 'NoAnswers', 'NoTopQAns']

questionInfoFrame = graphlab.SFrame(questionInfo)
del questionInfo


userInfo = pd.read_csv('UserInfo.csv')

# userInfo = pd.read_csv('user_info.txt', header = None, delimiter = '\t')
# userInfo.columns = ['UID', 'UserTags', 'WordIDSeq', 'CharIDSeq']

userInfoFrame = graphlab.SFrame(userInfo)
del userInfo

trainingInfo = pd.read_csv('invited_info_train.txt', header = None, delimiter ='\t')
trainingInfo.columns = ['QID','UID', 'Status']

trainingInfoFrame = graphlab.SFrame(trainingInfo)
del trainingInfo

#print userInfo.iloc[0]['WordIDSeq'].split('/')
testInfo = pd.read_csv('validate_nolabel.txt', header = None, delimiter =',')
testInfo.columns = ['QID','UID']

testInfoFrame = graphlab.SFrame(testInfo)
#
#a = graphlab.SFrame({'ID':[1,2,3,4,5], 'Age':[25,35,24,26,25], 'Sex':['M', 'F', 'F', 'M', 'F'],'Rating':[0,1,0,1,0]})
#
#print a

print 'Train Model'
m = graphlab.factorization_recommender.create(trainingInfoFrame, user_id = 'UID',item_id = 'QID', user_data = userInfoFrame, item_data = questionInfoFrame, target="Status",max_iterations =100, binary_target=True,side_data_factorization=True)


#b = graphlab.SFrame({'ID':[1,2], 'Age':[23,36], 'Sex':['M','F']})

print 'Predict Labels'
testInfo['label'] = m.predict(testInfoFrame)
#print m.recommend()

testInfo.to_csv('Validate_Prediction.txt', index = False, mode = 'a')

print 'Complete!'

