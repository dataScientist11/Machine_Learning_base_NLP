import dask.dataframe as dd
#pip install dask[complete] toolz cloudpickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud,STOPWORDS
from collections import Counter

data = pd.read_csv('C:/Users/prana/Desktop/SDS/sample/appceleratorstudio.csv')

data2 = pd.read_csv('C:/Users/prana/Desktop/SDS/sample/mesos.csv')

data3 = pd.read_csv('C:/Users/prana/Desktop/SDS/sample/springxd.csv')

data4 = pd.read_csv('C:/Users/prana/Desktop/SDS/sample/titanium.csv')

data = data.append(data2,ignore_index = True)

data = data.append(data3,ignore_index = True)

data = data.append(data4,ignore_index = True)

data.shape

data = data.dropna(axis = 0)

title = data["title"]

points = data["storypoint"]

desc = data["description"]

pointslist = []
pointslist = points
titlelist = []
titlelist = title
desclist = []
desclist = desc

'''
j = 0 
while j < m:
    special_list3.append([special_list1[j],special_list2[j]])
    j = j + 1
'''
#####################Genral Wordcloud
str1 = ''.join(str(genral_list))

stopwords= set(STOPWORDS)
cloud = WordCloud(max_words=50,width=1600,height=800,stopwords=stopwords).generate(str1)


plt.figure(figsize=(20,10),facecolor='k')
plt.imshow(cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show(cloud)


##################################################
genral_list = []

genral_list = [list(pair) for pair in zip(titlelist,desclist)]

size = len(genral_list)

l1 = ['{html}','<div>','<p>','</pre>','<code>','}','&lt','&gt','</div>']
count = []
k = 0
while k < size:
    s = str(genral_list[k])
    words = s.split()    
    words = [q for q in words if q not in l1]
    size_words = len(words)
    count.append(size_words)
    k = k + 1

len(count)

genral_list[58]
count[58]

data["count"] = count

data.shape
     
words = [q for q in words if q not in l1]
    
    
d1 = data[data['storypoint'] >= 13]

d1['storypoint']

#data['count'].corr(data['storypoint'])    

d1['count'].corr(d1['storypoint'])    #length of title and description combined has 10.5% correlation with story points(>=13)

len(genral_list)#length is 9616

####################################################
stopwords = list(STOP_WORDS)
import string
punctuations = string.punctuation

v = 0
most_freq = []
most_freq_count = []
for g in genral_list:
    temp1 = []
    temp = str(g)
    temp = temp.lower().split()
    for t in temp:
        t = t.replace("{html}","")
        t = t.replace("<div>","")
        t = t.replace("<p>","")
        t = t.replace("</pre>","")
        t = t.replace("}","")
        t = t.replace("&lt","")
        t = t.replace("&gt","")
        t = t.replace("</code>","")
        t = t.replace("</p>","")
        t = t.replace("{","")
        t = t.replace("[","")
        t = t.replace("]","")
        t = t.replace("<pre>","")
        t = t.replace("<code>","") 
        t = t.replace("</div>","")
        t = t.replace(";","")
        t = t.replace("(","")
        t = t.replace(")","")
        t = t.replace(",","")
        t = t.replace(":","")
        t = t.replace("'","")
        t = t.replace("''","")
        if t not in stopwords and punctuations:
            temp1.append(t)
    #print(temp1)
    #print(temp)
    #break
    count = Counter(temp1)
    most_occur = count.most_common(4)
        
    #print(most_occur)
    if most_occur[0][0] == '' or most_occur[0][0] == '*' or most_occur[0][0] == '=' or most_occur[0][0] == '-':
       most_freq.append(most_occur[1][0])
       most_freq_count.append(most_occur[1][1])
    else:
       most_freq.append(most_occur[0][0])        
       most_freq_count.append(most_occur[0][1])
    print("\nThe",v,"th element===>",most_freq[v],"and count is",most_freq_count[v])
    v = v + 1

cnt = Counter(most_freq)
freq = cnt.most_common(20)
print(freq)

'''most_freq list will hold most freq words in a user story and most_freq_count will hold count of respective frequent keyword in that story''' 


'''
Following keywords occured frequetly in many user stories so they are very important
[('code', 208), ('error', 180), ('android', 180), ('add', 178), 
('module', 171), ('windows', 156), ('studio', 144), ('project', 121), 
('job', 112), ('app', 98), ('create', 96), ('update', 95), 
('=', 88), ('titanium', 84), ('support', 84), ('ios', 80), ('build', 77), 
('sdk', 69), ('blackberry', 61), ('file', 61)]'''


##################################Using the count of these words in all user stories to correlate with story points.
temp_code = []
for gen in genral_list:
    count = 0
    temp3 = str(gen)
    temp3 = temp3.lower().split()
    for t in temp3:
        t = t.replace("{html}","")
        t = t.replace("<div>","")
        t = t.replace("<p>","")
        t = t.replace("</pre>","")
        t = t.replace("}","")
        t = t.replace("&lt","")
        t = t.replace("&gt","")
        t = t.replace("</code>","")
        t = t.replace("</p>","")
        t = t.replace("{","")
        t = t.replace("[","")
        t = t.replace("]","")
        t = t.replace("<pre>","")
        t = t.replace("<code>","") 
        t = t.replace("</div>","")
        t = t.replace(";","")
        t = t.replace("(","")
        t = t.replace(")","")
        t = t.replace(",","")
        t = t.replace(":","")
        t = t.replace("'","")
        t = t.replace("''","")
        if t not in stopwords and punctuations:
            if t == "code":
                count = count + 1
    #print(count)
    
    temp_code.append(count)
len(temp_code)
len(points)


data["code_count"] = temp_code

data['count'].corr(data['storypoint'])

d1 = data[data['storypoint'] > 13]

d1['code_count'].corr(d1['storypoint'])



