import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#the dataframe will only print up to 2 decimals
pd.options.display.float_format = "{:.2f}".format

#Changing the maximum rows and columns so that the entire dataframe will be displayed
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)

#Creating dictionaries for the names of the different txt files
#Creating a dictionary of String variables
names={}
stats={}
for i in range(14, 20):
    #getting the names of all the text files and storing them into dicitonaries i.e, n[14]: names14_15 etc
    temp="names"+ str(i)+"_"+str(i+1)
    names[i]=temp

for i in range(14, 20):
    temp="stats"+ str(i)+"_"+str(i+1)
    stats[i]=temp

#Reading the different txt files
#Creating a dictionary of panda dataframes
n={}
s={}
location='C:\\Users\\DY\\Desktop\\udemyScraping\\nba\\'

for i in range(14, 20):
    #for loop reads from each names20__-20__ and stats20__-20__ file
    #n[i] contains all the names and teams of the players for that sepecific year
    #s[i] contains all the stats of the players for that specfic year 
    n[i]=pd.read_csv(location + names[i] + ".txt", sep=" ", header=None)
    s[i]=pd.read_csv(location + stats[i] + ".txt", sep=" ", header=None)

#Renaming the columns
for i in range(14,20):
    n[i].rename(columns={0: 'First', 1: 'Last', 2: 'Team'}, inplace=True)
    s[i].rename(columns={0: 'Pos', 1: 'GP', 2: 'Min',
    3: 'PTS', 4: 'FGM', 5: 'FGA', 6: 'FG%', 7: 'TPM', 
    8: 'TPA', 9: 'TP%', 10: 'FTM', 11: 'FTA', 12: 'FT%',
    13: 'REB', 14: 'AST', 15: 'STL', 16: 'BLK', 17: 'TO',
    18: 'DD2', 19: 'TD3', 20:'PER'}, inplace=True)

#Merging the names of the players with their stats and storing it into a dictionary
d={}
for i in range(14,20):
    #Concatenating 2 dataframes side by side
    d[i]=pd.concat([n[i], s[i]], axis=1)


#Adding the year to the table as a new column
for i in range(14, 20):
    d[i]['Year']=2000+i

#calulating effective field goal, turnover percentage(estimate turnovers per 100 plays), true shooting % and points per 36 minutes
for i in range(14,20):
    d[i]['EFG'] = (d[i].FGM+0.5*d[i].TPM)/d[i].FGA
    d[i]['TOV%'] = d[i].TO/(d[i].FGA+0.44*d[i].FTA+d[i].TO)
    d[i]['TS%'] = (0.5*d[i].PTS)/(d[i].FGA+0.44*d[i].FTA)
    d[i]['P36'] = d[i].PTS*36/d[i].Min

#sorting the tables by position
pg={}
sg={}
sf={}
pf={}
c={}
g={}
f={}

for i in range(14, 20):
    pg[i]=d[i][d[i]['Pos']=='PG']
    sg[i]=d[i][d[i]['Pos']=='SG']
    sf[i]=d[i][d[i]['Pos']=='SF']
    pf[i]=d[i][d[i]['Pos']=='PF']
    c[i]=d[i][d[i]['Pos']=='C']
    g[i]=d[i][d[i]['Pos']=='G']
    f[i]=d[i][d[i]['Pos']=='F']

#Creating the dataframes for each of the players
giannis=pd.DataFrame()
kawhi=pd.DataFrame()
james=pd.DataFrame()
kevin=pd.DataFrame()
stephen=pd.DataFrame()
anthony=pd.DataFrame()
lebron=pd.DataFrame()
russell=pd.DataFrame()

#Going through each dataframe from 2014-2019 and adding the stats of each year accordingly to the specified players dataframe
for i in range(14, 20):
    temp=d[i][(d[i]['First']=='Giannis') & (d[i]['Last']=='Antetokounmpo')]
    giannis=pd.concat([giannis, temp])
    
    temp=d[i][(d[i]['First']=='Kawhi') & (d[i]['Last']=='Leonard')]
    kawhi=pd.concat([kawhi, temp])

    temp=d[i][(d[i]['First']=='James') & (d[i]['Last']=='Harden')]
    james=pd.concat([james, temp])

    temp=d[i][(d[i]['First']=='Kevin') & (d[i]['Last']=='Durant')]
    kevin=pd.concat([kevin, temp])

    temp=d[i][(d[i]['First']=='Stephen') & (d[i]['Last']=='Curry')]
    stephen=pd.concat([stephen, temp])

    temp=d[i][(d[i]['First']=='Anthony') & (d[i]['Last']=='Davis')]
    anthony=pd.concat([anthony, temp])

    temp=d[i][(d[i]['First']=='LeBron') & (d[i]['Last']=='James')]
    lebron=pd.concat([lebron, temp])

    temp=d[i][(d[i]['First']=='Russell') & (d[i]['Last']=='Westbrook')]
    russell=pd.concat([russell, temp])

#Creating an array of dataframes from the chosen players
#This is to be used in a for loop to reduce redundant code
players=pd.DataFrame()
players=[giannis, kawhi, james, kevin, stephen, anthony, lebron, russell]
fname=['Giannis', 'Kawhi', 'James', 'Kevin', 'Stephen', 'Anthony', 'LeBron', 'Russell']
lname=['Antetokounmpo', 'Leonard', 'Harden', 'Durant', 'Curry', 'Davis', 'James', 'Westbrook']

#---------------------------------Scatter PLot--------------------------------------------
for i in range(len(players)):
    #plt.plot takes in an array of values for its x and y parameter
    #i.e. it takes a column from the players dataframe
    plt.plot(players[i].Year, players[i].EFG, marker='o', label=fname[i]+ " " +lname[i])

location='upper left'

plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc=location)
plt.xticks(players[0].Year.tolist()+[2020])
plt.xlabel('Year')
plt.ylabel('EFG')
plt.title('EFG of top players in NBA over Time', fontdict={'fontsize':18})
#plt.show()

for i in range(len(players)):
    #plt.plot takes in an array of values for its x and y parameter
    #i.e. it takes a column from the players dataframe
    plt.plot(players[i].Year, players[i]['TS%'], marker='o', label=fname[i]+ " " +lname[i])

plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc=location)
plt.xticks(players[0].Year.tolist()+[2020])
plt.xlabel('Year')
plt.ylabel('TS%')
plt.title('TS% of top players in NBA over Time', fontdict={'fontsize':18})
#plt.show()

for i in range(len(players)):
    #plt.plot takes in an array of values for its x and y parameter
    #i.e. it takes a column from the players dataframe
    plt.plot(players[i].Year, players[i]['TD3'], marker='o', label=fname[i]+ " " +lname[i])

plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc=location)
plt.xticks(players[0].Year.tolist()+[2020])
plt.xlabel('Year')
plt.ylabel('TD3')
plt.title('TD3 of top players in NBA over Time', fontdict={'fontsize':18})
#plt.show()
#--------------------------------------------------------------------------------------


#------------------------------MACHINE LEARNING---------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Filtering data to get only the main 5 positions (we are removing F and G)
posFilter=d[15][d[15]['Pos'].isin(['PG', 'SG', 'PF', 'SF', 'C'])]

#Removing the string variables
x=posFilter[['PTS', 'FGM', 'FGA', 'FG%', 'TPM', 'TPA', 'TP%', 'FTM', 'FTA', 'FT%', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD2', 'TD3', 'PER', 'EFG', 'TOV%', 'TS%', 'P36']]
#Setting the position to be the dependant variable
y=posFilter.iloc[:,3]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('----------------  LDA  ----------------')

# Applying LDA
#Reducing the independent variables to only 4
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 4)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

#----------------------------------DOUBLE CHECKING HOW CODE WORKS-----------------------------------------
'''
y_test_list=pd.DataFrame(y_test).reset_index()
compare_testing=pd.DataFrame()
compare_testing=pd.concat([y_test_list, pd.DataFrame(y_pred)], axis=1)
compare_testing.columns=['Index', 'Pos', 'Pred_Pos']
compare_testing=compare_testing.sort_values(by='Index').reset_index()
del compare_testing['index']
print(compare_testing)
'''
#----------------------------------DOUBLE CHECKING HOW CODE WORKS-----------------------------------------

print(cm)
print(accuracy_score(y_test, y_pred))

#Getting the indeces of the players selected in the test set
indeces=list(y_test.index)
indeces=sorted(indeces)
compare=pd.DataFrame()

#Iterating through the d[15] dataset and only selecting the players that were chosen from the test set
#We are doing this to obatin the first and last names of the players
for i,row in enumerate(d[15].itertuples(),0):
    if i in indeces:
        compare=pd.concat([compare, pd.Series(row).to_frame().T], ignore_index=True)

#Renaming the columns to its original names
compare.columns=['Index', 'First', 'Last', 'Team', 'Pos', 'GP', 'Min', 'PTS', 'FGM',
    'FGA', 'FG%', 'TPM', 'TPA', 'TP%', 'FTM', 'FTA', 'FT%', 'REB', 'AST', 'STL', 'BLk',
    'TO', 'DD2', 'TD3', 'PER', 'Year', 'EFG', 'TOV%', 'TS%', 'P36']

#Filtering the dataframe to the columns we need
compare=compare[['First', 'Last', 'Pos']]

#Creating a dataframe of the indeces of the test players and their predicted positions
player_pred=pd.DataFrame()
player_pred=pd.concat([pd.DataFrame(y_test.index), pd.DataFrame(y_pred)],1)
player_pred.columns=['Old_Index', 'Pred_Pos']

#Sorting the dataframe so that the index is ascneding
player_pred=player_pred.sort_values(by='Old_Index')
player_pred=player_pred.reset_index(drop=True)

#Concatenating the original dataset with the new predicted positions of the test players
compare=pd.concat([compare, player_pred], axis=1)
del compare['Old_Index']

#print(compare)

accurate=pd.DataFrame()
innaccurate=pd.DataFrame()
good, bad, total= 0,0,0


for row in compare.itertuples():
    total+=1
    if(row[3]==row[4]):
        good+=1
    else:
        bad+=1


accurate=compare.loc[compare['Pos']==compare['Pred_Pos']]
inaccurate=compare.loc[compare['Pos']!=compare['Pred_Pos']]
print('\n', '------------------------------------\n\n', accurate.sort_values(by='Pos'), '\n\n', '------------------------------------', '\n')
print(inaccurate.sort_values(by='Pos'))
print("Accuracy: ", good/total)
print("Inaccuracy: ", bad/total)