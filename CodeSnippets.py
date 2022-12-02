Code snippets Machine Learning

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV


#Preprocessing 

for c in train.select_dtypes('object').columns:
    train[c].fillna(train[c].mode()[0],inplace=True)
    train[c] = train[c].astype('category')

def encode_categories(df):
    label_encoder = preprocessing.LabelEncoder()
    for c in df.select_dtypes('category').columns:
        df[c] = label_encoder.fit_transform(df[c])
		
def regressors_scores(df,models,run_times,test_size=0.3):
    scores = []
    for i in range(1,run_times+1):
        X_train,X_test,y_train,y_test = train_test_split(df[X],df[y],test_size=test_size)
        for m in models:
            m.fit(X_train,y_train.values.ravel())
            sc = m.score(X_test,y_test.values.ravel())
            scores.append({
                            'Run':i,
                            'Model':type(m).__name__,
                            'Score':sc
                          })
    return pd.DataFrame(scores)		
	
regressors_scores(train,
                 [
                     linear_model.LinearRegression(),
                     RandomForestRegressor(n_estimators=20)
                 ])
                 
#----------------------------------------------------------------------#                 
models = [
 linear_model.LinearRegression(),
 RandomForestRegressor(n_estimators=20),
 DecisionTreeRegressor(random_state=20)
]
d = regressors_scores(train,models,10)
plt.figure(figsize=(10,5))
for mdl in d.Model.unique():
    plt.plot(d.Run.unique(),d[d.Model==mdl].Score.values,label=mdl)
plt.legend()
plt.show()