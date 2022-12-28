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


#east, west, north, south, south-east, north-east, south-west and north-west.  (-1, -1), (-1, 1), (1, -1), (1, 1)
        #get 8 neighbours
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: #Adjacent squares

            #Get node position
            node_position = (current_node.position[0]+new_position[0],current_node.position[1]+new_position[1])
           
            #make sure within the range
            if node_position[0] > (len(maze)-1) or node_position[0] <0 or node_position[1] >(len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue
            #make sure wlakable 
            if maze[node_position[0]][node_position[1]]!=1:
                continue
            # Create new node 
            new_node = Node(current_node, node_position) 
            # Append 
            children.append(new_node)  
            
#not working
def hill_climbing_2(maze,start,end):

    if start == end:
        print('Already in destinaation')
        return end

    path = []
    current_position = start
    visited = []

    while current_position != end:
        print('current_position',current_position,end='\n')
        #add the current position to visited list
        visited.append(current_position)
        #get possible moves from the position
        possible_moves = get_valid_neighbours(maze,current_position)
        #[print(p) for p in possible_moves]
        #print('possible moves',possible_moves,'\n')
        #set the best move to be the current position (as we cant move)
        best_move = current_position
        #set shortest distance to be the distance from current position to end
        shortest_distance = eucledian_distance(current_position,end)

        #loop through possible moves
        for move in possible_moves:
            print('possible moves ',len(possible_moves),' ',move.position,'\n')
            #if move is already vistited, skip it
            if move in visited:
                continue
            distance = eucledian_distance(move,end)
            print('\ndistance,shortest_distance',distance,shortest_distance)
            #if distance is shorter than shortest distance, set the best move to be the current move
            if(distance < shortest_distance):
                best_move = move.position
                shortest_distance = distance
        #Set current position to be the best move
        current_position = best_move
        #Add best mvoe to the path
        path.append(best_move)
        print(best_move,"****************************")
        time.sleep(5)
    return path
            
            
# def getNeighbours(solution):
#     neighbours = []
#     for i in range(len(solution)):
#         for j in range(i+1,len(solution)):
#             neighbour = solution.copy()
#             neighbour[i] = solution[j]
#             neighbour[j] = solution[i]
#             neighbours.append(neighbour)
#     return neighbours


# def getBestNeighbour(maze,neighbours):
#     bestRouteLength = routeLength(maze,neighbours[0])
#     bestNeighbour = neighbours[0]

#     for neighbour in neighbours:
#         currentRouteLength = routeLength(maze,neighbour)
#         if currentRouteLength < bestRouteLength:
#             bestRouteLength = currentRouteLength
#             bestNeighbour = neighbour
#     return bestNeighbour,bestRouteLength


# def hillClimb(maze):
#     currentSolution = randomSolution(maze)
#     currentRouteLength = routeLength(maze, currentSolution)
#     neighbours = getNeighbours(currentSolution)
#     bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(maze, neighbours)

#     while bestNeighbourRouteLength < currentRouteLength:
#         currentSolution = bestNeighbour
#         currentRouteLength = bestNeighbourRouteLength
#         neighbours = getNeighbours(currentSolution)
#         bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(maze, neighbours)

#     return currentSolution, currentRouteLength
            