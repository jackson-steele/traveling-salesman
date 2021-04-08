# /usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
import random
from TSPClasses import *
import heapq
import itertools
import copy


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        original_cities = self._scenario.getCities()
        ncities = len(original_cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while not foundTour and time.time() - start_time < time_allowance and count < ncities:
            cities = original_cities.copy()
            startCity = cities[count]
            route = []

            route.append(startCity)
            cities.remove(startCity)
            currentCity = startCity

            # add the next closest city until all cities are added
            for i in range(ncities - 1):
                currentCity = self.findClosestCity(currentCity, cities)
                route.append(currentCity)
                cities.remove(currentCity)

            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    def findClosestCity(self, currentCity, cities):
        closestCity = None
        min = math.inf
        for c in cities:
            cost = currentCity.costTo(c)
            if (cost <= min):
                min = cost
                closestCity = c

        return closestCity

    # TODO: This isn't used anywhere. Should we remove it?
    def nearest_insertion_all(self, time_allowance=60.0):
        '''Tries all cities as a starting point and returns best route found'''
        # start with a random city, add to "tour"
        # IN LOOP: find nearest city outside tour to a city in the tour, add
        # Complexity: n cities * n_in_tour * (n_cities - n_in_tour)

        results = {}
        original_cities = self._scenario.getCities()
        ncities = len(original_cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        best_cost = math.inf
        best_bssf = None
        for count in range(ncities):
            cities = original_cities.copy()
            startCity = cities[count]
            route = []

            route.append(startCity)
            cities.remove(startCity)

            for i in range(ncities - 1):
                city_added = self.addClosestCityToRoute(route, cities)
                cities.remove(city_added)

            bssf = TSPSolution(route)
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

            if bssf.cost < best_cost:
                best_cost = bssf.cost
                best_bssf = bssf

        end_time = time.time()
        results['cost'] = best_bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count + 1
        results['soln'] = best_bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    # TODO: This isn't used anywhere. Should we remove it?
    def nearest_insertion(self, time_allowance=60.0):
        '''Add closest city to route in each iteration, returns when solution is found'''
        # start with a random city, add to "tour"
        # IN LOOP: find nearest city outside tour to a city in the tour, add
        # Complexity: n cities * n_in_tour * (n_cities - n_in_tour)

        results = {}
        original_cities = self._scenario.getCities()
        ncities = len(original_cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while not foundTour and time.time() - start_time < time_allowance and count < ncities:
            cities = original_cities.copy()
            startCity = cities[count]
            route = []

            route.append(startCity)
            cities.remove(startCity)

            for i in range(ncities - 1):
                city_added = self.addClosestCityToRoute(route, cities)
                cities.remove(city_added)

            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    def addClosestCityToRoute(self, route, cities):
        min_dist = math.inf
        city_to_add = None
        min_idx = 0

        for i, c1 in enumerate(route):
            closest = self.findClosestCity(c1, cities)
            dist = c1.costTo(closest)
            if dist < min_dist:
                min_dist = dist
                city_to_add = closest
                min_idx = i

        # add city_to_add to route between the two closest cities in the route
        # we already know the closest city, so the second closest will be its neighbor
        a_dist = city_to_add.costTo(route[min_idx - 1])
        b_dist = city_to_add.costTo(route[(min_idx + 1) % len(route)])
        if a_dist < b_dist:
            route.insert(min_idx, city_to_add)
        else:
            route.insert(min_idx + 1, city_to_add)

        return city_to_add

    def cheapest_insertion(self, time_allowance=60.0, swap=False):
        '''Find the point that increases the length of your tour by the least amount.'''
        results = {}
        original_cities = self._scenario.getCities()
        ncities = len(original_cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while not foundTour and time.time() - start_time < time_allowance and count < ncities:
            cities = original_cities.copy()
            startCity = cities[count]
            route = []

            route.append(startCity)
            cities.remove(startCity)
            city_added = self.addClosestCityToRoute(route, cities)
            cities.remove(city_added)  # initial adding of second city

            for i in range(ncities - 2):
                city_added = self.addCheapestCityToRoute(route, cities)
                cities.remove(city_added)

            if swap:
                route = self.swap_edges(route)

            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    def swap_edges(self, original_route):
        num = len(original_route)
        current_route = original_route.copy()
        improving = True
        while improving:
            bssf = TSPSolution(current_route)
            current_cost = bssf.cost
            for i in range(1, num + 1):
                for j in range(i + 1, num + 1):
                    improved_route = self.opt_swap(current_route, (i % num), (j % num))
                    improved_cost = TSPSolution(improved_route).cost
                    if improved_cost < current_cost:
                        current_route = improved_route
                        current_cost = improved_cost
                        improving = True
                        break
                    else:
                        improving = False

                if improving:
                    break

        return current_route

    def opt_swap(self, route, i, j):
        reversed_bit = route[i:j + 1]
        reversed_bit.reverse()
        return route[:i] + reversed_bit + route[j + 1:]

    def addCheapestCityToRoute(self, route, cities):
        # for each city in route, fin
        min_cost_increase = math.inf
        city_to_add = None
        min_idx = 0

        for i in range(len(route)):
            c1 = route[i]
            c3 = route[(i + 1) % len(route)]
            c2, cost_increase = self.findCheapestInsertion(c1, c3, cities)
            if cost_increase <= min_cost_increase:
                min_cost_increase = cost_increase
                city_to_add = c2
                min_idx = i

        route.insert(min_idx, city_to_add)

        return city_to_add

    def findCheapestInsertion(self, c1, c3, cities):
        city_to_add = None
        min_cost_increase = math.inf
        current_cost = c1.costTo(c3)

        for c2 in cities:
            cost_increase = c1.costTo(c2) + c2.costTo(c3) - current_cost
            if (cost_increase <= min_cost_increase):
                min_cost_increase = cost_increase
                city_to_add = c2

        return city_to_add, min_cost_increase

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

   #finds the best solution using a branch and bound method
    #If the optimal solution is not found within the time limit it simply returns the best solution so far
    #Worst case time complexity O(n^3*n!)
    #worst case space complexity of O(n^2*n!)
    def branchAndBound( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        numCities = len(cities)
        bssf = self.greedy(time_allowance)['soln']
        bssf_cost = bssf.cost
        route = []
        reducedCostMatrix,lowestCost,startCity=self.findReducedCostMatrix(cities)
        route.append(cities[startCity])
        pathIndexes = [0]
        state1 = State(reducedCostMatrix, lowestCost, 0, route ,startCity, False, pathIndexes)
        statesQueue = []
        statesQueue.append((state1.keyValue,state1))
        heapq.heapify(statesQueue)
        numSolutions = 0
        numPruned = 0
        numStates = 1
        maxQueueSize = 1

        start_time = time.time()

        #worst case scenario is a O(n^3*n!) if not states are pruned, with pruning it is much less
        #worst case space complexity of O(n^2*n!)
        while statesQueue and time.time()-start_time < time_allowance:
            key,currentState = heapq.heappop(statesQueue)
            if currentState.cost >= bssf_cost:
                numPruned = numPruned + 1
            else:
                originCity = currentState.lastCityVisited

                #O(n^3) becuase find ChildState is o(n^2) however that function will not actually run n times thanks to the first if statement
                #Space complexity O(n^2)
                for destCity in range(numCities):
                    if currentState.matrix[originCity][destCity] != math.inf and destCity not in currentState.currentPathIndexs:
                        childState = self.findChildState(currentState,originCity,destCity,cities)
                        numStates = numStates + 1
                        if childState.isSolution:
                            numSolutions = numSolutions + 1
                            solution=TSPSolution(childState.currentPath)
                            solution_cost = solution.cost
                            if(solution_cost < bssf_cost):
                                bssf_cost = solution_cost
                                bssf = solution
                            else:
                                numPruned = numPruned + 1
                        elif childState.cost <= bssf_cost :
                            heapq.heappush(statesQueue, (childState.keyValue, childState))
                            if len(statesQueue) > maxQueueSize:
                                maxQueueSize = len(statesQueue)
                        else:
                            numPruned = numPruned + 1

        end_time = time.time()
        results['cost'] = bssf_cost
        results['time'] = end_time - start_time
        results['count'] = numSolutions
        results['soln'] = bssf
        results['max'] = maxQueueSize
        results['total'] = numStates
        results['pruned'] = numPruned

        return results

        #finds and returns the child state
    #The find reduced matrix function which is called dominates the time and space complexity and results in a O(n^2)
    def findChildState(self, parentState, orginCity, destCity, cities):
        
        matrix = copy.deepcopy(parentState.matrix)
        numCities = len(matrix)
        costToCity = matrix[orginCity][destCity]
        
        #Time complexity of O(n)
        for i in range(numCities):
            matrix[orginCity][i] = math.inf
            matrix[i][destCity] = math.inf
        matrix[destCity][orginCity] = math.inf

        currentPath = copy.deepcopy(parentState.currentPath)
        currentPath.append(cities[destCity])
        pathIndexs = copy.deepcopy(parentState.currentPathIndexs)
        pathIndexs.append(destCity)


        reducedMatrix, costIncrease = self.reduceMatrix(matrix, pathIndexs)
        cost = parentState.cost + costIncrease + costToCity
        depth = parentState.depth + 1
        
        lastCityVisited = destCity
        if len(currentPath) == numCities:
            isSolution = True
        else:
            isSolution = False

        childState = State(reducedMatrix, cost, depth, currentPath, lastCityVisited, isSolution, pathIndexs)

        return childState

    #find the initial reduced cost matrix for the parent state from the list of cities
    #tima and space complexity of O(n^2)
    def findReducedCostMatrix(self, cities):
        numCities = len(cities)
        startMatrix = [ [ 0 for j in range(numCities) ] for i in range(numCities) ]
        startCityIndex = 0
        
        for i in range(numCities):
                for j in range(numCities):
                    startMatrix[i][j] = cities[i].costTo(cities[j])
        
        pathIndexs = []
        reducedMatrix, costIncrease = self.reduceMatrix(startMatrix, pathIndexs)

        return reducedMatrix, costIncrease, startCityIndex

    #Reduces the given matrix
    #Time complexity of O(n^2), space complexity of O(n^2)
    def reduceMatrix(self, matrix, path):
        matrix = np.array(matrix)
        numCities = len(matrix)
        cost = 0

        for i in range(numCities):
            if(i not in path[:-1]):
                row = matrix[i,:]
                minValue = np.min(row)
                if(minValue != math.inf):
                    cost = cost + minValue
                    for j in range(numCities):
                        matrix[i][j] = matrix[i][j] - minValue

        for j in range(numCities):
            if(j not in path[1:]):
                col = matrix[:,j]
                minValue = np.min(col)
                if(minValue != math.inf):
                    cost = cost + minValue
                    for i in range(numCities):
                        matrix[i][j] = matrix[i][j] - minValue

        return matrix, cost

        


    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        return self.cheapest_insertion(time_allowance, swap=True)


#Class which holds each state's minimum matrix and other useful information
# Each function has a O(1)
# The matrix dominates the space complexity with a O(n^2), all other variables have O(n) or O(1)
class State:
    def __init__( self, matrix, cost, depth, currentPath, lastCityVisited, isSolution, currentPathIndexs):
        self.matrix = matrix
        self.depth = depth
        self.cost = cost
        self.currentPath = currentPath
        self.lastCityVisited = lastCityVisited
        self.isSolution = isSolution
        self.keyValue = self.cost - self.depth*3*len(matrix)
        self.currentPathIndexs = currentPathIndexs

    def __lt__(self, other):
        return ((self.keyValue) < (other.keyValue))