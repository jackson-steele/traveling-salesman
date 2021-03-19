#/usr/bin/python3

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



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
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
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
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

	def greedy( self,time_allowance=60.0 ):
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

			for i in range(ncities - 1):
				currentCity = self.findClosestCity(currentCity, cities)
				route.append(currentCity)
				cities.remove(currentCity)

			# route.append(startCity)

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


	def nearest_insertion_2(self, time_allowance=60.0 ):
		'''Tries all cities as a starting point and returns best route found'''
		#start with a random city, add to "tour"
		#IN LOOP: find nearest city outside tour to a city in the tour, add
		#Complexity: n cities * n_in_tour * (n_cities - n_in_tour)

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
			currentCity = startCity

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


	def nearest_insertion_1(self, time_allowance=60.0 ):
		'''Regular nearest insertion, returns when solution is found'''
		#start with a random city, add to "tour"
		#IN LOOP: find nearest city outside tour to a city in the tour, add
		#Complexity: n cities * n_in_tour * (n_cities - n_in_tour)

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


	def nearest_insertion_3(self, time_allowance=60.0 ):
		'''Choose one of 3 closest cities (randomly) to add to route, to try to minimize dead ends'''
		#start with a random city, add to "tour"
		#IN LOOP: find nearest city outside tour to a city in the tour, add
		#Complexity: n cities * n_in_tour * (n_cities - n_in_tour)

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

			for i in range(ncities - 1):
				city_added = self.addOneOfClosestCitiesToRoute(route, cities)
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

	def addOneOfClosestCitiesToRoute(self, route, cities):
		# for each city in route, fin
		# min_dist = math.inf
		# min_idx = 0
		min_distances = [math.inf, math.inf, math.inf]
		indexes = [0,0,0]
		city_to_add = None

		for i, c1 in enumerate(route):
			closest = self.findClosestCity(c1, cities)
			dist = c1.costTo(closest)

			if dist < min_distances[0]:
				min_distances.insert(0,dist)
				min_distances.pop()
				indexes.insert(0, i)
				indexes.pop()
			elif dist < min_distances[1]:
				min_distances.insert(1, dist)
				min_distances.pop()
				indexes.insert(1, i)
				indexes.pop()
			elif dist < min_distances[2]:
				min_distances.insert(2, dist)
				min_distances.pop()
				indexes.insert(2, i)
				indexes.pop()

			# if dist < min_dist:
			# 	min_dist = dist
			# 	city_to_add = closest
			# 	min_idx = i


		rand = random.randint(0,2)
		min_idx = indexes[rand]
		city_to_add = self.findClosestCity(route[min_idx], cities)

		#add to route
		#compare distance of city_to_add to the before and after cities
		a_dist = city_to_add.costTo(route[min_idx - 1])
		b_dist = city_to_add.costTo(route[(min_idx + 1) % len(route)])
		if a_dist < b_dist:
			route.insert(min_idx, city_to_add)
		else:
			route.insert(min_idx+1, city_to_add)

		return city_to_add

	def addClosestCityToRoute(self, route, cities):
		# for each city in route, fin
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

		#add to route
		#compare distance of city_to_add to the before and after cities
		a_dist = city_to_add.costTo(route[min_idx - 1])
		b_dist = city_to_add.costTo(route[(min_idx + 1) % len(route)])
		if a_dist < b_dist:
			route.insert(min_idx, city_to_add)
		else:
			route.insert(min_idx+1, city_to_add)

		return city_to_add

	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		return self.nearest_insertion_3(time_allowance)



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy(self, time_allowance=60.0 ):
		return self.nearest_insertion_1(time_allowance)
		



