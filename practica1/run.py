# Search methods

import search

ab = search.GPSProblem('S', 'L', search.romania)

print ("\nMetodo de busqueda en anchura")
print search.breadth_first_graph_search(ab)[0].path()
print ("Numero de nodos expandidos en anchura")
print search.breadth_first_graph_search(ab)[1]

print ("\nMetodo de busqueda en profundidad")
print search.depth_first_graph_search(ab)[0].path()
print ("Numero de nodos expandidos en profundidad")
print search.depth_first_graph_search(ab)[1]

#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()

print ("\nMetodo de busqueda sin subestimacion: [SIN HEURISTICA]")
print search.resolutionSinSubestimacion(ab)[0].path()
print ("Numero de nodos expandidos sin subestimacion: ")
print search.resolutionSinSubestimacion(ab)[1]

print ("\nMetodo de busqueda con subestimacion: [CON HEURISTICA]")
print search.resolutionConSubestimacion(ab)[0].path()
print ("Numero de nodos expandidos con subestimacion: ")
print search.resolutionConSubestimacion(ab)[1]


#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
