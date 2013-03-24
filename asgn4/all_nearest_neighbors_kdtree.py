# Created on Mar 15, 2013

from heapq import *

class kdtree:

    # adapted from wikipedia, with slight modification
    def __init__(self, point_list, depth=0):
        if not point_list:
            self.location = None
            self.axis = 0
            self.left_child = None
            self.right_child = None
        else:
            # Select axis based on depth so that axis cycles through all valid values
            # self.axis = self.calculate_optimal_axis(point_list)
            self.axis = depth % len(point_list[0])
    
            # Sort point list and choose median as pivot element
            point_list.sort(key=lambda point: point[self.axis])
            median = len(point_list) // 2  # choose median
    
            # Create node and construct subtrees
            self.location = point_list[median]
            self.left_child = kdtree(point_list[:median], depth + 1)
            self.right_child = kdtree(point_list[median + 1:], depth + 1)

    # calculate optimal axis as the axis along which variance is maximum (Ref: http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-1549.pdf)
    def calculate_optimal_axis(self, point_list):
        def variance(l, dim):
            mean = 0.
            for elem in l:
                mean += elem[dim]
            mean = mean/len(l)        
            variance = 0.
            for elem in l:
                variance += (elem[dim]-mean)**2
            return variance

        optimal_dim = -1
        max_var = 0.0
        for dim in range(len(point_list[0])):
            var = variance(point_list, dim)
            if var > max_var:
                optimal_dim = dim
                max_var = var
        return optimal_dim

    def __repr__(self):
        if self.location:
            return "(%d, %s, %s, %s)" % (self.axis, repr(self.location), repr(self.left_child), repr(self.right_child))
        else:
            return "None"
        
    # keep a max-heap of k best-distances found so far. Evict the max if you find a closer point
    def knn(self, k, query, knearest=[]):
        if self.location is None:
            return knearest
 
        cur_dist = self.sqrd_distance(self.location, query)
        if len(knearest) < k:
            heappush(knearest, (-cur_dist, self.location))
        elif cur_dist < -knearest[0][0]:
            heapreplace(knearest, (-cur_dist, self.location))

        nc = self.near_child(query)
        knearest = nc.knn(k, query, knearest)
 
        max_dist = -knearest[0][0]
        if len(knearest) < k or self.axis_sqrd_distance(query) < max_dist:
            fc = self.far_child(query)
            knearest = fc.knn(k, query, knearest)
 
        return knearest

    def near_child(self, cur_point):
        if cur_point[self.axis] < self.location[self.axis]:
            return self.left_child
        else:
            return self.right_child
 
    def far_child(self, cur_point):
        if cur_point[self.axis] < self.location[self.axis]:
            return self.right_child
        else:
            return self.left_child 

    def axis_sqrd_distance(self, point):
        # axis_point = list(point)
        # axis_point[self.axis] = self.location[self.axis]
        # return self.sqrd_distance(tuple(axis_point), point)
        return (self.location[self.axis] - point[self.axis])**2

    def sqrd_distance(self, p1, p2):        
        dist = 0.
        for i in range(len(p1)):
            dist += (p1[i]-p2[i])**2
        #print "printing distance..."
        #print p1, p2, dist
        return dist

"""
num_lines_to_read = 10 #1000
input_filename = 'input.dat'
txt = open(input_filename, 'r').readlines()
points = [eval(txt[i]) for i in range(num_lines_to_read)]        
"""
# points = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
# points = [(0, 1), (1,0), (3,2), (2,3)]
points = [(0.2,0), (0,0.2), (0.8,0), (0,0.8)]
query_points = list(points)
kdt = kdtree(points)
# print kdt
k = 2 # one of the points will be the point itself
for i in range(len(query_points)):
    print query_points[i], kdt.knn(k, query_points[i], [])

