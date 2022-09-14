# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    
    # END OF ROUND 1
    
    #------------- ROUND 2 ---------------------------
    start = time.time()
    elems = coreset.collect()
    end = time.time()
    time_round_1 = (end -start) * 1000

    start = time.time()
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
    
    S = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2.)
    end = time.time()      
    time_round_2 = (end - start) * 1000.
    
    print("Time Round 1:", time_round_1, "ms")
    print("Time Round 2:", time_round_2, "ms")


    return S
     
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P,W,k,z,alpha):
    P = np.array(P)
    W = np.array(W)
    distances_matrix = np.zeros(shape=(len(P),len(P)))

    for i in range(len(P)):
        for j in range(len(P)):
            distances_matrix[i][j] = euclidean(P[i], P[j])
    
    min_dist = np.min(distances_matrix[:k+z+1,:k+z+1]+np.diag(np.inf*np.ones(z+k+1)))
    
    r_min = min_dist/2.0         
    r = r_min  
    
    print("Initial guess =", r)

    n_guess = 1
    w = np.array(W)

    while(True):
        Z = np.ones(len(P))
        S = []
        Wz = sum(W)
        
        while ((len(S)<k) and (Wz>0)):
            
            max = 0
            for i in range(len(P)):
                
                Bz12 = distances_matrix[i, :]<=(1+2*alpha)*r
                ball_weight = np.sum(w[np.logical_and(Z==1,Bz12)])                
                
                if ball_weight > max:
                    max = ball_weight
                    indexNewCenter = i           
            S.append(P[indexNewCenter])    
            
            Bz34 = distances_matrix[indexNewCenter, :]<=(3+4*alpha)*r 
            
            Wz -= np.sum(w[np.logical_and(Z==1,Bz34)])       
            Z[np.logical_and(Z==1,Bz34)] = 0
            
        if Wz<=z:
            print("Final guess =", r)
            print("number of guesses =", n_guess)
            return S
        else:
            r = 2*r
            n_guess += 1


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# def computeObjective(points, centers, z):
#     vectMinDistFromK = np.zeros(len(points))  
#     points = np.array(points)
#     centers = np.array(centers)
#     vectMinDistFromK = np.zeros(len(points))  
#     for i in range(len(points)):
#         min = np.inf
#         for j in range(len(centers)):
#             min = np.minimum(min, euclidean(points[i],centers[j]) )
#         vectMinDistFromK[i] = min
#     result = np.sort(vectMinDistFromK, axis=None)
#     return result[-(z+1)]



def computeObjective(points, solution, z):
    #iterator for iterating in each partition
    RDD = points.mapPartitions(lambda iterator: find_max_Z_plus_one_points(iterator, z+1, solution))
    elem = RDD.collect()

    farthest_points = np.array(elem)

    for i in range(z):
        #remove outliers by putting it = 0 (more efficent)
        farthest_points[np.argmax(a=farthest_points)] = 0
    return np.sqrt(np.max(farthest_points))   



def find_max_Z_plus_one_points(iter, z_plus_1, solution) :

    points_part = np.array(list(iter))  # points in the current partition
    solution = np.array(solution) # my k centers
    n, k = points_part.shape[0], solution.shape[0]  # n: number of point for each partition  # k: number of k centers 

    matrix_dist_from_centers = np.zeros(shape=(n,k), dtype= float)
    for i in range(n): # for each point in the partition
        np.sum(np.square(np.subtract(points_part[i], solution)), out=matrix_dist_from_centers[i], axis=1, dtype=float)
    # consider the distance between point and their closer center
    min_dist = matrix_dist_from_centers.min(axis=1)   


    max_dists_for_part = min_dist[min_dist.argsort()[n-z_plus_1:]]  # takes biggest after sorting and removing z+1 
    return list(max_dists_for_part)

# Just start the main program
if __name__ == "__main__":
    main()

