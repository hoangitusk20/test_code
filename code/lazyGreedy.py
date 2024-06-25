import heapq
import numpy as np

############????
def _heappush_max(heap,item): # Thêm 1 phần tử mới vào heap
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)

def _heappop_max(heap): # Lấy phần tử có giá trị lớn nhất trong max heap 
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap,0)
        return returnitem
    return lastelt


"""
F: Hàm mục tiêu của thuật toán
V: một dãy các số từ 1 đến số lượng điểm trong không gian
B: Số điểm cần chọn lựa
"""
def lazy_greedy_heap(F, V, B, c):
    curVal = 0 # Giá trị hiện tại của hàm mục tiêu (không dùng)
    sset = [] # Danh sách các điểm được chọn
    vals = [] # cả curVal và vals hình như không dùng tới

    order = [] # Heap của chúng ta
    heapq._heapify_max(order)

    # Khởi tạo heap là tổng khoảng cách từ 1 điểm đến tất cả các điểm còn lại
    for index in V:
        _heappush_max(order, (F.inc(sset, index), index))
    for i in range(B):
        el =_heappop_max(order)
        sset.append(el[1])

    coresize = len(sset)
    sub_coresize = int(len(V) * c)

    return sset[-sub_coresize:],vals

    while order and len(sset) < B:
        if F.curVal == len(F.D): #????
            #all points covered
            break
        el = _heappop_max(order) # Phần tử cung cấp giá trị lớn nhất
        improv = F.inc(sset,el[1]) # 
        if improv > 0 : #Liệu có thể < 0 được không?
            if not order:
                curVal = F.add(sset, el[1], improv) #curVal hình như cũng không dùng vào đâu
                sset.append(el[1])
                vals.append(curVal) #Cái vals này không có dùng??
            else: 
                top = _heappop_max(order)
                if improv > top[0]:#Tại sao có thể xảy ra trường hợp này?
                    curVal = F.add(sset, el[1], improv)
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv,el[1]))
                _heappush_max(order,top)

    return sset,vals

def algo1(B,D):
    k = B
    sorted_matrix = np.sort(D, axis=1)
    k_smallest_elements = sorted_matrix[:, :k]
    sum_k_smallest = np.sum(k_smallest_elements, axis=1)
    sset = np.argsort(sum_k_smallest)[:k]
    return sset

import numpy as np

def k_medoids(D, k, max_iter=300):
    """
    K-medoids algorithm to find k medoids from a distance matrix D.
    
    Parameters:
    D (numpy array): Distance matrix of shape (n, n)
    k (int): Number of medoids to find
    max_iter (int): Maximum number of iterations
    
    Returns:
    medoids (list): Indices of the k medoids
    clusters (list): List of clusters with indices of the points
    """
    
    n = D.shape[0]
    
    # Initialize medoids randomly
    medoids = np.random.choice(n, k, replace=False)
    
    for iteration in range(max_iter):
        # Assign each point to the nearest medoid
        clusters = [[] for _ in range(k)]
        for i in range(n):
            distances_to_medoids = D[i, medoids]
            nearest_medoid_index = np.argmin(distances_to_medoids)
            clusters[nearest_medoid_index].append(i)
        
        new_medoids = np.zeros(k, dtype=int)
        
        # Update medoids
        for i in range(k):
            cluster = clusters[i]
            if len(cluster) == 0:  # handle empty clusters
                continue
            
            # Calculate the total distance from each point in the cluster to all other points in the cluster
            intra_cluster_distances = D[np.ix_(cluster, cluster)]
            total_distances = np.sum(intra_cluster_distances, axis=1)
            
            # Select the point with the minimum total distance as the new medoid
            new_medoid_index = np.argmin(total_distances)
            new_medoids[i] = cluster[new_medoid_index]
        
        # Check for convergence (if medoids do not change)
        if np.array_equal(medoids, new_medoids):
            break
        
        medoids = new_medoids
    
    return medoids#, clusters

# Example usage:
# Assume D is a precomputed distance matrix
# k = number of clusters
# D = np.array([...])
# k = 3
# medoids, clusters = k_medoids(D, k)
# print("Medoids:", medoids)
# print("Clusters:", clusters)
