import heapq
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
def lazy_greedy_heap(F, V, B):
    curVal = 0 # Giá trị hiện tại của hàm mục tiêu (không dùng)
    sset = [] # Danh sách các điểm được chọn
    vals = [] # cả curVal và vals hình như không dùng tới

    order = [] # Heap của chúng ta
    heapq._heapify_max(order)

    # Khởi tạo heap là tổng khoảng cách từ 1 điểm đến tất cả các điểm còn lại
    for index in V:
        _heappush_max(order, (F.inc(sset, index), index))

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

