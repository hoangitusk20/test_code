import numpy as np
"""
V: một dãy các số từ 1 đến số lượng điểm trong không gian
D: ma trận khoảng cách giữa các điểm trong không gian gradient
fnpy: Đường dẫn tới tệp chứa ma trận khoảng cách 
(có thể truyền trực tiếp ma trận khoảng cách vào hàm hoặc qua tệp)
"""
####
class FacilityLocationCIFAR:
    def __init__(self, V, D= None, fnpy = None):

        # Đọc vào ma trận khoảng cách
        if D is not None:
            self.D = D
        else:
            self.D = np.load(fnpy)

        self.D *=-1 #---Vì mục tiêu là tìm khoảng cách nhỏ nhất, với khoảng cách nhỏ, hàm mục tiêu cung cấp giá trị lớn, nên cần đảo dấu
        self.D -= self.D.min() # Để đảm bảo tất cả giá trị đều lớn hơn 0
        #( Bây giờ giá trị khoảng cách lớn nhất sẽ là 0, giá trị khoảng cách nhỏ nhất tương ứng với số lớn nhất trong ma trận)
        self.V = V
        self.curVal = 0 # ------Giá trị hiện tại của hàm mục tiêu
        self.gains = [] 
        self.cur_max = np.zeros_like(self.D[0]) #Khoảng cách từ cụm đến các điểm còn lại

    def inc(self, sset, ndx): # -----Tính toán giá trị mà hàm mục tiêu sẽ tăng nếu thêm điểm ndx vào sset
        ####
        if len(sset + [ndx]) > 1:#????
            new_dists = np.stack([self.cur_max, self.D[ndx]], axis = 0)
            return new_dists.max(axis = 0).sum() 
        else:
            return self.D[sset + [ndx]].sum()#???
    
    def add(self, sset, ndx, delta):# ---Cập nhật giá trị hàm mục tiêu sau khi thêm ndx
        self.curVal +=delta
        self.gains += delta # gains này hình như không dùng?
        self.cur_max = np.stack([self.cur_max, self.D[ndx]],axis = 0).max(axis = 0)
        return self.curVal
