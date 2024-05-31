import torch
import torchvision
from PIL import Image
import torchvision.transforms as transform
import numpy as np
import copy
import warnings

class MISLABELCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, mislabel_type ='agnostic', mislabel_ratio = 0.5, rand_number = 0, train = True, 
                  transform = None, target_transform = None, download = False):
        super(MISLABELCIFAR10,self).__init__(root, train, transform,target_transform,download)
        np.random.seed(rand_number)
        self.gen_mislabeled_data(mislabel_type = mislabel_type, mislabel_ratio = mislabel_ratio)

    def gen_mislabeled_data(self, mislabel_type, mislabel_ratio): # Để tạo dữ liệu thành nhiễu với tỷ lệ nhiễu cho trước
        new_targets = []
        num_cls = np.max(self.targets) + 1 # Số lượng lớp cần phân loại, đối với CIFAR10 là 10

        if mislabel_type == 'agnostic':# Nhãn nhiễu được tạo một cách ngẫu nhiên không biết trước
            for _, target in enumerate(self.targets): 
                if np.random.rand() < mislabel_ratio:
                    new_target = target
                    while new_target == target:
                        new_target = np.random.randint(num_cls)
                    new_targets.append(new_target)
                else:
                    new_targets.append(target)
        
        elif mislabel_type =='asym': # Nhãn nhiễu kiểu, ví dụ nhãn nhiễu là a thì nhãn thật luôn là b. (Dòng 41)
            ordered_list = np.arange(num_cls)
            while True:
                permu_list = np.random.permutation(num_cls) # Tạo ra định nghĩa cho nhãn nhiễu, nhãn nhiễu a thì nhãn thật sẽ là b.
                if np.any(ordered_list == permu_list):
                    continue
                else:
                    break

            for _, target in enumerate(self.targets):
                if np.random.rand() < mislabel_ratio:
                    new_target = permu_list[target]
                    new_targets.append(new_target)
                else:
                    new_targets.append(target)

        else:
            warnings.warn("Noise type is not listed")

        self.real_targets = self.targets #real_targets là nhãn thật của dữ liệu
        self.targets = new_targets # targets là nhãn của dữ liệu sau khi sinh nhiễu
        self.whole_data = self.data.copy() # whole_data là bản deep copy của dữ liệu
        self.whole_targets = copy.deepcopy(self.targets) #whole_targets là bản deep copy của targets
        self.whole_real_targets = copy.deepcopy(self.real_targets) # whole_real_targets bản deep copy của nhãn thật
        # Cần tạo các bản deep copy để lỡ như nếu có thay đổi giá trị thì vẫn có thể thay đổi về giá trị cũ

    def switch_data(self): # Sau 1 số phép biển đổi có thể dữ liệu đã bị thay đổi, hàm này để chuyển dữ liệu lại như ban đầu
        self.data = self.whole_data
        self.targets = self. whole_targets
        self.real_targets = self.whole_real_targets

    def adjust_base_indx_temp(self, idx): #Thay đổi data và target mới (Lựa chọn tập con) dựa trên idx, Vẫn có thể khôi phục dữ liệu

        # Cần lấy ra new_data, new_targets và new_real_targets theo idx
        new_data = self.whole_data[idx,...]
        target_np = np.array(self.whole_targets)
        new_targets = target_np[idx].tolist()
        real_targets_np = np.array(self.whole_real_targets)
        new_real_targets = real_targets_np[idx].tolist()

        # Chọn ra data và target mới dựa trên index
        self.data = new_data
        self.targets = new_targets
        self.real_targets = new_real_targets
        
    def adjust_base_indx_perma(self, idx):#Thay đổi data và target mới (Lựa chọn tập con) dựa trên idx, không khôi phục được dữ liệu

        # Cần lấy ra new_data, new_targets và new_real_targets theo idx
        new_data = self.whole_data[idx,...]
        target_np = np.array(self.whole_targets)
        new_targets = target_np[idx].tolist()
        real_targets_np = np.array(self.whole_real_targets)
        new_real_targets = real_targets_np[idx].tolist()

        # Chọn ra data và target mới dựa trên index
        self.data = new_data
        self.targets = new_targets
        self.real_targets = new_real_targets

        # Đồng thời thay đổi data và target của cả bản deep copy
        self.whole_data = self.data.copy()
        self.whole_targets = copy.deepcopy(self.targets)
        self.whole_real_targets = copy.deepcopy(self.real_targets)

    def estimate_label_acc(self): # Ước lượng độ chính xác của nhãn 
        targets_np = np.array(self.targets)
        real_tagets_np = np.array(self.real_targets)
        return np.sum((targets_np == real_tagets_np)) / len(targets_np)
    
    ####

    def print_class_dis(self):
        targets_np = np.array(self.targets)
        unique_values, counts = np.unique(targets_np, return_counts=True)

        # Combine the results into a dictionary
        count_dict = dict(zip(unique_values, counts))

        print(count_dict)

    def fetch(self, targets): # Chọn ra data từ target, sample đúng với target nhưng lấy ngẫu nhiên (Dòng 109)
        whole_targets_np = np.array(self.whole_targets)
        uniq_targets = np.unique(whole_targets_np)
        idx_dict = {}
        
        for uniq_target in uniq_targets:
            idx_dict[uniq_target] = np.where(whole_targets_np == uniq_target)[0] #Lọc ra các index của targets có nhãn là uniq_target

        idx_list = [] 
        for target in targets:
            idx_list.append(np.random.choice(idx_dict[target.item()],1)) # Chọn ra 1 index ngẫu nhiên có nhãn là target

        idx_list = np.array(idx_list).flatten()
        imgs = []
        for idx in idx_list:
            img = self.whole_data[idx]
            img = Image.fromarray(img)
            img = self.transform(img)
            imgs.append(img[None,...])
        train_data = torch.cat(imgs, dim = 0)
        return train_data
    
    def __getitem__(self,index): # Lấy dữ liệu từ một index cho trước, cái hàm này hình như không dùng
        img, target, real_target = self.data[index], self.targets[index], self.real_targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = target.transform(target)

        return img, target, real_target , index 
    
class MISLABELCIFAR100(MISLABELCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    file_name = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }