import numpy as np


def generate_next_vehicle_random_pose(map_of_world):
    h, w = map_of_world.shape
    while True:
        next_x = np.random.randint(1, h-1)
        next_y = np.random.randint(1, w-1)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy:
                    continue
                elif map_of_world[next_x+dx][next_y+dy] == 1:
                    return [next_x, next_y]


def generate_vehicle_coverage_idx(vehicle_x, vehicle_y, cell_height, cell_width, vehicle_size):
    # vehicle_x = np.arange((vehicle_x-self.vehicle_size),
    #                       (vehicle_x+self.vehicle_size)+1, 1)
    # vehicle_y = np.arange((vehicle_y - self.vehicle_size),
    #                       (vehicle_y + self.vehicle_size)+1, 1)
    vehicle_x *= cell_height
    vehicle_y *= cell_width
    vehicle_x = np.arange(vehicle_x, vehicle_x + vehicle_size, 1)
    vehicle_y = np.arange(vehicle_y, vehicle_y + vehicle_size, 1)
    xx, yy = np.meshgrid(vehicle_x, vehicle_y, sparse=True)
    return xx, yy


# def check_collision(_map, loc_x, loc_y):
#     for dx in [-1, 0, 1]:
#         for dy in [-1, 0, 1]:
#             if dx == dy:
#                 continue
#             else:
#                 if _map[loc_x+dx][loc_y+dy] != 1:
#                     return True
#     return False

class metrics_np():
    def __init__(self, n_class=1,hist=None):
        if hist is None:
            self.hist = np.zeros((n_class,n_class))
        else:
            self.hist = hist
        self.n_class = n_class

    def _fast_hist(self,label_true,label_pred,n_class):
        mask = (label_true>=0) & (label_true<n_class) # to ignore void label
        self.hist = np.bincount( n_class * label_true[mask].astype(int)+label_pred[mask],minlength=n_class**2).reshape(n_class,n_class)
        return self.hist

    def update(self,x,y):
        self.hist += self._fast_hist(x.flatten(),y.flatten(),self.n_class)

    def update_hist(self,hist):
        self.hist += hist

    def get(self,kind="miou"):
        if kind == "accu":
            return np.diag(self.hist).sum() / (self.hist.sum()+1e-5) *100 # total pixel accuracy
        elif kind == "precision":
            return np.diag(self.hist) / (self.hist.sum(axis=0)+1e-5) *100
        elif kind == "recall":
            return np.diag(self.hist) / (self.hist.sum(axis=1)+1e-5) *100
        elif kind in ["freq","fiou","iou","miou"]:
            iou = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist)+1e-5)
            if kind == "iou": return iou*100
            miou = np.nanmean(iou)
            if kind == "miou": return miou*100

            freq = self.hist.sum(axis=1) / (self.hist.sum()+1e-5) # the frequency for each categorys
            if kind == "freq": return freq*100
            else: return (freq[freq>0]*iou[freq>0]).sum()*100
        elif kind in ["dice","mdice"]:
            dice = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0)+1e-5)
            if kind == "dice": return dice
            else: return np.nanmean(dice)*100
        elif kind == 'f-score':
            pre = np.diag(self.hist) / (self.hist.sum(axis=0)+1e-5)
            recall = np.diag(self.hist) / (self.hist.sum(axis=1)+1e-5)
            return 2*pre.mean()*recall.mean()/(recall.mean()+pre.mean())*100
        return None

    def get_all(self):
        metrics = {}
        metrics["accu"] = np.diag(self.hist).sum() / (self.hist.sum()+1e-5) # total pixel accuracy
        metrics["precision"] = np.diag(self.hist) / (self.hist.sum(axis=0)+1e-5) # pixel accuracys for each category, np.nan represent the corresponding category not exists
        metrics["recall"] = np.diag(self.hist) / (self.hist.sum(axis=1)+1e-5) # pixel accuracys for each category, np.nan represent the corresponding category not exists
        metrics["iou"] = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0)-np.diag(self.hist)+1e-5)
        metrics["miou"] = np.nanmean(metrics["iou"])
        metrics["freq"] = self.hist.sum(axis=1) / (self.hist.sum()+1e-5) # the frequency for each categorys
        metrics["fiou"] = (metrics["freq"][metrics["freq"]>0]*metrics["iou"][metrics["freq"]>0]).sum()
        metrics['f-score'] = 2*metrics["precision"].mean()*metrics["recall"].mean()/(metrics["recall"].mean()+metrics["precision"].mean())
        for i in metrics.keys():
            metrics[i]*=100
        return metrics