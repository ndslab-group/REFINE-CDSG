from tqdm import tqdm
from skmultiflow.drift_detection import KSWIN

class KSWIN_cdd:
    def __init__(self,W=50):
        self.alpha_kswin = 0.005   
        self.win_size = 2 * W 
        self.stat_size = W
        print(f"Win size: {self.win_size}, Statistical Size: {self.stat_size}")
        self.kswin = KSWIN(alpha=self.alpha_kswin, window_size=self.win_size, stat_size=self.stat_size) 

    def detected_change_error(self, y_window, y_pred_window):
        drift_detected_kswin = False
        for true, pred in zip(y_window, y_pred_window):
            error = int(true != pred)  # 1 = error, 0 = correct
            self.kswin.add_element(error)
            if self.kswin.detected_change():
                drift_detected_kswin = True
        return drift_detected_kswin
    
    def detected_change(self, window):
        drift_detected_kswin = False
        for sample in window:
            self.kswin.add_element(sample)
            if self.kswin.detected_change():
                drift_detected_kswin = True
        return drift_detected_kswin