from HAT import HoeffdingAdaptiveTreeModel
import pickle
import os 

class HatTrainer:
    def __init__(self, data_dir, W, benign_label=0, attack_label=1, random_state=42):
        self.data_path = data_dir
        self.W = W
        self.benign_label = benign_label
        self.attack_label = attack_label
        self.random_state = random_state
        self.model = HoeffdingAdaptiveTreeModel(random_state=random_state, W=W)

    def train_eval_model(self, X_train, y_train):
        self.model.train(X_train=X_train, y_train=y_train)

    def retrain_model(self, X_retrain, y_retrain):
        self.model.retrain(X_train=X_retrain, y_train=y_retrain)

    def save_model(self, w_index, folder_path):
        full_folder = f"{folder_path}/models/{w_index}"
        os.makedirs(full_folder, exist_ok=True)
        model_path = os.path.join(full_folder, f"hat_model_{w_index}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, w_index, folder_path):
        full_folder = f"{folder_path}/models/{w_index}"
        model_filename = f"hat_model_{w_index}.pkl"
        model_path = os.path.join(full_folder, model_filename)
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data
        return self.model
    
    def predict(self, X):
        y_pred = self.model.predict_fn(X)
        return y_pred
    
    def predict_proba(self, X):
        return self.model.predict_proba_fn(X=X)