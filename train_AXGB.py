from AXGB import AdaptiveXGBoostClassifier
import pickle
import os 

class AXGBTrainer:
    def __init__(self, data_dir, W, benign_label=0, attack_label=1, random_state=42):
        self.data_path = data_dir
        self.W = W
        self.benign_label = benign_label
        self.attack_label = attack_label
        self.random_state = random_state
        self.model = AdaptiveXGBoostClassifier(
            update_strategy='push',
            n_estimators=3,
            learning_rate=0.3,
            max_depth=20,
            max_window_size=W/10,
            detect_drift=True
        )

    def train_eval_axgb(self, X_train, y_train):
        self.model.partial_fit(X=X_train, y=y_train)

    def retrain_model(self, X_retrain, y_retrain):
        self.model.partial_fit(X=X_retrain, y=y_retrain)

    def save_model(self, w_index, folder_path):
        full_folder = f"{folder_path}/models/{w_index}"
        os.makedirs(full_folder, exist_ok=True)
        model_path = os.path.join(full_folder, f"axgb_model_{w_index}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, w_index, folder_path):
        full_folder = f"{folder_path}/models/{w_index}"
        model_filename = f"axgb_model_{w_index}.pkl"
        model_path = os.path.join(full_folder, model_filename)
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data
        return self.model
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        raise NotImplementedError("predict_proba is not implemented for this method.")