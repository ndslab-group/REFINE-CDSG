from sklearn.ensemble import RandomForestClassifier
import pickle
import os 

class RFTrainer:
    def __init__(self, data_dir, W, benign_label=0, attack_label=1, random_state=42):
        self.data_path = data_dir
        self.W = W
        self.benign_label = benign_label
        self.attack_label = attack_label
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy', min_samples_split=10, min_samples_leaf=10)

    def train_eval_rf(self, X_train, y_train):
        self.model.fit(X=X_train, y=y_train)

    def save_model(self, w_index, folder_path):
        # folder path contiene 
        full_folder = f"{folder_path}/models/{w_index}"
        os.makedirs(full_folder, exist_ok=True)
        model_path = os.path.join(full_folder, f"rf_model_{w_index}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, w_index, folder_path):
        full_folder = f"{folder_path}/models/{w_index}"
        model_filename = f"rf_model_{w_index}.pkl"
        model_path = os.path.join(full_folder, model_filename)
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data
        return self.model
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)