import joblib
import os

model_path = os.path.join("models", "traffic_congestion_model.pkl")

model = joblib.load(model_path)

print("MODEL TYPE:", type(model))
print("MODEL CLASSES:", model.classes_)
print("FEATURE NAMES:", model.feature_names_in_)

print("\nMODEL STEPS:")
for name, step in model.named_steps.items():
    print(name, ":", type(step))
