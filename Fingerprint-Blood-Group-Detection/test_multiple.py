import os
import subprocess

# List of fingerprint images from the dataset (replace with your own paths)
images = [
    "dataset/dataset_blood_group/A+/cluster_0_1001.BMP",
    "dataset/dataset_blood_group/B+/cluster_2_10.BMP", 
    "dataset/dataset_blood_group/O-/cluster_7_1002.BMP",
    "dataset/dataset_blood_group/AB+/cluster_4_4906.BMP"
]

print("🧪 TESTING MULTIPLE FINGERPRINT IMAGES")
print("=" * 50)

for img_path in images:
    if os.path.exists(img_path):
        print(f"\n🧪 Testing: {os.path.basename(img_path)}")
        print("-" * 40)
        result = subprocess.run(["python", "predict.py", img_path], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    else:
        print(f"❌ File not found: {img_path}")

print("\n" + "=" * 50)
print("✅ Testing completed!")
