import os
import subprocess

# 🔧 CUSTOMIZE THIS LIST WITH YOUR OWN IMAGE PATHS
# Replace these paths with your actual fingerprint images
images = [
    "/path/to/your/fingerprint1.jpg",    # Your first image
    "/path/to/your/fingerprint2.png",    # Your second image
    "/path/to/your/fingerprint3.bmp",    # Your third image
    # Add more paths as needed
]

print("🧪 TESTING YOUR OWN FINGERPRINT IMAGES")
print("=" * 50)
print("📝 Instructions:")
print("   1. Replace the paths above with your actual image files")
print("   2. Make sure the images exist and are readable")
print("   3. Run this script: python test_your_own.py")
print("=" * 50)

found_count = 0
for img_path in images:
    if os.path.exists(img_path):
        found_count += 1
        print(f"\n🧪 Testing: {os.path.basename(img_path)}")
        print("-" * 40)
        try:
            result = subprocess.run(["python", "predict.py", img_path], 
                                  capture_output=True, text=True, timeout=30)
            print(result.stdout.strip())
            if result.stderr:
                print(f"⚠️  Warning: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("⏰ Timeout: Prediction took too long")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        print(f"❌ File not found: {img_path}")

print(f"\n{'='*50}")
if found_count == 0:
    print("❌ No valid image files found!")
    print("💡 Tip: Update the 'images' list above with your actual file paths")
else:
    print(f"✅ Tested {found_count} image(s) successfully!")
    print("🎯 Check the predictions above for your blood group results!")
