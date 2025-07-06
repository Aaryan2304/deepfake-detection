# main.py
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch.nn.functional as F
from tqdm import tqdm

# Import from your existing and modified scripts
from deepfake_starter_code import DeepfakeDetector, DeepfakeTrainer, DeepfakeDataset, DeepfakeExplainer, get_transforms
from data_utils import DFDCDataLoader, DeepfakeDatasetBuilder
from evaluation_ethics import ComprehensiveEvaluator, EthicsAnalyzer

# --- Configuration ---
DATASET_DIR = Path('./DFDC')  # IMPORTANT: Point this to your DFDC dataset directory
PROCESSED_DATA_DIR = Path('./processed_data')
SPLITS_DIR = PROCESSED_DATA_DIR / 'splits'
OUTPUT_DIR = Path('./output')
MODEL_PATH = OUTPUT_DIR / 'best_deepfake_model.pth'
ETHICS_REPORT_PATH = OUTPUT_DIR / 'ethics_report.md'
SUBMISSION_PATH = OUTPUT_DIR / 'submission.csv'
SAMPLE_SUBMISSION_PATH = DATASET_DIR / 'sample_submission.csv'

# Model and training settings
BACKBONE = 'efficientnet'
EPOCHS = 10 # Keep it low for a quick run, increase for better performance
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_submission(model, device, test_video_dir, sample_submission_path, output_path, transform):
    """
    Processes videos in the test set, makes predictions, and generates a submission file.
    """
    print("\n--- Step 6: Generating Submission File for Competition Data ---")
    model.eval()
    
    submission_df = pd.read_csv(sample_submission_path)
    predictions = []
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    with torch.no_grad():
        for filename in tqdm(submission_df['filename'], desc="Generating predictions"):
            video_path = str(Path(test_video_dir) / filename)
            
            if not Path(video_path).exists():
                # If video is not found, predict 0.5 as a neutral default
                predictions.append(0.5)
                continue

            # --- Frame and Face Extraction ---
            cap = cv2.VideoCapture(video_path)
            video_frames = []
            frame_count = 0
            while cap.isOpened() and len(video_frames) < 30: # Extract up to 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 3 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        face_crop = frame[y:y+h, x:x+w]
                        face_crop_resized = cv2.resize(face_crop, (224, 224))
                        rgb_frame = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
                        
                        # Transform and add to list
                        transformed_frame = transform(image=rgb_frame)['image']
                        video_frames.append(transformed_frame)
                frame_count += 1
            cap.release()
            
            # --- Prediction ---
            if not video_frames:
                # If no faces found, predict 0.5
                video_prediction = 0.5
            else:
                video_tensor = torch.stack(video_frames).to(device)
                outputs = model(video_tensor)
                probs = F.softmax(outputs, dim=1)[:, 1] # Probability of being 'FAKE'
                video_prediction = probs.mean().item()
                
            predictions.append(video_prediction)

    submission_df['label'] = predictions
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission file saved to {output_path}")


def main():
    """Main function to run the deepfake detection pipeline."""

    # --- 1. Data Preprocessing ---
    print("--- Step 1: Data Preprocessing ---")
    if not SPLITS_DIR.exists() or not any(SPLITS_DIR.iterdir()):
        print("Processed data not found. Starting preprocessing...")
        data_loader = DFDCDataLoader(root_dir=str(DATASET_DIR))
        # Using a subset of videos for a quicker run. Remove `max_videos` for the full dataset.
        dataset_info = data_loader.create_dataset_from_videos(output_dir=str(PROCESSED_DATA_DIR), max_videos=400)

        builder = DeepfakeDatasetBuilder(dataset_info)
        train_df, val_df, test_df = builder.create_splits()
        builder.save_splits(str(SPLITS_DIR), train_df, val_df, test_df)
    else:
        print("Found existing processed data. Loading splits.")
        train_df = pd.read_csv(SPLITS_DIR / 'train.csv')
        val_df = pd.read_csv(SPLITS_DIR / 'val.csv')
        test_df = pd.read_csv(SPLITS_DIR / 'test.csv')

    # --- 2. Model Training ---
    print("\n--- Step 2: Model Training ---")
    model = DeepfakeDetector(backbone=BACKBONE, num_classes=2)
    train_transform, val_transform = get_transforms()

    train_dataset = DeepfakeDataset(image_paths=train_df['path'].values, labels=train_df['label'].values, transform=train_transform)
    val_dataset = DeepfakeDataset(image_paths=val_df['path'].values, labels=val_df['label'].values, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if not MODEL_PATH.exists():
        print("No trained model found. Starting training...")
        trainer = DeepfakeTrainer(model, train_loader, val_loader, device=DEVICE)
        trainer.train(epochs=EPOCHS)
        # Rename the saved model for clarity
        Path('best_deepfake_model.pth').rename(MODEL_PATH)
    else:
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    
    model.to(DEVICE)

    # --- 3. Model Evaluation ---
    print("\n--- Step 3: Model Evaluation on Labeled Test Set---")
    test_dataset = DeepfakeDataset(image_paths=test_df['path'].values, labels=test_df['label'].values, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    evaluator = ComprehensiveEvaluator(model, device=DEVICE)
    evaluation_results = evaluator.evaluate_model(test_loader)

    print("\nEvaluation Metrics:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
    print(f"  False Positive Rate: {evaluation_results['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {evaluation_results['false_negative_rate']:.4f}")

    evaluator.plot_evaluation_results(evaluation_results, save_dir=str(OUTPUT_DIR))

    # --- 4. Explainability ---
    print("\n--- Step 4: Explainability (Grad-CAM) ---")
    explainer = DeepfakeExplainer(model, device=DEVICE)

    # Get a sample fake image from the test set for visualization
    fake_images = test_df[test_df['label'] == 1]
    if not fake_images.empty:
        sample_image_path = fake_images.iloc[0]['path']
        original_image = cv2.imread(sample_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Preprocess the image for the model
        transformed = val_transform(image=original_image)
        image_tensor = transformed['image']

        grayscale_cam = explainer.explain_prediction(image_tensor, target_class=1) # Target class 1 is 'Fake'
        explainer.visualize_explanation(
            np.array(original_image) / 255.0,
            grayscale_cam,
            save_path=str(OUTPUT_DIR / 'grad_cam_example.png')
        )
        print(f"Grad-CAM visualization saved to {OUTPUT_DIR / 'grad_cam_example.png'}")
    else:
        print("No fake images found in the test set to generate Grad-CAM for.")

    # --- 5. Ethical Analysis ---
    print("\n--- Step 5: Ethical Analysis ---")
    ethics_analyzer = EthicsAnalyzer(model, device=DEVICE)
    # Note: Demographic analysis is skipped as we don't have demographic labels in DFDC.
    # The report will highlight this limitation.
    report = ethics_analyzer.generate_ethics_report(evaluation_results, save_path=str(ETHICS_REPORT_PATH))
    print(f"Ethics report generated and saved to {ETHICS_REPORT_PATH}")

    # --- 6. Inference for Submission ---
    test_video_dir = DATASET_DIR / 'test_videos'
    if test_video_dir.exists() and SAMPLE_SUBMISSION_PATH.exists():
        generate_submission(model, DEVICE, test_video_dir, SAMPLE_SUBMISSION_PATH, SUBMISSION_PATH, val_transform)
    else:
        print("\nSkipping submission file generation: `test_videos` folder or `sample_submission.csv` not found.")


if __name__ == "__main__":
    main()