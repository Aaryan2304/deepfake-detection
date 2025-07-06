# evaluation_ethics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for deepfake detection models"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_model(self, test_loader, class_names=['Real', 'Fake']) -> Dict[str, any]:
        """Comprehensive model evaluation"""

        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_features = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Get model outputs
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                # Extract features from backbone (if available)
                if hasattr(self.model, 'backbone'):
                    features = self.model.backbone(data)
                    all_features.extend(features.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)

        # Additional analysis
        analysis = {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'features': np.array(all_features) if all_features else None,
            'class_names': class_names
        }

        return {**metrics, **analysis}

    def _calculate_metrics(self, targets, predictions, probabilities) -> Dict[str, float]:
        """Calculate comprehensive metrics"""

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, average='weighted')
        metrics['recall'] = recall_score(targets, predictions, average='weighted')
        metrics['f1_score'] = f1_score(targets, predictions, average='weighted')

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(targets, predictions, average=None)
        metrics['recall_per_class'] = recall_score(targets, predictions, average=None)
        metrics['f1_per_class'] = f1_score(targets, predictions, average=None)

        # ROC-AUC
        if probabilities.shape[1] == 2:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(targets, probabilities[:, 1])
        else:
            metrics['roc_auc'] = roc_auc_score(targets, probabilities, multi_class='ovr')

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)

        # False positive and false negative rates
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        return metrics

    def plot_evaluation_results(self, results: Dict[str, any], save_dir: str = None):
        """Generate comprehensive evaluation plots"""

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))

        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 4, 1)
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=results['class_names'], yticklabels=results['class_names'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # 2. ROC Curve
        ax2 = plt.subplot(3, 4, 2)
        if results['probabilities'].shape[1] == 2:
            fpr, tpr, _ = roc_curve(results['targets'], results['probabilities'][:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        # 3. Precision-Recall Curve
        ax3 = plt.subplot(3, 4, 3)
        if results['probabilities'].shape[1] == 2:
            precision, recall, _ = precision_recall_curve(results['targets'], results['probabilities'][:, 1])
            plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)

        # 4. Calibration Plot
        ax4 = plt.subplot(3, 4, 4)
        if results['probabilities'].shape[1] == 2:
            prob_true, prob_pred = calibration_curve(results['targets'], results['probabilities'][:, 1], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True)

        # 5. Prediction Confidence Distribution
        ax5 = plt.subplot(3, 4, 5)
        correct_mask = results['predictions'] == results['targets']

        plt.hist(results['probabilities'][correct_mask, 1], alpha=0.7, bins=30, label='Correct', density=True)
        plt.hist(results['probabilities'][~correct_mask, 1], alpha=0.7, bins=30, label='Incorrect', density=True)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()

        # 6. Per-Class Performance
        ax6 = plt.subplot(3, 4, 6)
        class_metrics = ['precision_per_class', 'recall_per_class', 'f1_per_class']
        x = np.arange(len(results['class_names']))
        width = 0.25

        for i, metric in enumerate(class_metrics):
            values = results[metric]
            plt.bar(x + i*width, values, width, label=metric.replace('_per_class', '').title())

        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Performance')
        plt.xticks(x + width, results['class_names'])
        plt.legend()

        # 7. Error Analysis
        ax7 = plt.subplot(3, 4, 7)
        error_types = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
        tn, fp, fn, tp = results['confusion_matrix'].ravel()
        error_counts = [tp, tn, fp, fn]
        colors = ['green', 'lightgreen', 'orange', 'red']

        plt.pie(error_counts, labels=error_types, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Error Distribution')

        # 8. Feature Distribution (if available)
        if results['features'] is not None:
            ax8 = plt.subplot(3, 4, 8)
            # Plot first two principal components or first two features
            real_features = results['features'][results['targets'] == 0]
            fake_features = results['features'][results['targets'] == 1]

            if real_features.shape[0] > 0 and fake_features.shape[0] > 0:
                plt.scatter(real_features[:, 0], real_features[:, 1], alpha=0.6, label='Real', s=10)
                plt.scatter(fake_features[:, 0], fake_features[:, 1], alpha=0.6, label='Fake', s=10)
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title('Feature Space Distribution')
                plt.legend()

        # 9. Model Certainty Analysis
        ax9 = plt.subplot(3, 4, 9)
        certainty_scores = np.max(results['probabilities'], axis=1)

        # Bin by certainty level
        certainty_bins = np.linspace(0.5, 1.0, 6)
        bin_accuracies = []
        bin_counts = []

        for i in range(len(certainty_bins) - 1):
            mask = (certainty_scores >= certainty_bins[i]) & (certainty_scores < certainty_bins[i+1])
            if mask.sum() > 0:
                bin_accuracy = accuracy_score(results['targets'][mask], results['predictions'][mask])
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        plt.bar(range(len(bin_accuracies)), bin_accuracies, alpha=0.7)
        plt.xlabel('Certainty Bin')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Model Certainty')
        plt.xticks(range(len(bin_accuracies)), [f'{certainty_bins[i]:.1f}-{certainty_bins[i+1]:.1f}'
                                              for i in range(len(bin_accuracies))])

        # 10. Metrics Summary
        ax10 = plt.subplot(3, 4, 10)
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_values = [results[m] for m in metrics_to_show]

        plt.barh(metrics_to_show, metric_values, color='skyblue')
        plt.xlabel('Score')
        plt.title('Overall Metrics Summary')
        plt.xlim(0, 1)

        # Add value labels on bars
        for i, v in enumerate(metric_values):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')

        plt.show()

        return fig

class EthicsAnalyzer:
    """Analyze ethical implications and biases in deepfake detection"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def analyze_demographic_bias(self, data_loader, demographic_labels=None) -> Dict[str, any]:
        """Analyze performance across different demographic groups"""

        if demographic_labels is None:
            logger.warning("No demographic labels provided. Bias analysis will be limited.")
            return {}

        results = {}

        # Get model predictions
        all_predictions = []
        all_probabilities = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Analyze by demographic groups
        unique_groups = np.unique(demographic_labels)
        for group in unique_groups:
            group_mask = np.array(demographic_labels) == group
            group_predictions = np.array(all_predictions)[group_mask]
            group_targets = np.array(all_targets)[group_mask]
            group_probabilities = np.array(all_probabilities)[group_mask]

            if len(group_predictions) > 0:
                results[group] = {
                    'accuracy': accuracy_score(group_targets, group_predictions),
                    'precision': precision_score(group_targets, group_predictions, average='weighted'),
                    'recall': recall_score(group_targets, group_predictions, average='weighted'),
                    'f1_score': f1_score(group_targets, group_predictions, average='weighted'),
                    'sample_count': len(group_predictions),
                    'false_positive_rate': self._calculate_fpr(group_targets, group_predictions),
                    'false_negative_rate': self._calculate_fnr(group_targets, group_predictions)
                }

        return results

    def _calculate_fpr(self, targets, predictions):
        """Calculate False Positive Rate"""
        cm = confusion_matrix(targets, predictions)
        tn, fp, _, _ = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    def _calculate_fnr(self, targets, predictions):
        """Calculate False Negative Rate"""
        cm = confusion_matrix(targets, predictions)
        _, _, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        return fn / (fn + tp) if (fn + tp) > 0 else 0

    def fairness_metrics(self, demographic_results: Dict[str, any]) -> Dict[str, float]:
        """Calculate fairness metrics"""

        if len(demographic_results) < 2:
            return {}

        groups = list(demographic_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate', 'false_negative_rate']

        fairness_metrics = {}

        for metric in metrics:
            values = [demographic_results[group][metric] for group in groups if metric in demographic_results[group]]

            if len(values) > 1:
                # Statistical parity difference
                fairness_metrics[f'{metric}_max_diff'] = max(values) - min(values)

                # Equalized odds (for FPR and FNR)
                if 'false_positive' in metric or 'false_negative' in metric:
                    fairness_metrics[f'{metric}_equalized_odds'] = np.std(values)

        return fairness_metrics

    def generate_ethics_report(self, evaluation_results: Dict[str, any],
                             demographic_results: Dict[str, any] = None,
                             save_path: str = None) -> str:
        """Generate comprehensive ethics report"""

        report = []
        report.append("# Deepfake Detection Model - Ethics and Bias Analysis Report")
        report.append("=" * 60)
        report.append("")

        # Model performance summary
        report.append("## Model Performance Summary")
        report.append(f"- Overall Accuracy: {evaluation_results['accuracy']:.4f}")
        report.append(f"- Precision: {evaluation_results['precision']:.4f}")
        report.append(f"- Recall: {evaluation_results['recall']:.4f}")
        report.append(f"- F1-Score: {evaluation_results['f1_score']:.4f}")
        report.append(f"- ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        report.append("")

        # Error analysis
        report.append("## Error Analysis")
        report.append(f"- False Positive Rate: {evaluation_results['false_positive_rate']:.4f} (Real content classified as Fake)")
        report.append(f"- False Negative Rate: {evaluation_results['false_negative_rate']:.4f} (Fake content classified as Real)")
        report.append("")

        # Ethical considerations
        report.append("## Ethical Considerations")
        report.append("")

        report.append("### 1. Bias and Fairness")
        if demographic_results:
            report.append("**Demographic Performance Analysis:**")
            for group, metrics in demographic_results.items():
                report.append(f"- **{group}**: Accuracy={metrics['accuracy']:.4f}, FPR={metrics['false_positive_rate']:.4f}, FNR={metrics['false_negative_rate']:.4f}")

            fairness_metrics = self.fairness_metrics(demographic_results)
            if fairness_metrics:
                report.append("\n**Fairness Metrics:**")
                for metric, value in fairness_metrics.items():
                    report.append(f"- {metric}: {value:.4f}")
        else:
            report.append("- **Limitation**: Demographic analysis was not performed. This is a critical gap.")
            report.append("- **Recommendation**: For any real-world deployment, it is crucial to collect or annotate data with demographic information (e.g., gender, race, age) to test for performance disparities across different groups. Without this, the model may perform unfairly on underrepresented groups.")

        report.append("")
        report.append("### 2. Privacy and Data Protection")
        report.append("- The model was trained on the DFDC dataset, which contains videos of individuals who consented to be part of the dataset.")
        report.append("- **Recommendation**: When using this model on new data, ensure that privacy is respected. If processing user-submitted content, have clear data handling and privacy policies. Avoid storing data unnecessarily.")
        report.append("")

        report.append("### 3. Potential Misuse and Societal Impact")
        report.append("- **False Positives**: Incorrectly flagging real content as fake can lead to censorship, damage reputations, and suppress legitimate expression. The False Positive Rate (FPR) should be closely monitored.")
        report.append("- **False Negatives**: Failing to detect deepfakes allows malicious content like misinformation, propaganda, or non-consensual pornography to spread, causing significant harm.")
        report.append("- **Dual-Use Nature**: While designed for detection, insights from this model could potentially be used by malicious actors to create more convincing deepfakes that evade detection (adversarial attacks).")
        report.append("")

        report.append("### 4. Transparency and Explainability")
        report.append("- Explainability methods like Grad-CAM are used to provide insights into model decisions, showing which parts of an image are influential.")
        report.append("- **Recommendation**: Always provide explanations alongside predictions where possible. This builds user trust and helps in debugging cases where the model is wrong.")
        report.append("")

        report.append("### 5. Adversarial Robustness")
        report.append("- **Vulnerability**: Like most deep learning models, this detector is likely vulnerable to adversarial attacks, where small, imperceptible changes to an image can cause misclassification.")
        report.append("- **Recommendation**: Future work should include testing the model against various adversarial attacks (e.g., FGSM, PGD) and exploring defenses like adversarial training to improve robustness.")
        report.append("")

        # Final Recommendations
        report.append("## Overall Recommendations & Responsible Deployment")
        report.append("- **Human-in-the-loop**: This model should be used as a tool to assist human moderators, not as a fully autonomous decision-maker.")
        report.append("- **Context is Key**: Decisions about content should not be based solely on the model's output but should consider the broader context of the content.")
        report.append("- **Continuous Monitoring**: The deepfake landscape evolves rapidly. The model must be continuously monitored and retrained on new data to remain effective.")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)

        return report_str