
def evaluate_model(model_path, test_loader, device):
    """Comprehensive model evaluation"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Load model
    inference_engine = InferenceEngine(model_path, device)
    
    # Get predictions
    predictions, confidences, targets = inference_engine.predict_batch(test_loader)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
    # ROC-AUC (need probabilities for both classes)
    # For binary classification, we can use the probability of positive class
    # Get probabilities for all test data
    all_probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = inference_engine.model(inputs)
            else:
                outputs = inference_engine.model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(targets, all_probs[:, 1])
    
    # Print results
    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Crack', 'Crack'],
                yticklabels=['No Crack', 'Crack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(targets, predictions, 
                                target_names=['No Crack', 'Crack']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
