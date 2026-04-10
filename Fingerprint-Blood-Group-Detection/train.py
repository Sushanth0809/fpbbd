import argparse
import logging
from data.dataset import create_datasets
from models.hybrid_model import HybridMultiModalNet
from evaluation.evaluate import evaluate_model, print_evaluation_report
from evaluation.visualize import plot_training_curves, plot_comparison_bars

def main():
    parser = argparse.ArgumentParser(description="Train Fingerprint Blood Group Detection Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode with fewer epochs')

    args = parser.parse_args()

    # Setup
    set_seed(RANDOM_SEED)
    setup_logging()

    if args.quick_test:
        args.epochs = 3
        logging.info("Quick test mode: training for 3 epochs")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets()

    # Create model
    model = HybridMultiModalNet()

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset)

    # Train
    history = trainer.train(num_epochs=args.epochs)

    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_metrics = evaluate_model(trainer.model, test_dataset, trainer.device)
    print_evaluation_report(test_metrics)

    # Generate visualizations
    logging.info("Generating training curves...")
    plot_training_curves(
        history['train_losses'], history['val_losses'],
        history['train_abo_accs'], history['val_abo_accs'],
        history['train_rh_accs'], history['val_rh_accs']
    )

    logging.info("Generating comparison charts...")
    plot_comparison_bars(test_metrics)

    logging.info("Training and evaluation completed!")

if __name__ == "__main__":
    main()