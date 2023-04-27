import torch
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils import comm


class FastercnnTrainer(DefaultTrainer):

    def __init__(self, cfg, early_stop_patience=5):
        super().__init__(cfg)
        self.early_stop_patience = early_stop_patience
        self.min_loss = float('inf')
        self.patience_counter = 0

    def after_step(self):
        super().after_step()

        # Early stopping condition
        if (self.storage.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
            val_loader = iter(build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST,
                                                          mapper=DatasetMapper(self.cfg, is_train=True)))
            val_loss = self.compute_validation_loss(val_loader)

            print(f"\033[32mValidation Loss: {val_loss}\033[0m")

            if val_loss < self.min_loss:
                self.min_loss = val_loss
                self.patience_counter = 0
                self.checkpointer.save("best_model")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stop_patience:
                # Evaluate the model on the test dataset and print the results
                evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
                val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
                inference_on_dataset(self.model, val_loader, evaluator)
                raise RuntimeError('Early stopping triggered')

    def compute_validation_loss(self, val_loader):
        total_loss = 0.0
        num_batches = len(val_loader)  # Calculate the number of batches in the validation loader
        # Iterate through the batches in the validation loader
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                loss_dict = self.model(data)  # Pass the data through the model and compute the loss dictionary
                losses = sum(
                    loss_dict.values())  # Sum the losses in the loss dictionary to get the total loss for the current batch
                assert torch.isfinite(
                    losses).all(), loss_dict  # Check if the computed loss values are finite and raise an exception with the loss dictionary if not
                total_loss += losses.item()  # Add the total loss of the current batch to the total loss across all batches

                # If the current process is the main process, log individual losses for the last batch
                if comm.is_main_process():
                    if i == num_batches - 1:
                        # Create a dictionary with the reduced individual losses and prefix keys with "val_"
                        loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                        # Log the individual losses using the storage object
                        self.storage.put_scalars(**loss_dict_reduced)

        # If the current process is the main process, log the total loss across all batches
        if comm.is_main_process():
            self.storage.put_scalar("val_total_loss", total_loss)

        return total_loss
