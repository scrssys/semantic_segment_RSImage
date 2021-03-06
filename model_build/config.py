from collections import namedtuple


Config = namedtuple("Config", [
  "train_data_path",
  "img_w",
  "img_h",
  "im_bands",
  "im_type",
  "target_name",
  "val_rate",
  "network",
  "BACKBONE",
  "activation",
  "encoder_weights",
  "nb_classes",
  "batch_size",
  "epochs",
  "optimizer",
  "loss",
  "metrics",
  "lr",
  "lr_steps",
  "lr_gamma",
  "lr_scheduler",
  "nb_epoch",
  "old_epoch",
  "test_pad",
  "model_dir",
  "base_model",
  "monitor",
  "save_best_only",
  "mode",
  "factor",
  "patience",
  "epsilon",
  "cooldown",
  "min_lr",
  "log_dir",
  "iter_size",
  "folder",
  "predict_batch_size",
  "results_dir",
  "loss_x",
  "loss_ema",
  "loss_kl",
  "loss_student",
  "ignore_target_size",
  "warmup",
  "lovasz",
  "unlabel_size",
  "negative_rate",
  "external_weights",
  "class_weights",
  "train_fold",
  "use_lb",
  "train_images",
  "train_images_unlabel",
  "train_masks",
  "train_image_suffix",
  "csv_file",
  "num_workers",
  "use_border",
  "class_picture",
  "use_good_image",
  "hard_negative_miner",
  "mixup"
])


