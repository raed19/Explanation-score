data:
  train_csv: data/sign_mnist_train.csv
  test_csv:  data/sign_mnist_test.csv
  img_size: 224
  batch_size: 32
  val_split: 0.2
  seed: 42

train:
  epochs: 7
  lr: 0.001
  patience: 5
  num_classes: 25

paths:
  model_out: outputs/models/mobilenetv2_finetuned.pth
  plots_dir: outputs/plots
