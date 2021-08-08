# ml_final_project

This is a repo containing code for a final project in Machine Learning course.

## Running 

The project is written as a python package thus to run it you have to install it, and then run it:
```shell
pip install <project-location>
python -m ml_final_project.run_simulation -d <csv-input-path> -o <csv-output-path> -n <network-configuration>  
```

### The arguments:
- `-d <csv-input-path` - this is a path to a csv that contain the dataset such that the class is the last column.
- `-o <csv-output-path` - this is a path to a csv that the package will create with the results.
- `-n <network-configuration>` - a json string that contains configurations for the created neural network. These are:
  - `inner_dim` - a parameter that control the width of the hidden layers.
  - `num_inner_layers` - a parameter that control the number of the hidden layers (depth).
  - `batch_size` - the batch size used for training.
  - `epochs` - the number of epochs for training.
  - `val` - if not `1` will not use any validation set. if `1` will split 20% of the training data for validation and will stop earlier if validation stops improving. 