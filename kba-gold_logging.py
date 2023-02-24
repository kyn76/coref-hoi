from run import Runner
import sys


def gold_logging(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    runner.csv_logging(model, examples_test, stored_info, 0, logging="gold")


if __name__ == '__main__':
    config_name, saved_suffix = sys.argv[1], sys.argv[2]
    if len(sys.argv) < 4:
        gpu_id = None
    else:
        gpu_id = int(sys.argv[3])
        
    gold_logging(config_name, gpu_id, saved_suffix)
