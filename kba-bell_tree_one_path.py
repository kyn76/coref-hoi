from run import Runner
import sys


def bell_tree_evaluation(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    runner.bell_tree_process(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'])


if __name__ == '__main__':
    config_name, saved_suffix = sys.argv[1], sys.argv[2]
    if len(sys.argv) < 5:
        gpu_id = None
    else:
        gpu_id = int(sys.argv[4])
    
    bell_tree_evaluation(config_name, gpu_id, saved_suffix)
