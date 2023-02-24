from run import Runner
import sys


def evaluate(config_name, gpu_id, saved_suffix, boundaries, k):
    assert(boundaries in ["predicted", "gold"])
    assert k > 0
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    gold_boundaries = (boundaries == "gold")
    runner.evaluate_from_csv(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'], gold_boundaries=gold_boundaries, k=k)


if __name__ == '__main__':
    config_name, saved_suffix, boundaries, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    if len(sys.argv) < 6:
        gpu_id = None
    else:
        gpu_id = int(sys.argv[5])
    
    evaluate(config_name, gpu_id, saved_suffix, boundaries, k)
