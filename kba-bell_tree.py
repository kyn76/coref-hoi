from run import Runner
import sys


def bell_tree_evaluation(config_name, gpu_id, saved_suffix, intra_aggregations, nbs_beams):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()
    
    verbose = False
    runner.bell_tree_process(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'], intra_aggregations=intra_aggregations, nbs_beams=nbs_beams, verbose=verbose)


if __name__ == '__main__':
    config_name, saved_suffix, intra_aggregations, nbs_beams = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    if len(sys.argv) < 6:
        gpu_id = None
    else:
        gpu_id = int(sys.argv[5])

    intra_aggregations = intra_aggregations.split("[")[1].split("]")[0].split(",")
    nbs_beams = nbs_beams.split("[")[1].split("]")[0].split(",")
    nbs_beams = list(map(int, nbs_beams))

    for intra_aggregation in intra_aggregations:
        assert(intra_aggregation in ["max", "avg"])
    bell_tree_evaluation(config_name, gpu_id, saved_suffix, intra_aggregations, nbs_beams)
