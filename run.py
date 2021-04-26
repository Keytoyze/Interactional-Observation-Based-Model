import json
import argparse
import os
import numpy as np
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Yahoo_letor', choices=['Yahoo_letor', 'MSLR_30k_letor'])
parser.add_argument('--click_setting', type=str, default='PBM', choices=['PBM', 'UBM', 'BDCM'])
parser.add_argument('--context_level', type=float, default=0)
parser.add_argument('--dependency_level', type=float, default=1)
parser.add_argument('--framework', type=str, default='DLA', choices=['DLA', 'RegressionEM'])

parser.add_argument('--propensity_lr', type=float, default=0.3)
parser.add_argument('--propensity_model', type=str, default='IOBM', choices=['IOBM', 'PBM', 'CPBM', 'UBM', 'DCM', 'ClickData', 'LabeledData'])
parser.add_argument('--propensity_model_param', type=str, default="")
parser.add_argument('--propensity_l2_loss', type=float, default=0)

parser.add_argument('--ranking_lr', type=float, default=0.005)
parser.add_argument('--ranking_model', type=str, default="DNN", choices=['DNN', 'SetRank', 'Linear'])
parser.add_argument('--ranking_model_param', type=str, default="")

parser.add_argument('--step', type=int, default=15000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='sgd')

parser = parser.parse_args()

def main():
    # work path
    base_dir = "temp"
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir(base_dir)
    settings_path = os.path.join(base_dir, "settings.json")
    click_path = os.path.join(base_dir, "click.json")

    # basic params
    params = {
        "max_train_iteration": parser.step,
        "data_dir": "./" + parser.dataset + "/tmp_data/",
        "batch_size": parser.batch_size,
        "setting_file": settings_path,
        "model_dir": os.path.join(base_dir, "model_dir") + "/",
        "output_dir": os.path.join(base_dir, "output_dir") + "/"
    }

    generate_click_setting(params['data_dir'], click_path)
    generate_model_setting(settings_path, click_path)

    params['setting_file'] = settings_path
    params['model_dir'] = os.path.join(base_dir, "model_dir") + "/"
    params['output_dir'] = os.path.join(base_dir, "output_dir") + "/"

    command = "python3 main.py " + " ".join(
        ["--" + key + "=" + str(value) for key, value in params.items()])
    print(open(click_path).read())
    print(open(settings_path).read())
    print(command)

    os.system(command)

def generate_model_setting(settings_path, click_path):
    settings = {
        "train_input_feed": "ultra.input_layer.ClickSimulationFeed",
        "valid_input_feed": "ultra.input_layer.ClickSimulationFeed",
        "test_input_feed": "ultra.input_layer.ClickSimulationFeed",
        "metrics": [
            "mrr", "ndcg"
        ],
        "metrics_topn": [1, 3, 5, 10],
        "objective_metric": "ndcg_10"
    }
    settings['ranking_model'] = "ultra.ranking_model." + parser.ranking_model
    settings['learning_algorithm'] = "ultra.learning_algorithm." + parser.framework
    settings['train_input_hparams'] = "click_model_json=" + click_path
    settings["test_input_hparams"] = "click_model_json=" + click_path + ",forward_only=true"
    settings['valid_input_hparams'] = "click_model_json=" + click_path + ",forward_only=true"
    settings["learning_algorithm_hparams"] = "learning_rate=%.8lf" % (parser.ranking_lr) + \
                                            ",propensity_learning_rate=%.8lf" % (parser.propensity_lr) + \
                                            ",max_propensity_weight=200" + \
                                            ",loss_func=click_weighted_softmax_cross_entropy" + \
                                            ",propensity_loss_func=click_weighted_log_loss" + \
                                            ",grad_strategy=" + parser.optimizer + \
                                            ",propensity_l2_loss=%.8lf" % parser.propensity_l2_loss + \
                                            ",oracle=" + str(parser.propensity_model == 'LabeledData')
    settings["propensity_model"] = "ultra.propensity_model." + parser.propensity_model
    settings["propensity_model_hparams"] = parser.propensity_model_param
    settings["ranking_model_hparams"] = parser.ranking_model_param
    with open(settings_path, "w") as f:
        json.dump(settings, f)

def generate_click_setting(data_dir, click_path):
    # click parameters
    click_weight = {
        'PBM': [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06],
        'UBM': [
            [1.0],
            [0.98, 1.0],
            [1.0, 0.62, 0.95],
            [1.0, 0.77, 0.42, 0.82],
            [1.0, 0.92, 0.55, 0.31, 0.69],
            [1.0, 0.96, 0.63, 0.4, 0.22, 0.54],
            [1.0, 0.99, 0.73, 0.46, 0.29, 0.17, 0.47],
            [1.0, 1.0, 0.89, 0.52, 0.35, 0.24, 0.14, 0.43],
            [1.0, 1.0, 0.95, 0.68, 0.4, 0.29, 0.19, 0.12, 0.41],
            [1.0, 1.0, 1.0, 0.96, 0.52, 0.36, 0.27, 0.18, 0.12, 0.43]
        ],
        'BDCM': [(1 / (j + 1)) for j in range(10)]
    }
    click_full_name = {
        'PBM': "position_biased_model",
        'UBM': "user_browsing_model",
        'BDCM': "bidirection_dcm"
    }

    # click settings
    exam_prob = click_weight[parser.click_setting]
    model_name = click_full_name[parser.click_setting]

    # click dependency level (only valid for UBM)
    if parser.click_setting == "UBM":
        exam_prob = [
            [x * parser.dependency_level + click_weight['PBM'][i] * (1 - parser.dependency_level)
                for x in exam_prob[i]]
            for i in range(len(exam_prob))
        ]

    # context dependency level (only valid for PBM)
    settings_json = json.load(open(os.path.join(data_dir, "settings.json")))
    feature_size = settings_json['feature_size'] # MSLR: 136, yahoo: 700
    max_label = int(settings_json['max_label']) + 1 # == 5
    context_full_w = np.zeros((feature_size,))
    if parser.click_setting == "PBM":
        importance_feature_size = 10
        feature_index = json.load(open(os.path.join(data_dir, "importance.json")))[:importance_feature_size]
        context_w = (np.random.rand(importance_feature_size) - 0.5) * parser.context_level * 2
        context_w -= np.mean(context_w)
        for i in range(importance_feature_size):
            context_full_w[feature_index[i]] = context_w[i]

    # click probability
    epsilon = 0.1
    click_prob = [epsilon + (1 - epsilon) * (2 ** i - 1) / (2 ** (max_label - 1) - 1) for i in
                    range(max_label)]
    click_prob.append(1)

    click_config = {
        "click_prob": click_prob,
        "eta": 1.0,
        "exam_prob": exam_prob,
        "model_name": model_name,
        'context_w': context_full_w.tolist()
    }
    with open(click_path, "w") as f:
        json.dump(click_config, f)

if __name__ == '__main__':
    main()