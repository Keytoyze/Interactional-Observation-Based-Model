import ultra
from ultra.input_layer.click_simulation_feed import ClickSimulationFeed
from ultra.utils.data_utils import Raw_data

def calculate(click_feed: ClickSimulationFeed, dataset: Raw_data):
    distribution = {} # { <qid: Int, pos: Int> -> { binary: Int ->  } }

    # for i in range(len(dataset.initial_list)):
    i = 0
    for epoch in range(128):
        input_feed, other_feed = click_feed.get_data_by_index(data_set, i, False)